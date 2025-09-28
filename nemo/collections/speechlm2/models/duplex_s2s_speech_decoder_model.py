# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import json
import os
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id, deduplicate_results
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
    Safe for GPU execution.
    """
    B, T = tokens.shape
    tokens = tokens.clone()
    device = tokens.device

    # Find all EOS positions
    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    # Flattened indices of EOS tokens
    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]  # [N]
    eos_pos = eos_indices[:, 1]  # [N]
    new_pos = eos_pos + shift  # [N]

    # Filter: new position must be in bounds and not overwrite EOS or PAD
    valid = (new_pos < T)
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        # Now, check overwrite safety in new positions
        target_vals = tokens[b_idx, new_pos]
        safe = (target_vals != eos_token_id)

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            # Move EOS token: clear original, set new
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        setup_audio_codec(self)
        self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # to be able to load older model
        if self.cfg.get("custom_codebook_size", None):
            self._codebook_size = self.cfg.get("custom_codebook_size")

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame
        # compute interpolation factor to interpolate
        self.interpolation_factor = self.target_fps / self.source_fps
        # x = torch.nn.functional.interpolate(x.unsqueeze(1), size=None, scale_factor=[1, self.interpolation_factor], mode='nearest-exact', align_corners=None, recompute_scale_factor=None, antialias=False)

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True, **self.cfg.get("override_tokens", {}))
        if 'Qwen2.5' in self.cfg.pretrained_llm:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.llm = getattr(llm, self.cfg.get("llm", {}).get("base_model_name", "model"))
        self.lm_head = llm.lm_head
        self.embed_tokens = getattr(self.llm, self.cfg.get("llm", {}).get("embeddings_name", "embed_tokens"))
        delattr(self.llm, self.cfg.get("llm", {}).get("embeddings_name", "embed_tokens"))
        
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder(self)

        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [
                (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                for subword in llm_tokenizer_vocab_items
            ]

        self.speech_generation = TransformerARSpeechDecoder(
            speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
            lantent_dim=self.llm.config.hidden_size,
            num_audio_codebooks=self._num_codebooks,
            num_audio_tokens_per_codebook=self.speech_vocab_size,
            llm_tokenizer_vocab_items=llm_tokenizer_vocab_items,
        )

        if self.cfg.get("pretrained_s2s_model", None):
            self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        # load pretrained TTS model
        if self.cfg.get("pretrained_tts", None):
            self.init_speech_generation_from_tts_checkpoint(self.cfg.pretrained_tts)

        # load speech decoder/speech generation module from another checkpoint
        if self.cfg.get("pretrained_tts_from_s2s", None):
            self.init_speech_generation_from_another_s2s_checkpoint(self.cfg.pretrained_tts_from_s2s)

        """
        self.embed_audio_tokens = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                for _ in range(self._num_codebooks)
            ]
        )
        self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)
        """

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )
        self._use_fsdp = self.cfg.get("use_fsdp", False)
        self._use_tp = False
        # Storage for collecting validation results across GPUs
        self.validation_results = defaultdict(list)

    def init_speech_generation_from_tts_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_speech_generation_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # filter keys to keep only speech generation keys and also
            checkpoint_state = {
                k.replace("model.speech_decoder.", "").replace("speech_generation.", ""): v
                for k, v in checkpoint_state.items()
                if "model.speech_decoder." in k or "speech_generation." in k
            }
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)
        self.use_silence_tokens = self.cfg.get("use_silence_tokens", False)
        if self.use_silence_tokens:
            self.silence_tokens = self.generate_silence_tokens(16000, self._num_codebooks)
            logging.info("Silence tokens enabled.")
        else:
            logging.info("Silence tokens disabled. Using speech_nosil_id for silence positions.")
        #self.silence_tokens = self.generate_silence_tokens(16000, self._num_codebooks)
    
    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_delay_id", None):
            return self.cfg.get("custom_speech_delay_id")
        return self._codebook_size + 2

    @property
    def speech_nosil_id(self) -> int:
        """Indicates speech when there is function calling in the text channel."""
        return self.speech_eos_id

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        input_audio_tokens=None,
        seq_mask=None,
        target_text_tokens=None,
        modality_adapter_emb=None,
        asr_emb=None,
        speaker_encoder_emb=None,
        llm_kwargs={},
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        kwargs = {
            "inputs_embeds": input_embeds,
            "return_dict": True,
        }
        kwargs.update(llm_kwargs)
        if cache is not None:
            kwargs['use_cache'] = True
            cache_key = self.cfg.get("llm", {}).get("cache_key", "past_key_values")
            kwargs[cache_key] = cache
        else:
            kwargs['use_cache'] = False
        out = self.llm(**kwargs)
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)
        
        if seq_mask is not None:
            # This is training Mode
            seq_mask = seq_mask[:, :, -1].reshape(seq_mask.size(0), seq_mask.size(1))
            # disable cache in training mode
            if self.speech_generation.use_input_cache:
                self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        # if inference time, uses the target text tokens sampled from the llm backbone
        if self.speech_generation.use_input_cache and not self.training:
            if self.cfg.get("inference_pad_boost", None):
                text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
            if self.cfg.get("inference_bos_boost", None):
                text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost
            if self.cfg.get("inference_eos_boost", None):
                text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost

            target_text_tokens = torch.argmax(text_logits, dim=-1).view(B, T).contiguous()

            if self.cfg.get('convert_pad_to_extra_id_on_speech_decoder', None):
                target_text_tokens[target_text_tokens == self.text_pad_id] = self.tokenizer.tokenizer._tokenizer.token_to_id("<|endoftext|>") # <|endoftext|> token id
        else:
            # Drop BOS tokens with per-token probability (augmentation)
            drop_bos_prob = getattr(self.cfg, "drop_text_bos_prob", 0.0)
            if drop_bos_prob > 0.0:
                bos_mask = (target_text_tokens == self.text_bos_id)
                # Generate random mask only for BOS positions
                drop_bos_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_bos_prob
                target_text_tokens = torch.where(bos_mask & drop_bos_mask, self.text_pad_id, target_text_tokens)

            # Drop EOS tokens with per-token probability (augmentation)
            drop_eos_prob = getattr(self.cfg, "drop_text_eos_prob", 0.0)
            if drop_eos_prob > 0.0:
                eos_mask = (target_text_tokens == self.text_eos_id)
                drop_eos_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_eos_prob
                target_text_tokens = torch.where(eos_mask & drop_eos_mask, self.text_pad_id, target_text_tokens)

        if input_audio_tokens is not None:
            if self.speech_generation.use_input_cache and not self.training:
                audio_logits, _ = self.speech_generation(
                    out['last_hidden_state'][:,-1:,:].transpose(0, 1),
                    seq_mask,
                    input_audio_tokens=input_audio_tokens,
                    target_text_tokens=target_text_tokens,
                    modality_adapter_emb=modality_adapter_emb[:,-1:,:],
                    asr_emb=asr_emb[:,-1:,:],
                    speaker_encoder_emb=speaker_encoder_emb,
                )
                audio_logits = audio_logits.view(B, 1, self._num_codebooks, self.speech_vocab_size)
            else:
                audio_logits, _ = self.speech_generation(
                    out['last_hidden_state'].transpose(0, 1),
                    seq_mask,
                    input_audio_tokens=input_audio_tokens,
                    target_text_tokens=target_text_tokens,
                    modality_adapter_emb=modality_adapter_emb,
                    asr_emb=asr_emb,
                    speaker_encoder_emb=speaker_encoder_emb,
                )
                audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)
        else:
            audio_logits = None

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out[cache_key]

        return ans

    def add_noise_to_batch(
        self,
        batch_audio,
        noise_folder,
        snr_db=20,
        noise_prob_scale_user=0.3,
        noise_prob_scale_user_min_snr=-15,
        noise_prob_scale_user_max_snr=24,
        snr_measure_dur=0.0,
        noise_resample=True,
        noise_prob_low_pass=0.1,
    ):

        batch_size, audio_length = batch_audio.shape

        import glob

        import librosa
        import numpy as np
        import soundfile as sf
        from scipy.signal import butter, lfilter

        noise_files = [f for f in glob.glob(noise_folder + "/*.wav")]
        if not noise_files:
            raise ValueError(f"No noise files found in {noise_folder}")

        for i in range(batch_size):

            def get_scale_factor(signal, noise, snr_db):
                if snr_measure_dur > 0:
                    signal = signal[: int(snr_measure_dur * self.source_sample_rate)]
                    noise = noise[: int(snr_measure_dur * self.source_sample_rate)]
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(noise_prob_scale_user_min_snr, noise_prob_scale_user_max_snr),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor

            def get_noise(noise_files):

                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')

                # resample noise from sr to self.cfg.data.train_ds.sample_rate
                if noise_resample and sr != self.source_sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.source_sample_rate)

                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)

                noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)
                scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
                noise_tensor = noise_tensor * scaling_factor
                return noise_tensor

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise3 = get_noise(noise_files)
            noise = torch.cat([noise, noise2, noise3], axis=0)

            if noise.size(0) < audio_length:
                repeat_times = (audio_length // noise.size(0)) + 1
                # For a 1D tensor, we want to repeat its elements.
                # If noise has other dimensions, adjust the repeat_times_tuple accordingly.
                # e.g., if noise is (C, L), and we want to repeat along L,
                # repeat_times_tuple = (1, repeat_times)
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                # If noise is a PyTorch tensor
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                # Or if noise was originally a list/numpy array and you want to keep Python's random
                # start_idx = random.randint(0, len(noise) - audio_length)
                noise = noise[start_idx : start_idx + audio_length]

            # Function to create a low-pass filter
            def butter_lowpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            # Function to apply the low-pass filter to data (tmp impl on cpu)
            def lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                b = torch.tensor(b, dtype=torch.float32).cuda()
                a = torch.tensor(a, dtype=torch.float32).cuda()
                # Apply the filter using lfilter function from scipy..numpysig.numpynal (CPU)
                y_cpu = lfilter(b.cpu().numpy(), a.cpu().numpy(), data.cpu().numpy())
                # Convert the filtered data back to torch tensor and move to GPU.numpy
                y_gpu = torch.tensor(y_cpu, dtype=torch.float32).cuda()
                return y_gpu

            if random.random() < noise_prob_low_pass:
                # Define the desired cutoff frequency (in Hz)
                cutoff = 1000.0
                # Apply low-pass filter to the WAV data
                noise = lowpass_filter(noise, cutoff, self.source_sample_rate)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'seq_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        if 'target_audio' in batch:
            assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
            assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        if self.cfg.get('use_old_noise_aug', None):
            # ToDo we are applying it in all datasets, old codebase does not applied in real conv data
            noise_prob = 0.99
            noise_min_snr = 20
            noise_max_snr = 50
            noise_path = self.cfg.get(
                'old_noise_aug_path',
                None
            )
            noise_path_name = "*"
            no_noise_audio = batch["source_audio"].clone()
            if (
                self.training
                and batch["formatter"][0] != 's2s_duplex_overlap_as_s2s_duplex'
                and noise_prob
                and random.random() < noise_prob
            ):
                batch["source_audio"] = self.add_noise_to_batch(
                    batch["source_audio"],
                    os.path.join(noise_path, noise_path_name),
                    snr_db=random.randint(noise_min_snr, noise_max_snr),
                    noise_prob_scale_user=0.3,
                    noise_prob_scale_user_min_snr=-15,
                    noise_prob_scale_user_max_snr=24,
                    snr_measure_dur=0.0,
                    noise_resample=True,
                    noise_prob_low_pass=0.1,
                )
        else:
            # change audio volume randomly
            if self.training and random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                min_scale_val = self.cfg.get('noise_scale_user_min', 0.0631)  # -15 snr
                max_scale_val = self.cfg.get('noise_scale_user_min', 5.6234)  # 24 snr

                # get a random float value between min and max
                scaling_factor = (
                    torch.rand(batch["source_audio"].size(0), device=batch["source_audio"].device)
                    * (max_scale_val - min_scale_val)
                    + min_scale_val
                )
                batch["source_audio"] = batch["source_audio"] * scaling_factor.unsqueeze(-1)

            # apply low pass filter
            if self.training and random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                cutoff_freq = self.cfg.get('noise_low_pass_cutoff_freq', 1000.0)
                # note here we are using a biquad filter, older codebase we are using a filter of order 5
                batch["source_audio"] = torchaudio.functional.lowpass_biquad(
                    waveform=batch["source_audio"], sample_rate=self.source_sample_rate, cutoff_freq=cutoff_freq
                )

        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        # if inference return speaker embedding None and it will uses the cached speaker embedding
        if not self.training:
            speaker_encoder_emb = None
        else:  # if training or eval extract embedding from first agent turn returned by the dataloader
            if self.speech_generation.use_speaker_encoder:
                if 'target_audio' in batch:
                    target_first_turn_audio = batch["target_first_turn_audio"]
                    target_first_turn_audio_lens = batch["target_first_turn_audio_lens"]
                    speaker_encoder_emb = self.speech_generation.get_speaker_embedding(
                        target_first_turn_audio, target_first_turn_audio_lens, self.target_sample_rate
                    )
                # speech 2 text
                else:
                    speaker_encoder_emb = None
            else:
                speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        if 'target_audio' in batch:
            with fp32_precision(), torch.no_grad():
                target_codes, target_codes_lens = self.audio_codec.encode(
                    audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
                )
            target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

            if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
                if tl < sl:
                    diff = sl - tl
                    source_encoded = source_encoded[:, :tl]
                    asr_emb = asr_emb[:, :tl]
                    target_tokens = target_tokens[:, :tl]
                    torch.clamp_(source_encoded_lens, max=tl)
                else:
                    diff = tl - sl
                    target_codes = target_codes[:, :sl]
                    torch.clamp_(target_codes_lens, max=sl)
                if diff > 2:
                    logging.warning(
                        f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                        f"This may indicate significant desynchronization in longer sessions."
                    )

            btt = target_tokens[..., None]
            target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
            target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

            # ToDo: implement in a way that we can set the number of speech delay > 1
            target_codes = torch.cat(
                [
                    torch.full(
                        [target_codes.shape[0], 1, target_codes.shape[-1]],
                        fill_value=self.speech_delay_id,
                        device=self.device,
                        dtype=torch.long,
                    ),
                    target_codes[:, :-1],
                ],
                dim=1,
            )
        # move back text channel by x, in inference it advance the text channel prediction
        # it is the oposite of speech delay applied on text channel
        if self.advance_text_channel_by:
            pad = torch.full(
                (target_tokens.shape[0], self.advance_text_channel_by),
                fill_value=self.text_pad_id,
                device=target_tokens.device,
                dtype=torch.long,
            )
            target_tokens = torch.cat([target_tokens[:, self.advance_text_channel_by :], pad], dim=-1)
            # make sure that eos/bos is in the place (it can cut tokens from the first advance_text_channel_by tokens and this will breaks everything)

        if self.cfg.get("delay_text_eos_by", None):
            target_tokens = delay_eos(target_tokens, self.text_eos_id, self.text_pad_id, shift=self.cfg.delay_text_eos_by)
        if 'target_audio' in batch:
            input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        else:
            # Speech to text
            input_ids = target_tokens[..., None]
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]
                asr_emb = asr_emb[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        if 'target_audio' in batch:
            audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
            audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

        # create sequence mask
        if 'target_audio' in batch:
            seq_mask = torch.ones_like(
                torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
                device=self.device,
                dtype=torch.bool,
            )
        # Speech to text
        else:
            seq_mask = torch.ones_like(
                text_labels.unsqueeze(-1),
                device=self.device,
                dtype=torch.bool,
            )

        if self.cfg.get("mask_sequence_loss", True):
            # set the mask based on the target_token_lens to disconsider sequence padding in loss
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                seq_mask[i, speech_end_idx:, :] = 0

            # check new mask consistency
            mask_lengths = seq_mask[:, :, 0].sum(-1)
            assert torch.allclose(batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0)

        # create loss scale mask by copying seq_mask to include mask sequence
        loss_scale = seq_mask.clone().float()

        if self.cfg.get("scale_loss_by") == 'non_sil_t':
            loss_scale[:, :, :1] = torch.where(
                text_labels.unsqueeze(-1) != self.text_pad_id,
                self.cfg.get("scale_loss_mask", self.cfg.get("nonsil_weight", 4.0)),
                loss_scale[:, :, :1],
            )

        # debug samples:
        if (
            self.cfg.get("debug_dataloader_audios_path", None)
            and self.training
            and "s2s_duplex_overlap_as_s2s_duplex" not in batch["formatter"][0]
        ):

            def count_leading_silence_tokens(tensor: torch.Tensor, silence_token: int = 0) -> int:
                """
                Count the number of consecutive silence tokens at the beginning of a 1D tensor.

                Args:
                    tensor (torch.Tensor): 1D tensor of tokens.
                    silence_token (int): The token considered as silence (default: 0).

                Returns:
                    int: Number of consecutive silence tokens at the beginning.
                """
                if tensor.ndim != 1:
                    raise ValueError("Input tensor must be 1D.")

                count = 0
                for token in tensor:
                    if token.item() == silence_token:
                        count += 1
                    else:
                        break
                return count

            def write_wave(one_audio_signal, file_name, sr=None):
                import numpy as np
                import soundfile as sf

                one_audio_signal = one_audio_signal.cpu().numpy()
                one_audio_signal = one_audio_signal.astype(np.float32)
                if sr is None:
                    sr = self.target_sample_rate
                # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
                sf.write(file_name, one_audio_signal, sr)

            # encode and decode the audio
            if 'target_audio' in batch:
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([batch["target_audio"].shape[1]] * batch["target_audio"].shape[0]).to(
                        self.audio_codec.device
                    )
                    reconstructed_audio_from_wav, _ = self.audio_codec(audio=batch["target_audio"], audio_len=lengths)
                    # reconstruct wav
                    audio_labels_ = replace_control_speech_codes(audio_labels, self._control_codes)
                    with fp32_precision(), torch.no_grad():
                        lengths = torch.tensor([audio_labels_.shape[1]] * audio_labels_.shape[0]).to(
                            self.audio_codec.device
                        )
                        reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                            tokens=audio_labels_.transpose(1, 2), tokens_len=lengths
                        )

            for i in range(audio_labels_.shape[0]):
                write_wave(
                    batch["source_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"source_audio_{i}.wav"),
                    sr=self.source_sample_rate,
                )
                if 'target_audio' in batch:
                    write_wave(
                        batch["target_audio"][i],
                        os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_{i}.wav"),
                        sr=self.target_sample_rate,
                    )
                    write_wave(
                        batch["target_first_turn_audio"][i],
                        os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"speaker_ref_{i}.wav"),
                        sr=self.target_sample_rate,
                    )
                    write_wave(
                    reconstructed_audio_from_tokens[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_tokens_{i}.wav"
                    ),
                    sr=self.target_sample_rate,
                    )
                    write_wave(
                    reconstructed_audio_from_wav[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"),
                        f"target_audio_reconstructed_from_waveform_{i}.wav",
                    ),
                    sr=self.target_sample_rate,
                    )
                
            num_bos_tokens = (text_labels.unsqueeze(-1) == self.text_bos_id).flatten(1, 2).sum(-1)
            # Count how many EOS tokens are present per sequence
            # Shape: [B]
            num_eos_tokens = (text_labels.unsqueeze(-1) == self.text_eos_id).flatten(1, 2).sum(-1)
            print("Num eos:", num_eos_tokens, "num bos:", num_bos_tokens)
            # check text
            print(
                "text_labels decoded:",
                tokens_to_str(
                    text_labels[-1:], target_codes_lens - 1, tokenizer=self.tokenizer, pad_id=self.text_pad_id
                ),
            )
            print(
                "target labels from dataloader decoded:",
                tokens_to_str(
                    batch["target_tokens"][-1:],
                    target_codes_lens - 1,
                    tokenizer=self.tokenizer,
                    pad_id=self.text_pad_id,
                ),
            )
            print(
                "Number of padding tokens on the begining:",
                count_leading_silence_tokens(text_labels[-1:].squeeze(), self.text_pad_id),
            )

            print(batch["formatter"])
            if 'target_audio' in batch:
                if audio_labels_.shape[0] > 1:
                    exit()
        ans = {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1 if 'target_audio' in batch else batch["target_token_lens"] - 1,
            "text_labels": text_labels,
            "seq_mask": seq_mask,
            "loss_scale": loss_scale,
            "perception_emb": source_encoded[:, :-1],
            "asr_emb": asr_emb[:, :-1],
            "speaker_encoder_emb": speaker_encoder_emb,
        }
        if 'target_audio' in batch:
            ans.update({
                "input_audio_tokens": audio_inputs,
                "audio_labels": audio_labels,
            })
        return ans


    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
            if is_frozen(m):
                m.eval()

        # text 2 text training
        if 'text_tokens' in batch:
            text_input_ids = batch["text_tokens"][:, :-1]
            text_target = batch["text_tokens"][:, 1:]

            text_out = self.llm(
                inputs_embeds=self.embed_tokens(text_input_ids),
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            if hasattr(self, "text_head"):
                text_logits = self.text_head(text_out['last_hidden_state'])  # (B, T, Vt)
            else:
                text_logits = self.lm_head(text_out['last_hidden_state'])  # (B, T, Vt)

            text_loss = torch.nn.functional.cross_entropy(
                text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                text_target.flatten(0, 1),
                ignore_index=self.text_pad_id,
            )
            #text_loss = text_loss * self.cfg.get("t2t_loss_scale", 10.0)
            loss = self.cfg.get('text_to_text_loss_weight', 1) * text_loss
            ans = {
                "learning_rate": (
                    torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
                ),
                    "text_loss": text_loss, # text_loss could be used for text_loss in s2s, s2t and t2t
                    "text_to_text_loss": text_loss,  # also keep track of text_to_text_loss separately
                    "loss": loss,
                    "text_batch_size": text_input_ids.shape[0],
                    "text_sequence_length": text_input_ids.shape[1],
                    "text_num_tokens": batch["text_token_lens"].sum(),
            }
            self.log_dict(ans, on_step=True)
            return ans
        # speech2speech and speech2text training
        else:
            # add function calling channel
            if 'call_responses' in batch and 'instructions' in batch and batch['instructions'] is not None: 
                inputs = self.prepare_inputs_fc(batch)
            else:
                inputs = self.prepare_inputs(batch)
            if 'target_audio' in batch:
                forward_outputs = self(
                    inputs["input_embeds"],
                    input_audio_tokens=inputs["input_audio_tokens"],
                    seq_mask=inputs["seq_mask"],
                    target_text_tokens=inputs["text_labels"],
                    modality_adapter_emb=inputs["perception_emb"],
                    asr_emb=inputs["asr_emb"],
                    speaker_encoder_emb=inputs["speaker_encoder_emb"],
                    )
            # speech to text 
            else:
                forward_outputs = self(
                    inputs["input_embeds"],
                    input_audio_tokens=None,
                    seq_mask=inputs["seq_mask"],
                    target_text_tokens=inputs["text_labels"],
                    modality_adapter_emb=inputs["perception_emb"],
                    asr_emb=inputs["asr_emb"],
                    speaker_encoder_emb=inputs["speaker_encoder_emb"],
                    )
            num_frames = inputs["input_lens"].sum()
            with loss_parallel():
                # mask audio logits to ignore sequence padding
                text_logits = forward_outputs["text_logits"]
                if self.cfg.get("mask_sequence_loss", True):
                    text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

                text_loss = (
                    torch.nn.functional.cross_entropy(
                        text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                        inputs["text_labels"].flatten(0, 1),
                        reduction="none",
                    )
                    * inputs["loss_scale"][:, :, 0].flatten(0, 1)
                ).sum(-1) / num_frames

                if 'target_audio' in batch:
                    # mask audio logits to ignore sequence padding
                    audio_logits = forward_outputs["audio_logits"]
                    if self.cfg.get("mask_sequence_loss", True):
                        audio_logits = audio_logits * inputs["seq_mask"][:, :, -1].unsqueeze(-1).unsqueeze(-1)
                    audio_loss = (
                        torch.nn.functional.cross_entropy(
                            audio_logits.flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                            inputs["audio_labels"].flatten(0, 2),
                            reduction="none",
                        )
                        * inputs["loss_scale"][:, :, 1:].flatten(0, 2)
                    ).sum(-1) / (num_frames * self._num_codebooks)

            loss = self.cfg.text_loss_weight * text_loss 
            if 'target_audio' in batch:
                loss += self.cfg.audio_loss_weight * audio_loss

            B, T = inputs["input_embeds"].shape[:2]
            ans = {
                "loss": loss,
                "learning_rate": (
                    torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
                ),
                "text_loss": text_loss,
                "num_frames": num_frames.to(torch.float32),  # avoid warning
                "padding_ratio": num_frames / (B * T),
            }

            self.log("batch_size", B, on_step=True, prog_bar=True, logger=True)
            self.log("sequence_length", T, on_step=True, prog_bar=True, logger=True)

            if 'target_audio' in batch:
                ans['audio_loss'] = audio_loss
            self.log_dict(ans, on_step=True)
            return ans           

    def on_train_epoch_start(self) -> None:
        setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32
        if hasattr(self.speech_generation, "use_speaker_encoder") and self.speech_generation.use_speaker_encoder:
            self.speech_generation.setup_speaker_encoder()  # potentially reloads the speaker encoder to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()

        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()
        tolerance = int(
            self.cfg.get("val_acc_tolerance", 160) / (1000 / self.target_fps)
        )  # 160 ms as default tolerance --> 2 tokens for 12.5FPS and 1 for 25FPS
        self.text_bos_acc = TokenAccuracy(
            token_name="text_bos", token_id=self.text_bos_id, tolerance=tolerance
        ).reset()
        self.text_eos_acc = TokenAccuracy(
            token_name="text_eos", token_id=self.text_eos_id, tolerance=tolerance
        ).reset()

    def save_validation_results_as_json(self, prefix="val"):
        """
        Save validation results as JSON files, gathering from all GPUs but writing only from GPU 0.
              
        Files are saved in JSONL format (one JSON object per line) in the following structure:
        - {prefix}_all_results.jsonl: All results combined
        - {prefix}_{dataset_name}_results.jsonl: Results per dataset
        
        Args:
            prefix (str): Prefix for the output files (default: "val" for validation, "test" for testing)
        """
        if not self.validation_results:
            return
            
        # Gather results from all GPUs
        all_results = {}
        for name, results in self.validation_results.items():
            # Convert results to a format that can be gathered
            if dist.is_initialized():
                gathered_results = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_results, results)
                
                # Combine results from all GPUs
                combined_results = []
                for gpu_results in gathered_results:
                    if gpu_results is not None:
                        combined_results.extend(gpu_results)
            else:
                # Single GPU case
                combined_results = results
            
            all_results[name] = combined_results
        
        # Save JSON files only from GPU 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir = self.cfg.get("json_save_path", None)
            if save_dir is None:
                save_dir = os.path.join(self.trainer.log_dir, f"{prefix}_results")
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Deduplicate results before saving
            # Lhotse can produce duplicate samples during distributed inference (see:
            # https://github.com/lhotse-speech/lhotse/blob/fda1a986e5e1e72a14c82049b4ee709fc09a81e6/lhotse/dataset/sampling/base.py#L349)
            # We remove duplicates where the prefix before '_dup' exists in the dataset
            for name, results in all_results.items():
                all_results[name] = deduplicate_results(results)
            
            # Save combined results
            combined_file = os.path.join(save_dir, f"{prefix}_all_results.jsonl")
            with open(combined_file, 'w', encoding='utf-8') as f:
                for name, results in all_results.items():
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Save per-dataset results
            for name, results in all_results.items():
                dataset_file = os.path.join(save_dir, f"{prefix}_{name}_results.jsonl")
                with open(dataset_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logging.info(f"Validation results saved to {save_dir}")
        
        # Clear the results for next epoch
        self.validation_results.clear()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_bos_acc = self.text_bos_acc.compute()
        for k, m in text_bos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_eos_acc = self.text_eos_acc.compute()
        for k, m in text_eos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        self.save_validation_results_as_json(prefix=prefix)

    def transcribe_audio(self, audio, audio_lens):
        if audio_lens is None:
            audio_lens = [audio.shape[1]] * audio.shape[0]
        
        hyps = self.asr_bleu.asr.transcribe(
            [aud[:alen] for aud, alen in zip(audio, audio_lens)],
            batch_size=audio.shape[0],
            verbose=False,
        )
        hyps = [hyp.text for hyp in hyps]
        return hyps

    def validation_step(self, batch: dict, batch_idx: int):

        # Update speaker embedding to reflect the one in the prompt during inference
        if self.speech_generation.use_speaker_encoder and self.speech_generation.inference_speaker_reference:
            self.speech_generation.update_inference_speaker_embedding(
                self.speech_generation.inference_speaker_reference
            )

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            metadata = dataset_batch.get('metadata', [{}] * len(dataset_batch['target_texts']))
            if 'instructions' in dataset_batch and dataset_batch['instructions'] is not None:
                results = self.offline_inference_fc(
                    dataset_batch["source_audio"],
                    dataset_batch["source_audio_lens"],
                    dataset_batch["instructions"],
                    dataset_batch["instructions_len"],
                )
            else:
                results = self.offline_inference(
                    dataset_batch["source_audio"],
                    dataset_batch["source_audio_lens"],
                )

            # Get ASR hypotheses for the generated audio
            # torchaudio resample is fragile to bfloat16 default dtype as well
            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                )

            # Collect results for JSON saving
            for i in range(len(dataset_batch["target_texts"])):
                result_entry = {
                    "text": dataset_batch["target_texts"][i],
                    "pred_text": results["text"][i],
                    "speech_preds_transcribed": asr_hyps[i],
                    "sample_id": dataset_batch["sample_id"][i] if "sample_id" in dataset_batch else i,
                    "dataset_name": name,
                    "audio_filepath": metadata[i]['audio_filepath']
                }
                if 'instructions' in dataset_batch and dataset_batch['instructions'] is not None:
                    result_entry['sys_prompt'] = dataset_batch['instructions_raw_text'][i]
                    result_entry['call_response'] = dataset_batch['call_responses_raw_text'][i]
                self.validation_results[name].append(result_entry)

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            self.text_bos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])
            self.text_eos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        if self.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            input_signal, sr = torchaudio.load(self.cfg.custom_sample_inference)
            input_signal = input_signal.to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
        B, T_local, H = source_encoded.shape

        # Determine decoding length and pad if FSDP
        print("self._use_fsdp", self._use_fsdp)
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:

                last_frame_source = source_encoded[:, T_local - 1 : T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_encoded = torch.cat([source_encoded, pad_source], dim=1)
                last_frame_asr = asr_emb[:, T_local - 1 : T_local, :]
                pad_asr = last_frame_asr.repeat(1, T - T_local, 1)
                asr_emb = torch.cat([asr_emb, pad_asr], dim=1)
        else:
            T = T_local

        # Apply channel weight
        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)
        # This cache is for self.llm
        llm_use_cache = self.cfg.get("llm", {}).get("use_cache", True)
        if llm_use_cache:
            cache_class = self.cfg.get("llm", {}).get("cache_class", "DynamicCache")
            if cache_class == "DynamicCache":
                from transformers import DynamicCache
                cache = DynamicCache()
                print(f"Cache class {cache_class} initialized during inference")
            elif cache_class == "HybridMambaAttentionDynamicCache":
                from transformers.models.nemotron_h.modeling_nemotron_h import HybridMambaAttentionDynamicCache
                cache = HybridMambaAttentionDynamicCache(
                    self.llm.config, batch_size=B, dtype=self.llm.dtype, device=self.llm.device
                )
                print(f"Cache class {cache_class} initialized during inference")
            else:
                logging.warning(f"Cache class {cache_class} not supported. Using no cache.")
                llm_use_cache = False
                cache = None
        else:
            cache = None
        
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True) #if llm_use_cache else False)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )

        if llm_use_cache:
            if self.cfg.get("llm", {}).get("architecture", "transformers") == "nemotron_h":
                llm_kwargs = {
                    "attention_mask": torch.ones_like(source_encoded[:, :1, 0]), # shape (B, 1)
                    "cache_position": torch.arange(1, device=source_encoded.device, dtype=source_encoded.dtype), # shape (1)
                }
                print(f"LLM kwargs initialized during inference")
            else:
                llm_kwargs = {}
                print(f"LLM kwargs initialized empty during inference")
        else:
            llm_kwargs = {}
            print(f"LLM kwargs initialized empty during inference")
        ans = self(
            input_embeds[:, :1],
            cache=cache,
            input_audio_tokens=first_audio,
            seq_mask=None,
            target_text_tokens=None,  # text input will be sampled from llm backbone
            modality_adapter_emb=source_encoded[:, :1],
            asr_emb=asr_emb[:, :1],
            speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
            llm_kwargs=llm_kwargs,
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        speech_state = torch.zeros(B, device=self.device, dtype=torch.long)
        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb

            current_audio = gen_audio[:, t - 1 : t, :]

            # If llm_use_cache is disabled, pass entire sequences up to current timestep
            if llm_use_cache:
                input_embeds_slice = input_embeds[:, t : t + 1]
                source_encoded_slice = source_encoded[:, t : t + 1]
                asr_emb_slice = asr_emb[:, t : t + 1]
                if self.cfg.get("llm", {}).get("architecture", "transformers") == "nemotron_h":
                    # Grow attention mask by one valid token
                    new_mask = torch.ones((B, 1), dtype=llm_kwargs['attention_mask'].dtype, device=llm_kwargs['attention_mask'].device)
                    llm_kwargs['attention_mask'] = torch.cat([llm_kwargs['attention_mask'], new_mask], dim=1)
                    # Set absolute position of the new token
                    llm_kwargs['cache_position'] = torch.tensor([llm_kwargs['attention_mask'].shape[1]-1], dtype=llm_kwargs['attention_mask'].dtype, device=llm_kwargs['attention_mask'].device)  # shape: (1,)
            else:
                input_embeds_slice = input_embeds[:, : t + 1]
                source_encoded_slice = source_encoded[:, : t + 1]
                asr_emb_slice = asr_emb[:, : t + 1]
                ans["cache"] = None

            ans = self(
                input_embeds_slice,
                cache=ans["cache"],
                input_audio_tokens=current_audio,
                seq_mask=None,
                target_text_tokens=None,  # text input will be sampled from llm backbone
                modality_adapter_emb=source_encoded_slice,
                asr_emb=asr_emb_slice,
                speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
                llm_kwargs=llm_kwargs,
            )
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

            if self.cfg.get('inference_force_speech_state', None):
                # state 0 - silence, state 1 - speech
                speech_state = torch.where(
                    gen_text[:, t] == self.text_bos_id, torch.ones_like(speech_state), speech_state
                )
                speech_state = torch.where(
                    gen_text[:, t] == self.text_eos_id, torch.zeros_like(speech_state), speech_state
                )
                gen_audio[:, t] = torch.where(
                    speech_state.unsqueeze(-1) == 0,
                    gen_audio[:, 0],  # silence
                    gen_audio[:, t],  # speech
                )
            # inference trick force speech decoder eos/bos to make the model more robust
            num_speech_delay = 1
            if self.cfg.get('inference_force_speech_bos', None) and num_speech_delay < gen_text.shape[1]:
                gen_audio[:, t] = torch.where(
                    (gen_text[:, t - num_speech_delay].unsqueeze(-1) == self.text_bos_id)
                    * (torch.sum(gen_audio[:, t - num_speech_delay :] == self.speech_bos_id, 1) == 0),
                    self.speech_bos_id,
                    gen_audio[:, t],
                )

            if self.cfg.get('inference_force_speech_eos', None) and gen_text.shape[
                1
            ] > num_speech_delay + self.cfg.get("advance_text_channel_by", 0):
                # tmp solution: force to stop talking if user interruption is detected
                gen_audio[:, t] = torch.where(
                    (
                        (
                            gen_text[:, t - num_speech_delay - self.cfg.get("advance_text_channel_by", 0)].unsqueeze(
                                -1
                            )
                            == self.text_eos_id
                        )
                    ),
                    self.speech_eos_id,
                    gen_audio[:, t],
                )

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        # Call reset_input_and_kv_cache to reset cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if self.cfg.get("custom_sample_inference", None):
            print(ans["audio"].shape, input_signal.shape)
            self.results_logger.merge_and_save_audio(self.cfg.custom_sample_inference+"inf.wav", pred_audio=ans["audio"][0], pred_audio_sr=self.target_sample_rate, user_audio=input_signal[0], user_audio_sr=self.source_sample_rate)
            exit()
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)

    def generate_silence_tokens(self, time_steps: int, num_codebooks: int) -> torch.Tensor:
            """
            Generate silence tokens with the given time steps and number of codebooks.
            
            Args:
                time_steps (int): Number of time steps (rows) in the output tensor
                num_codebooks (int): Number of codebooks (columns) in the output tensor
                
            Returns:
                torch.Tensor: Tensor of shape (time_steps, num_codebooks) containing silence tokens
            """
            # generate silence tokens
            with torch.no_grad():
                silence_tokens , silence_tokens_lens = self.audio_codec.encode(audio = torch.zeros(1,16000,dtype=self.audio_codec.dtype,device=self.audio_codec.device),audio_len = torch.tensor([16000], device=self.audio_codec.device))
                unique_values = silence_tokens[0, :, 0]
            
            return unique_values

    def create_silence_encoded(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with fp32_precision(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
                ):
                codes_len = torch.tensor([codes.shape[0]], device=codes.device)
                audio, audio_lens = self.audio_codec.decode(
                            tokens = codes.unsqueeze(0).transpose(1, 2), 
                            tokens_len = codes_len
                        ) 
                audio = resample(audio, 22050, 16000)
                audio_lens = torch.tensor([audio.shape[1]], device=audio.device)
                encoded,encoded_lens = self.perception(
                    input_signal=audio, input_signal_length=audio_lens
                )
                encoded = encoded.squeeze(0)
                
                # Ensure encoded shape[0] matches codes.shape[0]
                codes_length = codes.shape[0]
                encoded_length = encoded.shape[0]
                
                if encoded_length > codes_length:
                    # Cut the difference from the end
                    encoded = encoded[:codes_length]
                elif encoded_length < codes_length:
                    # Repeat the last column to match the codes length
                    diff = codes_length - encoded_length
                    last_column = encoded[-1:].repeat(diff, 1)
                    encoded = torch.cat([encoded, last_column], dim=0)
        
        return encoded

    def convert_to_silence_tokens(self, speech_nosil_tokens):
        """
        Convert speech_nosil_id tokens to actual silence tokens through decode-encode process.
        
        Args:
            speech_nosil_tokens: Tensor of shape [time_steps, num_codebooks] with speech_nosil_id values
            
            
        Returns:
            silence_tokens: Tensor of shape [time_steps, num_codebooks] with actual silence tokens
        """
        
        
        # Decode tokens to speech
        with torch.no_grad():
            with fp32_precision(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
                ):

                speech_nosil_tokens_lengths = torch.tensor([speech_nosil_tokens.shape[0]], device=speech_nosil_tokens.device)

                audio, audio_lens = self.audio_codec.decode(
                    tokens = speech_nosil_tokens.unsqueeze(0).transpose(1, 2), 
                    tokens_len = speech_nosil_tokens_lengths
                ) 
                
                pure_silence = torch.full_like(audio, fill_value=0.0)
                pure_silence_lengths = torch.tensor([pure_silence.shape[1]], device=pure_silence.device)               
                silence_tokens , silence_tokens_lens = self.audio_codec.encode(
                    audio = pure_silence,
                    audio_len = pure_silence_lengths
                )
                
                            
                silence_tokens = silence_tokens.transpose(1, 2).squeeze(0)
                pure_silence_1 = torch.zeros(1,16000,device=pure_silence.device)
                pure_silence_1_lengths = torch.tensor([pure_silence_1.shape[1]], device=pure_silence.device)
                silence_tokens_1 , silence_tokens_lens_1 = self.audio_codec.encode(
                    audio = pure_silence_1,
                    audio_len = pure_silence_1_lengths
                )
                unique_values = silence_tokens_1[0, :, 0]
                silence_tokens_repeated = unique_values.repeat(n, 1)
                # Ensure the length matches speech_nosil_tokens_lengths
                if silence_tokens.shape[0] > speech_nosil_tokens_lengths[0]:
                    # Truncate if too long
                    silence_tokens = silence_tokens[:speech_nosil_tokens_lengths[0]]
                if silence_tokens.shape[0] < speech_nosil_tokens_lengths[0]:
                    padding_needed = speech_nosil_tokens_lengths[0] - silence_tokens.shape[0]  
                    
                    last_n_rows = silence_tokens[-padding_needed:]  

                    silence_tokens = torch.cat([silence_tokens, last_n_rows], dim=0)
                
                
                return silence_tokens

    def prepare_inputs_fc(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'loss_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        
        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
        target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        btt = target_tokens[..., None]
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)
        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        source_encoded_fc = []
        target_codes_fc = []
        target_tokens_fc = []
        # over batch dimension
        for i, target_code in enumerate(target_codes):
            target_token = target_tokens[i]
            encoded_user = source_encoded[i]
            
            if 'call_responses' in batch and 'instructions' in batch and batch['instructions'] is not None: # add function calling channel

                sys_prompts = batch['instructions'][i]
                sys_prompt_lens = batch['instructions_len'][i]
                sys_prompts = sys_prompts[:sys_prompt_lens] # ignore pad tokens
                call_responses = batch['call_responses'][i]
                call_response_lengths = batch['call_response_lengths'][i]
                call_response_steps =  batch['call_response_steps'][i]

                if call_responses.shape[-1] > sys_prompts.shape[-1]:
                    sys_prompt_pad = torch.full([(call_responses.shape[-1]-sys_prompts.shape[-1])], self.text_pad_id, device=sys_prompts.device)
                    sys_prompts_extended = torch.cat([sys_prompts, sys_prompt_pad])
                    call_responses_extened = call_responses
                elif call_responses.shape[-1] < sys_prompts.shape[-1]:
                    call_response_pad = torch.full([call_responses.shape[0], (sys_prompts.shape[-1]-call_responses.shape[-1])], self.text_pad_id, device=call_responses.device)
                    call_responses_extened = torch.cat([call_responses, call_response_pad], axis=1)
                    sys_prompts_extended = sys_prompts
                else:
                    sys_prompts_extended = sys_prompts
                    call_responses_extened = call_responses
                
                # call_response_steps indicates when each function call occurs. Since we prepend the system instruction,
                # we need to shift each call_response_step by adding the system instruction length to maintain correct timing
                #call_response_steps_updated = call_response_steps + len(sys_prompts_extended) # update steps with system instruction
                # update steps with system instruction if it is not -1 (pad tokens)

                # Filter out padding values (-1) and add offset, then convert to tensor
                valid_steps = call_response_steps[call_response_steps >= 0]
                call_response_steps_updated = valid_steps + len(sys_prompts_extended)
                call_response_steps_extended = torch.cat([torch.tensor([0]).to(self.device), call_response_steps_updated], axis=-1)
                
                call_responses_extended = torch.cat([sys_prompts_extended.unsqueeze(0), call_responses_extened], axis=0)
                
                call_response_lengths_extended = torch.cat([sys_prompt_lens.unsqueeze(0), call_response_lengths], axis=-1)
                n_steps = len(call_response_steps_extended)

                shift_length = 0
                for j in range(n_steps):
                    if call_response_steps_extended[j] >= 0:
                        shift_length = call_response_steps_extended[j]
                        if shift_length == 0:
                            target_token = torch.cat([call_responses_extended[j], target_token], axis=0)
                            encoded_user = torch.cat([torch.full([len(call_responses_extended[j]), encoded_user.shape[-1]], 0.0, device=encoded_user.device), encoded_user], axis=0)
                            if self.use_silence_tokens:
                                target_code= torch.cat([self.silence_tokens.repeat(len(call_responses_extended[j]), 1), target_code], axis=0)
                            else:
                                target_code= torch.cat([torch.full([len(call_responses_extended[j]), target_code.shape[-1]], self.speech_nosil_id, device=target_code.device), target_code], axis=0) 
                            #target_code = torch.cat([self.silence_tokens.repeat(len(call_responses_extended[j]), 1), target_code], axis=0)
                            #silence_codes_agent = self.silence_tokens.repeat(len(call_responses_extended[j]), 1) # option 3
                            #target_code = torch.cat([silence_codes_agent, target_code], axis=0)
                            #silence_encoded_user = self.create_silence_encoded(silence_codes_agent)
                            #encoded_user = torch.cat([silence_encoded_user, encoded_user], axis=0)
                            
                        else:
                            target_token = torch.cat([target_token[:shift_length], call_responses_extended[j], target_token[shift_length:]], axis=0)
                            encoded_user = torch.cat([encoded_user[:shift_length], torch.full([len(call_responses_extended[j]), encoded_user.shape[-1]], 0.0, device=encoded_user.device), encoded_user[shift_length:]], axis=0)
                            if self.use_silence_tokens:
                                target_code = torch.cat([target_code[:shift_length], self.silence_tokens.repeat(len(call_responses_extended[j]), 1), target_code[shift_length:]], axis=0)
                            else:
                                target_code = torch.cat([target_code[:shift_length], torch.full([len(call_responses_extended[j]), target_code.shape[-1]], self.speech_nosil_id, device=target_code.device), target_code[shift_length:]], axis=0)                            
                            #target_code = torch.cat([target_code[:shift_length], self.silence_tokens.repeat(len(call_responses_extended[j]), 1), target_code[shift_length:]], axis=0)
                            #silence_codes_agent = self.silence_tokens.repeat(len(call_responses_extended[j]), 1)
                            #silence_encoded_user = self.create_silence_encoded(silence_codes_agent)
                            #encoded_user = torch.cat([encoded_user[:shift_length], silence_encoded_user, encoded_user[shift_length:]], axis=0)

            elif 'instructions' in batch and batch['instructions'] is not None: # system instruction only data
                sys_prompts = batch['instructions'][i]
                sys_prompt_lens = batch['instructions_len'][i]
                sys_prompts = sys_prompts[:sys_prompt_lens] # ignore pad tokens
                sys_prompts_extended = sys_prompts
                
                target_token = torch.cat([sys_prompts_extended, target_token], axis=0)
                encoded_user = torch.cat([torch.full([len(sys_prompts_extended), encoded_user.shape[-1]], 0.0, device=encoded_user.device), encoded_user], axis=0)
                if self.use_silence_tokens:
                    target_code = torch.cat([self.silence_tokens.repeat(len(sys_prompts_extended), 1), target_code], axis=0)
                else:
                    target_code = torch.cat([torch.full([len(sys_prompts_extended), target_code.shape[-1]], self.speech_nosil_id, device=target_code.device), target_code], axis=0)

            target_codes_fc.append(target_code)
            target_tokens_fc.append(target_token)
            source_encoded_fc.append(encoded_user)
        
        # Pad sequences to the same length
        if 'instructions' in batch and batch['instructions'] is not None:       
            target_tokens_fc = pad_sequence(target_tokens_fc, batch_first=True, padding_value=self.text_pad_id)
            source_encoded_fc = pad_sequence(source_encoded_fc, batch_first=True, padding_value=0.0)
            target_codes_fc = pad_sequence(target_codes_fc, batch_first=True, padding_value=self.speech_nosil_id)
            
            input_ids = torch.cat([target_codes_fc, target_tokens_fc[..., None]], dim=-1)
        else:
            input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                if 'instructions' in batch and batch['instructions'] is not None:
                    source_encoded_fc = source_encoded_fc[:, :-remainder]
                else:
                    source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        if 'instructions' in batch and batch['instructions'] is not None:
            input_embeds.add_(
                source_encoded_fc[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0)
            )
        else:
            input_embeds.add_(
                source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0)
            )

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        # [TODO] Recheck the loss mask since the base code does not include it.
        # if self.cfg.get("mask_sequence_loss", True):
        #     # set the mask based on the target_token_lens to disconsider sequence padding in loss
        #     for i in range(batch["target_token_lens"].size(0)):
        #         speech_end_idx = batch["target_token_lens"][i]
        #         loss_mask[i, speech_end_idx:, :] = 0

        #     # check new mask consistency
        #     mask_lengths = loss_mask[:, :, 0].sum(-1)
        #     assert torch.allclose(
        #         batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0
        #     )

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
        }

    @torch.no_grad()
    def offline_inference_fc(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        sys_prompts: torch.Tensor,
        sys_prompt_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        
        input_embeds_system_prompt = self.embed_tokens(sys_prompts)
        # No need this step since encoded_user is all zeros
        #encoded_user = torch.full([len(sys_prompts), 2048], 0.0, device=input_signal.device)
        #input_embeds.add_(encoded_user * self.cfg.get("duplex_user_channel_weight", 1.0))
        if input_signal.shape[1] > 22050 * 200:
            input_signal = input_signal[:, :22050 * 200]
            input_signal_lens = torch.clamp(input_signal_lens, max=22050 * 200)

        input_embeds, lengths = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
        )
        
        input_embeds = torch.cat([input_embeds_system_prompt, input_embeds], dim=1)
        
        B, T_local, H = input_embeds.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=input_embeds.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame = input_embeds[:, T_local - 1 : T_local, :]  # (B,1,H)
                pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                input_embeds = torch.cat([input_embeds, pad], dim=1)
        else:
            T = T_local

        # Apply channel weight
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # This cache is for self.llm
        llm_use_cache = self.cfg.get("llm", {}).get("use_cache", True)
        if llm_use_cache:
            cache_class = self.cfg.get("llm", {}).get("cache_class", "DynamicCache")
            if cache_class == "DynamicCache":
                from transformers import DynamicCache
                cache = DynamicCache()
                """
                #ToDo: Add support for HybridCache
                elif cache_class == "HybridCache":
                    from transformers import HybridCache
                    cache = StaticCache()
                """
            else:
                logging.warning(f"Cache class {cache_class} not supported. Using no cache.")
                llm_use_cache = False
                cache = None
        else:
            cache = None
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        
        # If llm_use_cache is disabled, pass entire sequences up to current timestep
        if not llm_use_cache:
            input_embeds_slice = input_embeds[:, : t + 1]
        else:
            input_embeds_slice = input_embeds[:, t : t + 1]

        ans = self(
            input_embeds_slice, 
            cache=cache, 
            input_audio_tokens=first_audio, 
            loss_mask=None
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(
                input_embeds[:, t : t + 1], 
                cache=ans["cache"], 
                input_audio_tokens=current_audio
            )
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)
        
        # remove parts corresponding to system prompt
        sys_prompt_batch_len = input_embeds_system_prompt.shape[-2]
        
        gen_text = gen_text[:, sys_prompt_batch_len:]
        gen_audio = gen_audio[:, sys_prompt_batch_len:]
        
        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        return ans
    
    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)