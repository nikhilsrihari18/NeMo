import os

import torch
import librosa
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torchaudio

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.utils import logging

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
DEVICE = torch.cuda.current_device()
INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_PRECISION = torch.float32
OUTPUT_AUDIO_SAMPLE_RATE = 22050
MODEL_PRECISION = torch.bfloat16

class DuplexS2SSpeechDecoderSampleInference:

    def __init__(self, model_config_path: str, model_ckpt_path: str, generate_audio: bool):
        logging.info("Initializing DuplexS2SSpeechDecoderSampleInference...")
        self.model_config_path = model_config_path
        self.model_ckpt_path = model_ckpt_path
        self.generate_audio = generate_audio
        self.model_config = OmegaConf.to_container(OmegaConf.load(self.model_config_path), resolve=True)
        self.model_config["exp_manager"]["resume_from_checkpoint"] = self.model_ckpt_path
        self._load_model()
        logging.info("DuplexS2SSpeechDecoderSampleInference initialized successfully.")


    def _load_model(self):
        logging.info(f"Loading model...")
        torch.distributed.init_process_group(backend="nccl")
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        trainer = Trainer(**resolve_trainer_cfg(OmegaConf.create(self.model_config["trainer"])))

        with trainer.init_module():
            self.model = DuplexS2SSpeechDecoderModel(self.model_config)
            self.model.to(DEVICE)
            self.model.to(MODEL_PRECISION)
            self.model.eval()
            self.model.on_train_epoch_start()
        
        self.tokenizer = self.model.tokenizer
        logging.info("Model loaded successfully.")


    def load_and_preprocess_audio(self, wav_path: str):
        """To be moved into DuplexS2SSpeechDecoderModel class"""
        logging.info(f"Loading audio: {wav_path}")
        # Load audio
        waveform, sr = librosa.load(wav_path, sr=None)
        # Resample to target sample rate
        if sr != INPUT_AUDIO_SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=INPUT_AUDIO_SAMPLE_RATE)
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(waveform, dtype=INPUT_AUDIO_PRECISION).unsqueeze(0).to(DEVICE)
        audio_lens = torch.tensor([len(waveform)], dtype=torch.int32).to(DEVICE)
        return audio_tensor, audio_lens


    def save_combined_audio(self, pred_audio_path: str, pred_audio_data: torch.tensor, input_audio_data: torch.tensor):
        if len(pred_audio_data.shape) > 1:
            pred_audio_data = pred_audio_data[0]  # Take first batch
        if len(input_audio_data.shape) > 1:
            input_audio_data = input_audio_data[0]  # Take first batch

        # Saving combined audio
        input_audio_data = torchaudio.functional.resample(input_audio_data.float(), INPUT_AUDIO_SAMPLE_RATE, OUTPUT_AUDIO_SAMPLE_RATE)
        T1, T2 = pred_audio_data.shape[0], input_audio_data.shape[0]
        max_len = max(T1, T2)
        pred_audio_data_padded = torch.nn.functional.pad(pred_audio_data, (0, max_len - T1), mode='constant', value=0)
        input_audio_data_padded = torch.nn.functional.pad(input_audio_data, (0, max_len - T2), mode='constant', value=0)
        combined_wav = torch.cat(
            [
                input_audio_data_padded.squeeze().unsqueeze(0).detach().cpu(),
                pred_audio_data_padded.squeeze().unsqueeze(0).detach().cpu(),
            ],
            dim=0,
        )
        torchaudio.save(pred_audio_path, combined_wav.squeeze(), OUTPUT_AUDIO_SAMPLE_RATE)
        logging.info(f"Combined audio saved at: {pred_audio_path}")


    def save_output_audio(self, pred_audio_path: str, pred_audio_data: torch.tensor):
        if len(pred_audio_data.shape) > 1:
            pred_audio_data = pred_audio_data[0]  # Take first batch

        # Saving combined audio
        pred_audio_data = pred_audio_data.squeeze().unsqueeze(0).detach().cpu()
        torchaudio.save(pred_audio_path, pred_audio_data.squeeze(), OUTPUT_AUDIO_SAMPLE_RATE)
        logging.info(f"Output audio saved at: {pred_audio_path}")


    def save_text(self, text_path: str, text: str):
        with open(text_path, "w") as f:
            f.write(text)
        logging.info(f"Text output written to {text_path}.")

    
    @torch.no_grad()
    def run_inference(self, audio_path: str = None, input_text: str = None):
        logging.info("Starting inference...")
        if self.model.speech_generation.use_speaker_encoder and self.model.speech_generation.inference_speaker_reference:
            self.model.speech_generation.update_inference_speaker_embedding(
                self.model.speech_generation.inference_speaker_reference
            )
        if input_text:
            response = self.model.offline_inference(
                input_text=input_text
            )
        else:
            input_signal, input_signal_lens = self.load_and_preprocess_audio(audio_path)
            logging.info(f"Input signal: {input_signal.shape}")
            logging.info(f"Input signal lens: {input_signal_lens}")
            logging.info(f"Input text: {input_text}")
            response = self.model.offline_inference(
                input_signal=input_signal,
                input_signal_lens=input_signal_lens
            )
        return response, input_signal


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to the model config yaml file from the experiment directory")
    parser.add_argument("--model_ckpt_path", type=str, required=True, help="Path to the model checkpoint file from the experiment directory")
    parser.add_argument("--input_audio_path", type=str, required=False, default=None, help="Path to the input audio wav file")
    parser.add_argument("--input_text_path", type=str, required=False, default=None, help="Path to the input text txt file")
    parser.add_argument("--input_text", type=str, required=False, default=None, help="Input text string")
    parser.add_argument("--generate_audio", action="store_true", help="Boolean to decide if audio should be generated. For now, this param is ignored and audio is always generated")
    parser.add_argument("--output_text_path", type=str, required=False, default=None, help="Path to the output text txt file")
    parser.add_argument("--output_audio_path", type=str, required=False, default=None, help="Path to the combined input andoutput audio wav file")
    args = parser.parse_args()

    # Validate input params
    assert args.input_audio_path is not None or args.input_text_path is not None \
        or args.input_text is not None, "One of input_audio_path or input_text_path or input_text must be provided"
    if args.input_text_path:
        assert args.input_text is None, "input_text and input_text_path cannot be provided together"
        with open(args.input_text_path, "r") as f:
            args.input_text = f.read()
    if args.input_text:
        assert args.input_text_path is None, "input_text and input_text_path cannot be provided together"
    # Validate output params
    args.generate_audio = True
    if args.generate_audio:
        assert args.output_audio_path is not None, "output_audio_path must be provided if generate_audio is True"
    else:
        assert args.output_audio_path is None, "output_audio_path must be None if generate_audio is False"

    inference = DuplexS2SSpeechDecoderSampleInference(args.model_config_path, args.model_ckpt_path, args.generate_audio)

    response, input_audio_data = inference.run_inference(args.input_audio_path, args.input_text)

    logging.info(f"Response Text: {response['text']}")
    if args.output_text_path:
        inference.save_text(args.output_text_path, response['text'])

    if args.generate_audio:
        inference.save_combined_audio(args.output_audio_path, response['audio'], input_audio_data)


if __name__ == "__main__":
    main()