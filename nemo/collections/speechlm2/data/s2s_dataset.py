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

# flake8: noqa: E501, E302

import re

import torch
import torch.utils.data
import torchaudio

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.supervision import AlignmentItem
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id, collate_and_pad_1d, collate_and_pad_2d, collate_and_pad
from nemo.utils import logging
from nemo.collections.common.data.lhotse.text_adapters import Formattable

from typing import Tuple

MIN_FRAMES_FOR_TEXT = 3




def first_nonzero_idx_torch(x: torch.Tensor, zero_value: int = 0, none_value: int = -1):
    # x: LongTensor/FloatTensor of shape (B, T)
    mask = x != zero_value                 # (B, T) boolean
    idx = mask.float().argmax(dim=1)       # first True index (garbage if no True)
    has = mask.any(dim=1)                  # (B,)
    fill = torch.full_like(idx, none_value)
    return torch.where(has, idx, fill)     # (B,) LongTensor


class DuplexS2SDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        text_max_tokens: int = 10000,
        use_word_pad: bool = False,
        use_alignment_items: bool = False,
        system_prompt: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.use_word_pad = use_word_pad
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.text_max_tokens = text_max_tokens
        self.use_alignment_items = use_alignment_items
        self.system_prompt = system_prompt

        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, cuts: CutSet) -> dict:
        if getattr(cuts[0], "s2s_duplex_functioncalling", False):
            return self.__getitem__duplex_functioncalling_(cuts)
        elif getattr(cuts[0], "t2t", False):
            return self.__getitem__t2t_(cuts)
        elif getattr(cuts[0], "s2s_duplex", False):
            return self.__getitem__duplex__(cuts)
        else:
            return self.__getitem__duplex__(cuts)

    def __getitem__duplex__(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)
        is_s2t = getattr(cuts[0], "s2t", False)
        pad_id = get_pad_id(self.tokenizer)
        
        system_tokens = (
            [] if self.system_prompt is None 
            else get_system_prompt_ids(self.tokenizer, self.system_prompt)
        )
        offset = len(system_tokens)
        
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_tokens, target_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.output_roles,
            use_alignment_items=self.use_alignment_items,
            use_word_pad=self.use_word_pad
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.input_roles,
            use_alignment_items=self.use_alignment_items,
            use_word_pad=self.use_word_pad
        )
        
        # Add system prompt at the beginning of earliest among source and target
        # Note: in our data, esp. real audio, the "agent" may start speaking before the "user"
        B = target_tokens.shape[0]
        target_first_positions = first_nonzero_idx_torch(target_tokens, pad_id)
        source_first_positions = first_nonzero_idx_torch(source_tokens, pad_id)
        
        # Make room for system prompt at the beginning of the sequence
        if any(source_first_positions <= offset) or any(target_first_positions <= offset):  
            target_tokens = torch.cat(
                [torch.full((B, offset), fill_value=pad_id, dtype=target_tokens.dtype), target_tokens],
                dim=1
            )
            
            source_tokens = torch.cat(
                [torch.full((B, offset), fill_value=pad_id, dtype=target_tokens.dtype), source_tokens],
                dim=1
            )
        # Insert in the sequences    
        for idx in range(B):
            if (t_pos := target_first_positions[idx]) < (s_pos := source_first_positions[idx]): 
                target_tokens[idx, :offset] = torch.tensor(
                    system_tokens,
                    dtype=target_tokens.dtype
                )
            else:
                source_tokens[idx, :offset] = torch.tensor(
                    system_tokens,
                    dtype=source_tokens.dtype
                )
         
        # Common metadata processing
        metadata = []
        for id, cut in enumerate(cuts):
            metadata.append({'audio_filepath': cut.id + '.wav'})

        # Base return dictionary with common fields
        result = {
            "sample_id": [str(cut.id) for cut in cuts],
            "n_system_tokens": offset,
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in cuts],
            "metadata": metadata,
        }

        # Speech to speech  
        if not is_s2t:
            target_audio, target_audio_lens = collate_audio(
                cuts.resample(self.target_sample_rate), recording_field="target_audio"
            )
            # extract target speaker first turn audio to uses for speaker conditioning
            target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
                cuts.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
            )
            result.update({
                "target_audio": target_audio,
                "target_audio_lens": target_audio_lens,
                "target_first_turn_audio": target_first_turn_audio,
                "target_first_turn_audio_lens": target_first_turn_audio_lens,
            })

        return result


    def __getitem__t2t_(self, cuts: CutSet) -> dict:
        text_cuts = cuts.filter(lambda c: isinstance(c, Formattable))
        text_data = None
        if text_cuts:
            text_tokens = []
            text_token_lens = []
            for c in text_cuts:
                if c.input_ids.shape[0] > self.text_max_tokens:
                    # randomly select a segment of input_ids
                    # start = torch.randint(0, c.input_ids.shape[0] - self.text_max_tokens + 1, (1,)).item()
                    # end = start + self.text_max_tokens
                    # text_ids = c.input_ids[start:end]
                    raise RuntimeError(f"Text too long: {c.input_ids.shape[0]} > {self.text_max_tokens}")
                else:
                    text_ids = c.input_ids

                text_tokens.append(text_ids)
                text_token_lens.append(text_ids.shape[0])

            text_tokens = collate_vectors(
                text_tokens, padding_value=get_pad_id(self.tokenizer)
            )
            text_token_lens = torch.tensor(text_token_lens, dtype=torch.long)
            text_data = {
                "text_tokens": text_tokens,
                "text_token_lens": text_token_lens,
            }
        return text_data

    def __getitem__duplex_functioncalling_(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_audio, target_audio_lens = collate_audio(
            cuts.resample(self.target_sample_rate), recording_field="target_audio"
        )
        target_tokens, target_token_lens = collate_token_channel_fc(
            cuts, self.tokenizer, self.frame_length, roles=self.output_roles
        )
        source_tokens, source_token_lens = collate_token_channel_fc(
            cuts, self.tokenizer, self.frame_length, roles=self.input_roles
        )

        # Handle function calling 
        metadata = []
        num_turns = []
        call_responses, call_responses_lengths = [], []
        call_responses_times, call_responses_steps = [], []
        call_responses_raw_text = []
        instruction_texts, instruction_text_lengths = [], []
        instruction_raw_text = []
        
        def get_step_by_time(text_start_time):
            text_start_step = (
                text_start_time
                * self.codec_sample_rate 
                / self.codec_model_downsampling_factor
                // self.decoder_reduction_factor
            )
            return int(text_start_step) - 1

        def validate_time(input_time):
            if input_time > cut.duration + 0.16:
                logging.info(f"{input_time} > {cut.duration} in {cut}")
            return min(input_time, cut.duration)

        def get_text_from_segments_fc(segments): #, total_steps):
            call_responses = []
            call_response_lengths = []
            call_response_times = []
            call_response_steps = []
            call_responses_raw_text = []
            for i in range(0, len(segments), 2):
                pattern = r"<\|\d+\|>"
                call = segments[i]
                call_responses_raw_text.append(call.custom['function'])
                # Check if there's a response (next segment exists)
                if i + 1 < len(segments):
                    response = segments[i+1]
                    call_response_text = " ".join([call.custom['function'], response.custom['function']])             
                    call_responses_raw_text.append(response.custom['function'])
                else:
                    # Only call exists, no response
                    call_response_text = call.custom['function']
                output_text = re.sub(pattern, "", call_response_text)
                output_text = re.sub(r'\s+', ' ', output_text).strip()
                # The original code is overly complicated, but it essentially converts text to token IDs.
                #target_text = self.text_processor._process_example(context="", output=output_text)
                # -1 to remove the eos token added by the text processor
                #target_text, target_text_length = torch.as_tensor(target_text["answer_ids"][:-1]), torch.as_tensor(
                #    len(target_text["answer_ids"]) - 1
                #)
                target_text = torch.as_tensor(self.tokenizer.text_to_ids(output_text))
                target_text_length = torch.as_tensor(len(target_text))
                call_responses.append(target_text)
                call_response_lengths.append(target_text_length)

                text_start_time = call.start
                text_start_time = validate_time(text_start_time)

                text_start_step = compute_num_frames(
                    duration=(text_start_time),
                    frame_shift=self.frame_length,
                    sampling_rate=self.target_sample_rate
                )
                call_response_times.append(text_start_time)
                call_response_steps.append(text_start_step)
            return call_responses, call_response_lengths, call_response_times, call_response_steps, call_responses_raw_text

        def get_text_from_instruction(segment):
            pattern = r"<\|\d+\|>"
            output_text = re.sub(pattern, "", segment.text)
            output_text = re.sub(r'\s+', ' ', output_text).strip()
            target_text = torch.as_tensor(self.tokenizer.text_to_ids(output_text))
            target_text_length = torch.as_tensor(len(target_text))
            return target_text, target_text_length, segment.text

        # iterate over all cuts in a batch
        for id, cut in enumerate(cuts):
            num_turns.append(len(cut.supervisions) - 1) # 1st supervision is system instruction
            # [TODO Check]
            metadata.append({'audio_filepath': cut.id + '.wav'})
            
            # Add logging before assertion to debug failures
            if cut.supervisions[0].speaker != 'system':
                logging.error(f"Assertion failed: cut.id={cut.id}, first supervision speaker='{cut.supervisions[0].speaker}', expected='system'")
                logging.error(f"Cut object: {cut}")
            
            assert cut.supervisions[0].speaker == 'system'
            instruction_segment = cut.supervisions[0]
            
            if 'function' in cut.supervisions[1].custom:
                function_segments = [sup for sup in cut.supervisions[1:] if sup.custom['function'] != '']
            else:
                function_segments = []


            cur_instruction_text, cur_instruction_text_length, cur_instruction_raw_text = get_text_from_instruction(instruction_segment)
            instruction_texts.append(cur_instruction_text)
            instruction_text_lengths.append(cur_instruction_text_length)
            instruction_raw_text.append(cur_instruction_raw_text)
            if len(function_segments) > 0:
                cur_call_responses, cur_call_response_lengths, cur_call_response_times, cur_call_response_steps, cur_call_responses_raw_text = get_text_from_segments_fc(function_segments)
    
                call_responses.append(collate_and_pad(cur_call_responses, get_pad_id(self.tokenizer))[0])
                call_responses_lengths.append(cur_call_response_lengths)
                call_responses_times.append(cur_call_response_times)
                call_responses_steps.append(cur_call_response_steps)
                call_responses_raw_text.append(cur_call_responses_raw_text)

        instruction_texts, instruction_text_lengths = collate_and_pad(instruction_texts, get_pad_id(self.tokenizer))

        if len(call_responses) > 0:
            
            call_responses = collate_and_pad_2d(call_responses, get_pad_id(self.tokenizer)) # [b, t, l]
            call_responses_lengths= collate_and_pad_1d(call_responses_lengths) # [b, t]
            call_responses_times = collate_and_pad_1d(call_responses_times) # [b, t]
            call_responses_steps = collate_and_pad_1d(call_responses_steps) # [b, t]
        else:
            call_responses = None
            call_responses_lengths = None
            call_responses_times = None
            call_responses_steps = None
        metadata = []
        for id, cut in enumerate(cuts):
            metadata.append({'audio_filepath': cut.id + '.wav'})
        return {
            "sample_id": [cut.id for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "call_responses": call_responses,
            "call_response_lengths": call_responses_lengths,
            "call_response_times": call_responses_times,
            "call_response_steps": call_responses_steps,
            "instructions": instruction_texts, #None,
            "instructions_len": instruction_text_lengths, #None,
            "instructions_raw_text": instruction_raw_text,
            "call_responses_raw_text": call_responses_raw_text,
            "metadata": metadata,
        }


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def get_word_pad_ids(tokenizer: TokenizerSpec):
    # TODO: move the token list to config
    return tokenizer.tokenizer.convert_tokens_to_ids(
        ["<|wd_pad_id|>", "<|wd_epad_id|>"]
    ) 

       
def get_system_prompt_ids(tokenizer: TokenizerSpec, system_prompt: str):
    # TODO: move the token list to config
    return tokenizer.tokenizer.convert_tokens_to_ids(
        ['<|begin_of_text|>', '<|start_header_id|>', 'system', '<|end_header_id|>'] +
         tokenizer.tokenizer.tokenize(system_prompt) +
         ['<|eot_id|>']
    )

    
def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    use_alignment_items: bool = False,
    use_word_pad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    
    pad_id = get_pad_id(tokenizer)
    if use_word_pad:
        word_pad_id, word_epad_id = get_word_pad_ids(tokenizer)
    else:
        word_pad_id = word_epad_id = pad_id
    
    tokens = [
        build_token_channel(
            c, tokenizer=tokenizer, frame_length=frame_length, roles=roles,
            pad_id=pad_id,
            use_alignment_items=use_alignment_items,
            word_pad_id=word_pad_id,
            word_epad_id=word_epad_id
        )
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def collate_token_channel_fc(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel_fc(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_aligned_tokens(
    alignment: AlignmentItem,
    item_index: int,
    superv_start_pos: int,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    sampling_rate: int,
    tokens_len: int,
    diagnostic: str,
    # pad_id: int = -1,
    # use_alignment_items: bool = False,
    # word_pad_id: int = -2,
    # word_epad_id: int = -3
) -> Tuple[torch.Tensor, int, int, bool]:
    
    text_overflow = False
    if item_index == 0:  # first word in the sequence, add bos
        text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(alignment.symbol))
    else:  
        # Add space before each word so it's tokenized similarly to a sentence. Note that token('<word>') != token(' <word>')
        text_ids = torch.as_tensor(tokenizer.text_to_ids(" " + alignment.symbol))
    
    start_pos = compute_num_frames(alignment.start, frame_length, sampling_rate)
    if superv_start_pos + start_pos >= tokens_len:
        logging.warning(
            f"Ill-constructed example: the beginning offset of a word {superv_start_pos + start_pos} is larger " \
            f"than or equal to the example's length {tokens_len}. {diagnostic}\n"
        )
        text_overflow = True
        return torch.empty(0), start_pos, start_pos, start_pos, text_overflow

    eos_pos = compute_num_frames(alignment.end, frame_length, sampling_rate)
    
    available_frames_for_text = eos_pos - start_pos
    # We leave some margin for short duration words (could be imperfect)
    if available_frames_for_text < MIN_FRAMES_FOR_TEXT:
        available_frames_for_text = MIN_FRAMES_FOR_TEXT
        eos_pos = start_pos + MIN_FRAMES_FOR_TEXT
          
    if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
        # Truncate text_ids to fit before the eos position.
        text_ids = text_ids[:available_frames_for_text]
    elif available_frames_for_text <= 0:
        # If there's no space for text (e.g., start >= end), use an empty sequence.
        text_ids = torch.tensor([], dtype=torch.long)

    end_pos = start_pos + len(text_ids)
    
    if end_pos + superv_start_pos > tokens_len:
        trunc_len = superv_start_pos + end_pos - tokens_len
        logging.warning(
            f"Truncating training example's *word* text_ids of length {len(text_ids)} by {trunc_len} because end pos {end_pos + superv_start_pos} > {tokens_len=}. {diagnostic}\n"
        )
        text_ids = text_ids[:trunc_len]
        end_pos = start_pos + len(text_ids) 
        text_overflow = True 

    return text_ids, start_pos, end_pos, eos_pos, text_overflow


def build_token_channel(
        cut: Cut,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
        pad_id: int = -1,
        use_alignment_items: bool = False,
        word_pad_id: int = -2,
        word_epad_id: int = -3
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            
            start_pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            
            if use_alignment_items and supervision.alignment is not None:
                
                prev_end_pos = 0
                for idx, alignment in enumerate(supervision.alignment['word']):
                    if alignment is None:
                        logging.warning(
                            f"Empty alignment found at index {idx} - info: {diagnostic}\n"
                        )
                        continue
                    
                    text_ids, w_start_pos, w_end_pos, w_eos_pos, text_overflow = build_aligned_tokens(
                        alignment, idx, start_pos, tokenizer, frame_length, cut.sampling_rate, len(tokens),
                        diagnostic, 
                        # use_alignment_items=use_alignment_items,
                        # pad_id=pad_id, word_pad_id=word_pad_id, word_epad_id=word_epad_id
                    )                 
                    if text_overflow:                        
                        break
                    try:
                        if idx > 0:  # Add word padding tokens
                            tokens[start_pos+prev_end_pos:start_pos+w_start_pos] = word_pad_id
                        if idx > 0 and w_start_pos-1 > prev_end_pos:  # Add EAPD token
                            tokens[start_pos+w_start_pos-1] = word_epad_id
                            
                        tokens[start_pos+w_start_pos:start_pos+w_end_pos] = text_ids
                        
                        prev_end_pos = w_end_pos
                    except Exception as e:
                        raise RuntimeError(
                            f"{tokens.shape=} {start_pos+w_start_pos=} {start_pos+w_end_pos=} " \
                            f"{text_ids.shape=} {diagnostic}"
                        ) from e

                if start_pos+w_eos_pos < len(tokens):
                    tokens[start_pos+w_eos_pos] = tokenizer.eos
                else:  # more text than expected, we should still force an eos
                    tokens[-1] = tokenizer.eos
            else:
                text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))

                if start_pos >= len(tokens):  # Changed from > to >= for robustness
                    logging.warning(
                        f"Ill-constructed example: the beginning offset of a supervision {start_pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}\n"
                    )
                    continue

                eos_pos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
                
                # alignment info may be missing in some cases, e.g., when ASR failed
                # if use_alignment_items and supervision.alignment is None:  
                    # ...
                    # Nothing to do
                
                available_frames_for_text = eos_pos - start_pos
                if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                    # Truncate text_ids to fit before the eos position.
                    text_ids = text_ids[:available_frames_for_text]
                elif available_frames_for_text <= 0:
                    # If there's no space for text (e.g., start >= end), use an empty sequence.
                    text_ids = torch.tensor([], dtype=torch.long)

                end_pos = start_pos + len(text_ids)
                if end_pos > len(tokens):
                    trunc_len = len(tokens) - start_pos
                    logging.warning(
                        f"Truncating training example's text_ids of length {len(text_ids)} by " \
                        f"{trunc_len} because {end_pos=} > {len(tokens)=}. {diagnostic}\n"
                    )
                    text_ids = text_ids[:trunc_len]
                    end_pos = start_pos + len(text_ids)  

                try:
                    tokens[start_pos:end_pos] = text_ids
                except Exception as e:
                    raise RuntimeError(f"{tokens.shape=} {start_pos=} {end_pos=} {text_ids.shape=} {diagnostic}") from e

                if eos_pos < len(tokens):
                    tokens[eos_pos] = tokenizer.eos
                else:  # more text than expected, we should still force an eos
                    tokens[-1] = tokenizer.eos

    return tokens

def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces

def build_token_channel_fc(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    # Skip system supervision (first supervision) and function calling segments
    for supervision in cut.supervisions[1:]:  # Skip first supervision (system)
        if supervision.speaker in roles and ('function' not in supervision.custom or supervision.custom['function'] == ''):
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # Insert EOS at the end of the supervision segment
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            if eospos < len(tokens):  # skip otherwise - unfinished turn
                tokens[eospos] = tokenizer.eos

    return tokens
