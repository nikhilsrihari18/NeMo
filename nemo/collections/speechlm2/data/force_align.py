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

import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio

from lhotse import CutSet, MonoCut, Seconds, SupervisionSegment


class ForceAligner:
    """Force alignment utility using wav2vec2-based models for speech-to-text alignment."""
    
    def __init__(self, device: str = None, frame_length: float = 0.02):
        """
        Args:
            device: Device to run alignment on (default: auto-detect)
            frame_length: Frame length in seconds for timestamp conversion
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_length = frame_length
        self.wav2vec2_model = None
        self.wav2vec2_tokenizer = None
        self.wav2vec2_aligner = None
        self.wav2vec2_bundle = None
        self._load_wav2vec2_model()
    
    def _load_wav2vec2_model(self):
        """Load the wav2vec2 model and related components."""
        try:
            device = torch.device(self.device)
            logging.info(f"Loading wav2vec2 model for force alignment on device {device}")
            from torchaudio.pipelines import MMS_FA as bundle
            self.wav2vec2_bundle = bundle
            self.wav2vec2_model = bundle.get_model().to(device)
            self.wav2vec2_tokenizer = bundle.get_tokenizer()
            self.wav2vec2_aligner = bundle.get_aligner()
            self.wav2vec2_model.eval()
            logging.info("Wav2vec2 model loaded successfully for force alignment")
        except Exception as e:
            logging.error(f"Failed to load wav2vec2 model for force alignment: {e}")
            self.wav2vec2_model = None
    
    def batch_force_align_user_audio(self, cuts: CutSet, roles: set[str] | None, source_sample_rate: int = 16000) -> None:
        """
        Perform batch force alignment on all user audio segments with debug logging.

        Args:
            cuts: CutSet containing all cuts to process
            source_sample_rate: Source sample rate of the audio
        """
        if self.wav2vec2_model is None:
            logging.warning("Wav2vec2 model not available for force alignment, skipping batch alignment")
            return

        user_supervisions = []
        user_cuts = []

        # Collect all user supervisions
        for cut in cuts:
            for supervision in cut.supervisions:
                if supervision.speaker in roles:
                    user_supervisions.append(supervision)
                    user_cuts.append(cut)
        
        if not user_supervisions:
            logging.info("No user supervisions found for force alignment")
            return

        logging.info(f"[DEBUG] Performing force alignment on {len(user_supervisions)} user audio segments")

        audio_tensors = []
        texts = []

        for i, (supervision, cut) in enumerate(zip(user_supervisions, user_cuts)):
            user_cut = cut.truncate(offset=supervision.start, duration=supervision.duration)
            audio = user_cut.load_audio()

            # Convert multi-channel audio to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Ensure tensor type
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            # Resample if needed
            target_sample_rate = 16000
            if source_sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=source_sample_rate,
                    new_freq=target_sample_rate
                )
                audio = resampler(audio)

            audio_tensors.append(audio)
            texts.append(self._strip_timestamps(supervision.text))

            # Debug log a few samples
            if i < 3:
                logging.info(f"[DEBUG] Original text ({i}): {supervision.text}")
                logging.info(f"[DEBUG] Audio shape: {audio.shape}, duration: {supervision.duration:.2f}s")

        # Run batch alignment
        alignments_batch = self._wav2vec2_batch_align_tensors(audio_tensors, texts)

        for i, alignment_result in enumerate(alignments_batch):
            if alignment_result is not None:
                original_text = user_supervisions[i].text
                timestamped_text = self._convert_wav2vec2_alignment_to_timestamped_text(alignment_result, original_text)
                logging.info(f"[DEBUG] Aligned text ({i}): {timestamped_text}")

                # Show word-level timestamps for first 3 supervisions
                if i < 3:
                    logging.info(f"[DEBUG] Word-level timestamps for supervision {i}:")
                    for word_info in alignment_result:  # <- iterate directly over list
                        if word_info is not None:
                            logging.info(f"  {word_info['word']} {word_info['start']:.3f}s - {word_info['end']:.3f}s")

                # Update supervision text with alignment
                user_supervisions[i].text = timestamped_text
    
    def _wav2vec2_batch_align_tensors(self, audio_tensors: List[torch.Tensor], texts: List[str]) -> List[Optional[List[Dict[str, Any]]]]:
        """
        Perform batch force alignment using wav2vec2 with in-memory audio tensors.
        
        Args:
            audio_tensors: List of audio waveform tensors
            texts: List of text transcripts corresponding to each audio tensor
            
        Returns:
            List of alignment results for each audio tensor
        """
        alignments = []
        
        for audio_tensor, text in zip(audio_tensors, texts):
            try:
                alignment_result = self._wav2vec2_align(audio_tensor, 16000, text)
                alignments.append(alignment_result)
            except Exception as e:
                logging.error(f"Failed to align audio tensor: {e}")
                alignments.append(None)
        
        return alignments
    
    def _wav2vec2_align(self, waveform: torch.Tensor, sample_rate: int, transcript: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform forced alignment using wav2vec2.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            transcript: Text transcript
            
        Returns:
            List of word segments with timing information
        """
        normalized_transcript = self._normalize_transcript(transcript)
        transcript_words = normalized_transcript.split()
        
        if not transcript_words:
            logging.warning(f"No valid words found in transcript: {transcript}")
            return None
        
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        device = torch.device(self.device)
        waveform = waveform.to(device)
        
        with torch.no_grad():
            emission, _ = self.wav2vec2_model(waveform)
        
        tokens = self.wav2vec2_tokenizer(transcript_words)
        token_spans = self.wav2vec2_aligner(emission[0], tokens)
        
        if not token_spans:
            logging.warning(f"No alignment found for transcript: {transcript}")
            return None
        
        word_segments = []
        ratio = waveform.size(1) / emission.size(1) / 16000
        
        for word, spans in zip(transcript_words, token_spans):
            if spans:
                start_time = spans[0].start * ratio
                end_time = spans[-1].end * ratio
                avg_score = sum(span.score * len(span) for span in spans) / sum(len(span) for span in spans)
                
                word_segments.append({
                    'word': word,
                    'start': start_time,
                    'end': end_time,
                    'score': avg_score
                })
        
        return word_segments
    
    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript by removing punctuation and converting to lowercase."""
        text = transcript.lower()
        text = text.replace("'", "'")
        text = re.sub(r"[^a-z' ]", " ", text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _convert_wav2vec2_alignment_to_timestamped_text(self, alignment_result: List[Dict[str, Any]], original_text: str) -> str:
        """
        Convert wav2vec2 alignment results to timestamped text format.
        
        Args:
            alignment_result: List of word segments with timing information
            original_text: Original text without timestamps
            
        Returns:
            Text with timestamp tokens in the format <|start_frame|>word<|end_frame|>
        """
        timestamped_words = []
        for word_seg in alignment_result:
            word = word_seg["word"]
            start_frame = int(word_seg["start"] / self.frame_length)
            end_frame = int(word_seg["end"] / self.frame_length)
            timestamped_words.append(f"<|{start_frame}|> {word} <|{end_frame}|>")
        return " ".join(timestamped_words)
    
    def _strip_timestamps(self, text: str) -> str:
        """Strip timestamp tokens from text."""
        text = re.sub(r'<\|[0-9]+\|>', '', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
