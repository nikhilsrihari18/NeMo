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

import sacrebleu
import torch
import torchmetrics
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.utils import logging


class BLEU(torchmetrics.Metric):
    """
    Computes BLEU scores on text predictions.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    
    This is a PyTorch Lightning compatible metric that accumulates references
    and hypotheses across batches and computes corpus-level BLEU scores.
    """

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        # Note: For text metrics that store lists of strings, we cannot use
        # add_state() with tensor types. Instead, we manually manage state.
        # This means distributed training would need custom all_gather logic.
        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        """Reset the metric state for a new epoch/validation phase."""
        self._refs.clear()
        self._hyps.clear()

    def update(self, name: str, refs: list[str], hyps: list[str]) -> None:
        for ref, hyp in zip(refs, hyps):
            self._refs[name].append(self.normalizer(ref))
            self._hyps[name].append(self.normalizer(hyp))
            if self.verbose:
                asrb = sacrebleu.sentence_bleu(hyp, [ref]).score
                logging.info(f"\n\n[REF]\t{ref}\n[HYP]\t{hyp} [{asrb:.2f}]")

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute the corpus-level BLEU scores from accumulated data."""
        corpus_metric = {}
        for name in self._refs.keys():
            metric = torch.tensor(sacrebleu.corpus_bleu(self._hyps[name], [self._refs[name]]).score)
            corpus_metric[f"txt_bleu_{name}"] = metric
        
        if corpus_metric:
            corpus_metric["txt_bleu"] = torch.stack(list(corpus_metric.values())).mean()
        else:
            corpus_metric["txt_bleu"] = torch.tensor(0.0)
        
        return corpus_metric


def _identity(x):
    return x
