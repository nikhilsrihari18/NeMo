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
import warnings

import torch
import torch.nn.functional as F
import re
from torch.nn.utils.rnn import pad_sequence

def get_pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad
    if pad_id is not None:
        return pad_id
    pad_id = tokenizer.unk_id
    if pad_id is not None:
        return pad_id
    warnings.warn(
        "The text tokenizer has no <pad> or <unk> tokens available, using ID 0 for padding (this may lead to silent bugs)."
    )
    return 0

def collate_and_pad_1d(data, pad_id=-1):
    """
    Collate and pad 1D sequences for batch processing.
    
    This function takes a list of variable-length sequences and pads them
    to the same length using PyTorch's pad_sequence utility.
    
    Args:
        data (List[List[int]]): List of sequences, where each sequence is a list of integers.
                               Each sequence can have different lengths.
        pad_id (int, optional): Padding value to use for shorter sequences. Defaults to -1.
    
    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, max_sequence_length]
                     where batch_size = len(data) and max_sequence_length is the
                     length of the longest sequence in the input.
    
    Example:
        Input:  [[43, 399], [203], [64, 481]]
        Output: tensor([[ 43, 399],
                       [203,  -1],
                       [ 64, 481]])
    """
    return pad_sequence([torch.tensor(seq) for seq in data], batch_first=True, padding_value=pad_id)


def collate_and_pad_2d(tensors, pad_id):
    """
    Collate and pad 2D tensors for batch processing.
    
    This function takes a list of 2D tensors with different shapes and pads them
    to the same size using PyTorch's F.pad utility.
    
    Args:
        tensors (List[torch.Tensor]): List of 2D tensors with varying shapes.
        pad_id (int): Padding value to use for shorter dimensions.
    
    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, max_rows, max_cols]
                     where batch_size = len(tensors), max_rows and max_cols
                     are the maximum dimensions across all input tensors.
    
    Example:
        Input:  [tensor([[1, 2], [3, 4]]), tensor([[5, 6, 7]])]
        Output: tensor([[[1, 2, 0],
                        [3, 4, 0]],
                       [[5, 6, 7],
                        [0, 0, 0]]])
    """
    # Find max dimensions across all tensors
    max_rows = max(t.shape[0] for t in tensors)
    max_cols = max(t.shape[1] for t in tensors)

    # Pad each tensor to (max_rows, max_cols) using F.pad
    # F.pad format: (pad_left, pad_right, pad_top, pad_bottom) for 2D
    padded_tensors = [
        F.pad(t, (0, max_cols - t.shape[1],  # Pad right side (columns)
                 0, max_rows - t.shape[0]),  # Pad bottom side (rows)
            value=pad_id)  # Padding value
        for t in tensors
    ]

    # Stack into a single batch
    return torch.stack(padded_tensors)

def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id=4033, n_speech_codebooks=4, decoder_reduction_factor=1):
    """
    Create a 3D empty tensor for compatibility with legacy code.
    
    This function is maintained for backward compatibility with older versions
    of the codebase.
    
    """
    return torch.cat(
        [
            torch.full((batch_size, length, 1), text_fill_id),
            torch.full(
                (batch_size, length, n_speech_codebooks * decoder_reduction_factor), speech_fill_id
            ),
        ],
        dim=2,
    )

def collate_and_pad(inputs,text_pad_id,speech_pad_id=4033, n_speech_codebooks=4, decoder_reduction_factor=1):
    token_lengths = [len(seq) for seq in inputs]
    if len(inputs[0].shape) < 2:  # 1D sequences
        tokens = pad_sequence(inputs, batch_first=True, padding_value=text_pad_id)
    else:  # 2D sequences
        max_length = max(token_lengths)
        tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, speech_pad_id, n_speech_codebooks, decoder_reduction_factor)
        for i in range(len(tokens)):
            tokens[i, : token_lengths[i], :] = inputs[i]
    
    return tokens, torch.LongTensor(token_lengths)


def is_duplicate_sample_id(sample_id):
    """Check if sample_id is a duplicate (contains '_dup' pattern)."""
    return re.search(r"^.+_dup\d+", sample_id) is not None


def deduplicate_results(results):
    """Remove duplicate results where the prefix exists in the dataset."""
    if not results:
        return results
    
    # Collect all sample_ids
    sample_ids = set()
    for result in results:
        sample_id = result.get('sample_id', '')
        sample_ids.add(sample_id)
    
    # Filter out duplicates
    filtered_results = []
    for result in results:
        sample_id = result.get('sample_id', '')
        if is_duplicate_sample_id(sample_id):
            prefix = sample_id.split('_dup')[0]
            if prefix in sample_ids:
                continue  # Remove this duplicate
        filtered_results.append(result)
    
    return filtered_results
