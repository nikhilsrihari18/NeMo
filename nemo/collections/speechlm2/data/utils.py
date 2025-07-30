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
import re

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