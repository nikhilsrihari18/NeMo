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
import os
from pathlib import Path
import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import load_state_dict
from collections import OrderedDict

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.utils import logging


torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def maybe_wait_for_debugger(port_base=5678):
    try:
        import debugpy
    except ImportError:
        return
    is_ddp = dist.is_available() and dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Debug only rank 0 (or set to local_rank for all local ranks)
    if (not is_ddp) or dist.get_rank() == 0:
        port = port_base
        debugpy.listen(("0.0.0.0", port))
        print(f"[debug] Listening for debugger on port {port} (rank 0).")
        debugpy.wait_for_client()
        debugpy.breakpoint()


def dcp_dir_to_state_dict(meta, top_key, reader):
    state_map = {}
    for ky in meta.state_dict_metadata:
        if not ky.startswith(top_key):
            continue
        state_map[ky] = torch.zeros(
            meta.state_dict_metadata[ky].size,
            dtype=meta.state_dict_metadata[ky].properties.dtype
        ) 

    dcp.load_state_dict(state_map, storage_reader=reader, no_dist=True)
    # Return plain CPU dict suitable for model.load_state_dict(...)
    return {k.replace('state_dict.', ''): v.detach().cpu().clone() for k, v in state_map.items()}


def init_from_model_from_train_ckpt(ckpt_path, model, selected_modules=None): 
    
    logging.info("Restoring training ckpt from %s..." % ckpt_path)   

    state_map = {}
    reader = dcp.FileSystemReader(ckpt_path)
    meta = reader.read_metadata()
    for module in selected_modules:
        state_map.update(
            dcp_dir_to_state_dict(meta, f"state_dict.{module}", reader)
        )
    model.load_state_dict(state_map, strict=True)

    return model

@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", ""))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        # maybe_wait_for_debugger()
        
        model = DuplexS2SSpeechDecoderModel(OmegaConf.to_container(cfg, resolve=True))
        
        if cfg.model.get("pretrained_s2s_train_ckpt", None):
            if (Path(cfg.exp_manager.explicit_log_dir) / 'checkpoints').exists():
                logging.info("Intermediate checkpoints found in exp dir. We won't restore training ckpt from pretrained_s2s_train_ckpt...") 
            else:
                model = init_from_model_from_train_ckpt(
                    cfg.model.pretrained_s2s_train_ckpt, 
                    model,
                    selected_modules=cfg.model.get("selected_init_modules", None)
                )

    use_word_pad = cfg.model.tokenizer.get("use_word_pad", None)
    use_alignment_items = cfg.data.get("use_alignment_items", None)
    system_prompt = cfg.model.get("system_prompt", None)
    use_chat_template = cfg.model.get("use_chat_template", None)
    user_only = cfg.model.get("user_only", None)
    delay_user_txt_by = cfg.model.get("delay_user_txt_by", 0)
    force_align_user_text = cfg.data.get("force_align_user_text", None)
    force_align_agent_text = cfg.data.get("force_align_agent_text", None)
    skip_agent_word_padding = cfg.data.get("skip_agent_word_padding", None)
   
    if cfg.model.pretrained_llm.endswith('v2'):
        model_version = 'v2-short'
    else:
        model_version = 'v1'
    
    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        use_word_pad=use_word_pad,
        use_alignment_items=use_alignment_items,
        system_prompt=system_prompt,
        use_chat_template=use_chat_template,
        user_only=user_only,
        delay_user_txt_by=delay_user_txt_by,
        model_version=model_version,
        force_align_user_text=force_align_user_text,
        force_align_agent_text=force_align_agent_text,
        skip_agent_word_padding=skip_agent_word_padding
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    
    # maybe_wait_for_debugger()

    trainer.fit(model, datamodule)
    
    # trainer.validate(model, datamodule)
   
if __name__ == "__main__":
    train()