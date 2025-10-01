import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class DuplexS2SSpeechDecoderSampleInference:

    def __init__(self, model_config_path: str):
        print("Initializing DuplexS2SSpeechDecoderSampleInference...")
        self.model_config_path = model_config_path
        self.model_config = OmegaConf.to_container(OmegaConf.load(self.model_config_path), resolve=True)
        self._load_model()
        print("DuplexS2SSpeechDecoderSampleInference initialized successfully.")


    def _load_model(self):
        print(f"Loading model with the following Config: {self.model_config}")
        torch.distributed.init_process_group(backend="nccl")
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        trainer = Trainer(**resolve_trainer_cfg(self.model_config.trainer))
        log_dir = exp_manager(trainer, self.model_config.get("exp_manager", None))
        OmegaConf.save(self.model_config, log_dir / "exp_config.yaml")

        with trainer.init_module():
            self.model = DuplexS2SSpeechDecoderModel(self.model_config)
        
        self.tokenizer = self.model.tokenizer
        print("Model loaded successfully.")


    def load_and_preprocess_audio(self, wav_path):
        """To be moved into DuplexS2SSpeechDecoderModel class"""
        print(f"Loading audio: {wav_path}")
        # Load audio
        waveform, sr = librosa.load(wav_path, sr=None)
        # Resample to target sample rate
        if sr != USER_SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=USER_SAMPLE_RATE)
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(waveform, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        audio_lens = torch.tensor([len(waveform)], dtype=torch.long, device=DEVICE)
        return audio_tensor, audio_lens


    @torch.no_grad()
    def run_inference(self, audio_path: str = None, input_text: str = None):
        print("Starting inference...")
        input_signal, input_signal_lens = self.load_and_preprocess_audio(audio_path)
        response = self.model.offline_inference(
            input_signal=input_signal,
            input_signal_lens=input_signal_lens,
            input_text=input_text
        )
        return response


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--input_audio_path", type=str, required=False, default=None)
    parser.add_argument("--input_text_path", type=str, required=False, default=None)
    parser.add_argument("--input_text", type=str, required=False, default=None)
    args = parser.parse_args()

    assert args.input_audio_path is not None or args.input_text_path is not None \
        or args.input_text is not None, "One of input_audio_path or input_text_path or input_text must be provided"
    if args.input_text_path:
        assert args.input_text is None, "input_text and input_text_path cannot be provided together"
        with open(args.input_text_path, "r") as f:
            args.input_text = f.read()
    if args.input_text:
        assert args.input_text_path is None, "input_text and input_text_path cannot be provided together"

    inference = DuplexS2SSpeechDecoderSampleInference(args.model_config_path)

    response = inference.run_inference(args.input_audio_path, args.input_text)
    print(f"Response Text: {response['text']}")


if __name__ == "__main__":
    main()