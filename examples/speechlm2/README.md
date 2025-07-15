# Duplex Speech-to-Speech Model

This repository contains the implementation of a **Duplex Speech-to-Speech (S2S) model**, designed for real-time, full-duplex conversational AI. The model can simultaneously process incoming user speech while generating a spoken response, creating a more natural and fluid interaction.

This implementation is built using the **NVIDIA NeMo toolkit** and leverages powerful pretrained models for its core components.

---

## Model Architecture

The `DuplexS2SModel` is an auto-regressive, multi-modal model that predicts both **text** and **discrete audio codes** in a single forward pass. It integrates a Large Language Model (LLM) as the central processing unit, augmented with specialized speech and audio components.

### Key Components:

1.  **Pretrained LLM Core**: The model uses a pretrained decoder-only Large Language Model (e.g., from the Llama/Mistral family) as its foundation. We use the transformer blocks but replace the standard text embedding and language model head.
2.  **Speech Encoder (Perception)**: A pretrained Automatic Speech Recognition (ASR) encoder (like a Conformer or FastConformer) processes the incoming source (user) audio stream. It converts the raw waveform into a sequence of hidden-state embeddings.
3.  **Audio Codec**: A pretrained neural audio codec (e.g., Encodec, SoundStream) is used to discretize the target (assistant) audio into a sequence of integer codes from multiple codebooks. These codes serve as the prediction target for the audio modality.
4.  **Multimodal Embeddings**: The model's input at each timestep is a sum of three embeddings:
    * **Source Speech Embedding**: The output of the Speech Encoder for the current audio frame.
    * **Target Text Embedding**: The embedding of the previously generated text token.
    * **Target Audio Embedding**: The sum of embeddings for the previously generated audio codes (one from each codebook).
5.  **Dual Output Heads**: The LLM's output hidden state is fed into two separate linear heads:
    * **LM Head**: Predicts the probability distribution over the next text token in the vocabulary.
    * **Audio Head**: Predicts the probability distributions for the next set of audio codes, one for each codebook of the audio codec.

During training, the model is optimized on a combined loss function, which is a weighted sum of the cross-entropy loss for text prediction and the cross-entropy loss for audio code prediction.

---

## Data Preparation

The model is trained on conversational data where each sample represents a full conversation turn. The data is expected to be in a NeMo-style manifest format, which is a **JSONL file** (one JSON object per line).

Each JSON object must contain the source (user) audio, the target (assistant) audio, and detailed supervision information.

### Manifest Entry Example:

Here is an example of a single entry in the training manifest:

```json
{
  "id": "conv_000000-0",
  "start": 0,
  "duration": 69.92,
  "channel": 0,
  "supervisions": [
    {
      "id": "conv_000000-0",
      "recording_id": "conv_000000-0",
      "start": 0,
      "duration": 6.62,
      "channel": 0,
      "text": "So, I've noticed our local radio station is switching from classic hits to adult hits. What do you think about that?",
      "language": "EN",
      "speaker": "User"
    },
    {
      "id": "conv_000000-0",
      "recording_id": "conv_000000-0",
      "start": 6.62,
      "duration": 4.9,
      "channel": 0,
      "text": "That's an interesting change. The station might be trying to appeal to a broader audience with this format.",
      "language": "EN",
      "speaker": "Assistant"
    }
  ],
  "recording": {
    "id": "user_conv_000000-0",
    "sources": [
      {
        "type": "file",
        "channels": [ 0 ],
        "source": "/path/to/user_audio/user_conv_000000-0.wav"
      }
    ],
    "sampling_rate": 24000,
    "num_samples": 1678080,
    "duration": 69.92
  },
  "custom": {
    "target_audio": {
      "id": "agent_conv_000000-0",
      "sources": [
        {
          "type": "file",
          "channels": [ 0 ],
          "source": "/path/to/assistant_audio/agent_conv_000000-0.wav"
        }
      ],
      "sampling_rate": 24000,
      "num_samples": 1678080,
      "duration": 69.92
    }
  },
  "type": "MonoCut"
}
```
## How to Run

Follow these steps to launch a training job.

---

### 1. Setup Environment

First, set the required environment variables:

```bash
# Add NeMo to your Python path
export PYTHONPATH="/path/to/your/NeMo:$PYTHONPATH"

# Set a home directory for HuggingFace models to be cached
export HF_HOME="/path/to/hfcache/"

# (Optional) Set W&B to offline mode if you don't want to log to the cloud
export WANDB_MODE=offline
```


### 2. Run Training

Execute the training script with the appropriate configuration. The main training script uses a YAML configuration file (conf/s2s_duplex.yaml) and allows for overriding parameters from the command line.
```bash
python s2s_duplex_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    model.pretrained_audio_codec="/path/to/your/audio_codec.nemo" \
    exp_manager.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    data.train_ds.input_cfg='/path/to/your/train_manifest.jsonl' \
    ++data.validation_ds.datasets.val_set_0.shar_path='/path/to/your/validation_manifest.jsonl'
```
### 3.Parameter Explanation

| Argument                                         | Description                                                                 |
|--------------------------------------------------|-----------------------------------------------------------------------------|
| `--config-path`                                  | Path to the directory containing the YAML configuration files.             |
| `--config-name`                                  | Name of the main YAML configuration file (without the `.yaml` extension).  |
| `model.pretrained_audio_codec`                   | **Required.** Path to the pretrained NeMo model file for audio codec.      |
| `exp_manager.name`                               | Name of the experiment (used for logging).                                 |
| `exp_manager.wandb_logger_kwargs.name`           | Name of the run as it appears on Weights & Biases.                         |
| `exp_manager.explicit_log_dir`                   | Directory where training artifacts (checkpoints, logs) will be saved.      |
| `data.train_ds.input_cfg`                        | Path to the training manifest `.jsonl` file.                               |
| `++data.validation_ds.datasets.val_set_0...`     | Path to the validation manifest `.jsonl` file. Uses `++` to override config. |

---

## Citation

This model architecture is based on the work described in the following paper.  
If you use this code or model in your research, please consider citing:

```bibtex
@article{hu2025efficient,
  title={Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model},
  author={Hu, Ke and Hosseini-Asl, Ehsan and Chen, Chen and Casanova, Edresson and Ghosh, Subhankar and {\.Z}elasko, Piotr and Chen, Zhehuai and Li, Jason and Balam, Jagadeesh and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2505.15670},
  year={2025}
}
