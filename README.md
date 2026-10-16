# Agent Dojo

A fine-tuning dojo for small language models. Bring a Hugging Face model
and a curriculum dataset. Leave with a trained LoRA adapter that has
learned your data.

---

## Table of Contents

1. Overview
2. How It Works
3. Architecture
4. Project Structure
5. Requirements
6. Installation
7. Configuration
8. Running the Application
9. Dataset Format
10. Sample Datasets
11. Training Workflow
12. Inference Workflow
13. Hardware Requirements
14. GPU vs CPU
15. Known Limitations
16. Roadmap
17. Contributing
18. License

---

## 1. Overview

Agent Dojo is a fine-tuning platform for small language models. It takes a
Hugging Face model and a small instruction dataset, trains a LoRA adapter on
top of that model, and returns the adapter as a downloadable archive.

The platform removes the usual friction of fine-tuning. There is no need to
write training scripts, configure quantization, manage checkpoints, or set
up evaluation. The entire workflow runs through a web interface.

The primary use cases are:

- Specializing a general model on a narrow domain such as customer support,
  code explanation, or tutoring
- Rapid experimentation with fine-tuning without building a training stack
- Teaching the fine-tuning workflow end to end

The project is deliberately small. It is not a replacement for production
training infrastructure. It is a working, understandable implementation of
the core loop: dataset in, adapter out.

---

## 2. How It Works

Agent Dojo operates in two phases.

### Training Phase

1. The user provides a Hugging Face model ID.
2. The user uploads a dataset in JSONL format using the `messages` schema.
3. The system validates the dataset and reports the number of examples.
4. The base model is loaded. On a GPU, it is loaded in 4-bit precision to
   reduce memory usage. On CPU, it is loaded in float32.
5. A LoRA adapter is attached to the model. Target modules are detected from
   the model architecture automatically.
6. The adapter is trained on the dataset. Only the adapter weights are
   updated. The base model remains frozen.
7. The trained adapter, its tokenizer, and a metadata file are packaged into
   a ZIP archive.
8. The user downloads the archive.

### Inference Phase

1. The user uploads a previously trained adapter archive.
2. The system reads `training_metadata.json` from inside the archive and
   extracts the base model ID recorded at training time.
3. That base model is loaded and the adapter is attached.
4. The user chats with the model through the interface.

Because the base model is never modified, one base model can serve any number
of adapters. This is the same pattern used by production multi-tenant
inference systems.

---

## 3. Architecture

The application is organized around two planes that share only the adapter
artifact and the base model identifier.

```
   Dataset (.jsonl)
        |
        v
   +----------------+      +-----------------+      +------------------+
   |  Data Ingest   | ---> |  LoRA Trainer   | ---> |  Adapter (.zip)  |
   +----------------+      +-----------------+      +------------------+
                                                             |
                                                             v
                                                    +------------------+
                                                    |  Chat Interface  |
                                                    +------------------+
```

The base model is selected per job in the interface. It is not fixed at
deployment time. This allows the same installation to serve any model that
supports 4-bit loading on GPU or float32 on CPU.

---

## 4. Project Structure

```
agent-dojo/
├── app.py                     Streamlit user interface
├── core/
│   ├── __init__.py
│   ├── config.py              Training configuration
│   ├── data.py                Dataset loading and validation
│   ├── model_utils.py         Architecture inspection and target modules
│   ├── trainer.py             LoRA training loop
│   ├── eval.py                Response generation
│   └── registry.py            Adapter save and load
├── datasets/
│   ├── customer_support.jsonl Example curriculum
│   ├── code_explainer.jsonl   Example curriculum
│   └── reading_tutor.jsonl    Example curriculum
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Each module in `core/` has a single responsibility. `app.py` is a thin
client that calls into `core/` and renders results. This separation allows
the training logic to be reused in a command line tool, an API server, or a
background job without modification.

---

## 5. Requirements

- Python 3.10 or newer
- PyTorch 2.4 or newer
- Approximately 10 GB of free disk space for the base model and dependencies
- A CUDA-capable GPU for practical training (CPU mode is supported but slow)

The full dependency list is in `requirements.txt`:

```
streamlit>=1.40
torch>=2.4
transformers>=4.46
peft>=0.13
trl>=0.12,<0.13
bitsandbytes>=0.44
datasets>=3.0
accelerate>=1.0
safetensors>=0.4
```

Note that `trl` is pinned below version 0.13. Newer versions renamed
several configuration arguments. The code includes a compatibility helper
so it works across versions, but the pinned range is the tested path.

---

## 6. Installation

Clone the repository and set up a virtual environment.

```bash
git clone https://github.com/YOUR_USERNAME/agent-dojo.git
cd agent-dojo

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify that CUDA is available if you intend to train on a GPU.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, the application will still run, but training will
use CPU and will be significantly slower.

---

## 7. Configuration

All training parameters have sensible defaults. To override them, copy the
example environment file.

```bash
cp .env.example .env
```

The following variables are recognized:

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_BASE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Initial value in the UI |
| `EPOCHS` | `3` | Passes over the training data |
| `BATCH_SIZE` | `2` | Per-device training batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `LR` | `2e-4` | Learning rate |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha, the scaling factor |
| `MAX_SEQ_LEN` | `1024` | Maximum tokens per example |
| `OUTPUT_DIR` | `outputs` | Directory for training checkpoints |
| `ADAPTER_DIR` | `adapters` | Directory for saved adapters |

The default base model is a 0.5B parameter instruction-tuned model. It is
small enough to train on a single consumer GPU and produces coherent output
after fine-tuning. Larger models can be substituted by changing
`DEFAULT_BASE_MODEL` or by typing a different model ID in the interface.

---

## 8. Running the Application

Once dependencies are installed, start the Streamlit server.

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`. The sidebar contains a
mode selector with two options: Train and Chat with a trained AI.

The first run downloads the base model from Hugging Face. This is
approximately 1 GB for the default 0.5B model and may take several minutes
depending on network speed. Subsequent runs use the local cache.

---

## 9. Dataset Format

Datasets must be provided as JSONL, which stands for JSON Lines. Each line
is one training example in the following format.

```json
{"messages":[{"role":"system","content":"You are a helpful tutor."},{"role":"user","content":"What is a closure?"},{"role":"assistant","content":"A closure is a function that remembers variables from the scope where it was defined, even after that scope has returned."}]}
```

Rules:

- Each line must be a single valid JSON object.
- The top-level key must be `messages`.
- `messages` must be an array of objects, each with a `role` and a `content`.
- Valid roles are `system`, `user`, and `assistant`.
- At least one `user` message and one `assistant` message must be present.
- The `system` role is optional but recommended for controlling tone and
  behavior.
- Files must be UTF-8 encoded.

The platform applies the model's chat template automatically. Do not
pre-format your data for a specific model. Provide the raw conversation
structure and let the trainer handle the conversion.

Invalid lines raise an error that identifies the exact line number. Fix the
line and re-upload.

---

## 10. Sample Datasets

Three example curricula are included in the `datasets/` directory. Each is
small enough to train in a few minutes on a mid-range GPU. They are intended
as starting points, not as production data.

### customer_support.jsonl

A support agent for a fictional cloud hosting company. Demonstrates the use
of a `system` prompt to fix tone and behavior across many examples.

### code_explainer.jsonl

A patient code explainer that describes what code does and flags common
pitfalls. Demonstrates technical content and multi-sentence explanations.

### reading_tutor.jsonl

A reading tutor for middle school students. Demonstrates how to encode
pedagogical behavior, such as always ending with a comprehension question.

Each file contains between five and twenty examples. Real curriculums for a
specialized model typically contain hundreds to thousands of examples. The
samples exist to validate the pipeline, not to produce a useful model on
their own.

To use a sample, upload the corresponding file in the Train tab.

---

## 11. Training Workflow

The following steps describe what happens when a training job is submitted.

1. The uploaded file is written to a temporary location and parsed line by
   line. Invalid lines raise an error identifying the line number.
2. The base model and tokenizer are loaded. The tokenizer is configured with
   a padding token if one is not already defined.
3. On a GPU, the model is quantized to 4-bit using the NF4 format with double
   quantization. This reduces memory usage by roughly four times compared to
   bfloat16 while retaining most of the model quality. On CPU, the model is
   loaded in float32 without quantization.
4. A LoRA configuration is created. Target modules are detected by inspecting
   the loaded model for its linear layers. This works across most modern
   transformer architectures.
5. The dataset is split into training and evaluation sets if it contains at
   least twenty examples. Smaller datasets are used entirely for training.
6. The SFTTrainer from the TRL library runs the training loop. Only the LoRA
   parameters are updated. The base model remains frozen.
7. After training, the adapter weights, tokenizer, and a metadata file are
   saved to disk and packaged into a ZIP archive.
8. The archive is offered as a download.

The entire process runs in the same process as the Streamlit app. There is
no separate worker or queue.

---

## 12. Inference Workflow

To interact with a trained adapter, switch to the Chat with a trained AI
mode in the sidebar and upload the adapter archive that was previously
downloaded.

The system performs the following steps.

1. Extracts the archive to a temporary directory.
2. Reads `training_metadata.json` to determine which base model was used
   during training.
3. Loads that base model.
4. Attaches the adapter using the PEFT library.
5. Enables a chat interface for sending prompts and viewing responses.

The base model is determined by the metadata inside the archive, not by the
current configuration. This ensures that the correct base model is always
loaded, even if the default has changed since training.

---

## 13. Hardware Requirements

The following table summarizes approximate hardware requirements for
training with the default settings.

| Model size | Minimum GPU memory | Notes |
|---|---|---|
| 0.5B | 8 GB | Fast iteration, suitable for demos |
| 1B to 1.5B | 12 GB to 16 GB | Several minutes for 3 epochs |
| 3B | 24 GB | Recommended for production use |
| 7B | 40 GB or more | Requires gradient checkpointing |

On cloud platforms, an NVIDIA T4 is the minimum viable option. An A10G or
L4 offers a substantially better experience for larger models.

---

## 14. GPU vs CPU

The application detects whether a CUDA GPU is available at runtime and
adjusts the training configuration accordingly.

On GPU, the model is loaded in 4-bit precision using bitsandbytes, the
optimizer is `paged_adamw_8bit`, and bfloat16 is used for computation.

On CPU, the model is loaded in float32, the optimizer is `adamw_torch`, and
bfloat16 is disabled. No quantization is applied because bitsandbytes
requires a GPU.

CPU training is supported but is not practical for real workloads. A 0.5B
model on a small dataset may take tens of minutes to an hour per epoch. For
models above 1B parameters, CPU training is generally unusable. The CPU
path exists so that the pipeline can be validated on a laptop without a
GPU. For real training, use a cloud GPU.

---

## 15. Known Limitations

The following are acknowledged limitations of the current version.

- Only one training job can run at a time. There is no queue.
- There is no authentication. Do not deploy this publicly without adding an
  authentication layer in front of the Streamlit app.
- Adapter only. Full model upload is not supported. Models must be reachable
  by Hugging Face model ID.
- No persistent storage. Adapters are written to local disk and are lost if
  the container restarts.
- No concurrent serving of multiple adapters from a single base model. Each
  loaded adapter occupies its own process and its own copy of the base
  weights.
- Evaluation is limited to a fixed set of sanity-check prompts. There is no
  task-specific metric.

Each of these has a well-understood solution and most are addressed in the
roadmap.

---

## 16. Roadmap

Planned improvements, roughly in order of priority.

- Job queue backed by Celery and Redis, allowing concurrent training jobs
- Authentication and per-user adapter storage on object storage such as S3
- vLLM serving with dynamic LoRA routing for multi-tenant inference
- A structured evaluation report comparing base and fine-tuned model output
- A catalog of preset curriculums that users can select without uploading
  their own data

Contributions toward any of these are welcome.

---

## 17. Contributing

Issues and pull requests are welcome. For larger changes, please open an
issue first to discuss the approach.

Before submitting a pull request:

- Run the application locally and confirm both training and inference still
  work.
- Keep changes scoped. One concern per pull request.
- Update this README if behavior or configuration changes.

There is no formal coding style guide. Match the style of the existing code.

---

## 18. License

This project is released under the MIT License. See the LICENSE file for the
full text.