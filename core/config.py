import os
from dataclasses import dataclass

# Fallback default. The user overrides this per job in the UI.
DEFAULT_BASE_MODEL = os.getenv("DEFAULT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    hf_token: str | None = None

    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = 0.05

    max_seq_length: int = int(os.getenv("MAX_SEQ_LEN", "1024"))
    epochs: int = int(os.getenv("EPOCHS", "3"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    grad_accum: int = int(os.getenv("GRAD_ACCUM", "4"))
    learning_rate: float = float(os.getenv("LR", "2e-4"))
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 0.3

    # Populated at runtime by detect_target_modules
    target_modules: list[str] | None = None