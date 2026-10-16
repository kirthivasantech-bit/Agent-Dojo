import time
import inspect
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from core.config import TrainConfig
from core.data import load_jsonl, format_for_training
from core.model_utils import detect_target_modules


def _sft_config_kwargs(base_kwargs: dict, seq_len: int, warmup_steps: int) -> dict:
    """Return SFTConfig kwargs compatible with the installed TRL version."""
    params = inspect.signature(SFTConfig.__init__).parameters

    if "max_length" in params:
        base_kwargs["max_length"] = seq_len
    else:
        base_kwargs["max_seq_length"] = seq_len

    base_kwargs["warmup_steps"] = warmup_steps
    base_kwargs.pop("warmup_ratio", None)
    return base_kwargs


def _load_base(cfg: TrainConfig):
    use_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, token=cfg.hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_gpu:
        # 4-bit quantization path (GPU only)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb,
            device_map="auto",
            token=cfg.hf_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # CPU fallback: no quantization, float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            token=cfg.hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    return model, tokenizer, use_gpu


def train(dataset_path: str, cfg: TrainConfig, progress_callback=None):
    started = time.time()

    model, tokenizer, use_gpu = _load_base(cfg)

    target_modules = cfg.target_modules or detect_target_modules(model)

    dataset = load_jsonl(dataset_path)
    dataset = format_for_training(dataset, tokenizer, cfg.max_seq_length)

    if len(dataset) >= 20:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        do_eval = True
    else:
        train_ds, eval_ds = dataset, None
        do_eval = False

    steps_per_epoch = max(
        1, len(train_ds) // (cfg.batch_size * cfg.grad_accum)
    )
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="steps" if do_eval else "no",
        eval_steps=50 if do_eval else None,
        dataset_text_field="text",
        packing=False,
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
        bf16=use_gpu,                    # only on GPU
        fp16=False,
        use_cpu=not use_gpu,             # explicit CPU flag
        report_to="none",
    )

    sft_kwargs = _sft_config_kwargs(
        sft_kwargs,
        seq_len=cfg.max_seq_length,
        warmup_steps=warmup_steps,
    )

    sft_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    metadata = {
        "base_model": cfg.base_model,
        "target_modules": target_modules,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "max_seq_length": cfg.max_seq_length,
        "epochs": cfg.epochs,
        "num_examples": len(dataset),
        "trained_at": int(started),
        "duration_seconds": int(time.time() - started),
        "tokenizer_name": tokenizer.name_or_path,
        "device": "gpu" if use_gpu else "cpu",
    }

    return trainer.model, tokenizer, trainer.state.log_history, metadata