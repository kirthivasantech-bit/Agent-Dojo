"""
Inference service — load a base model + one or more LoRA adapters
and serve chat completions. Designed to be importable from FastAPI,
Streamlit, or a CLI.
"""
import os
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_DEFAULT_BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class InferenceEngine:
    """
    Loads a base model once, then hot-swaps LoRA adapters on top.
    Only one adapter active at a time — sufficient for v1.
    """

    def __init__(self, base_model_id: str = _DEFAULT_BASE):
        self.base_model_id = base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.base_model.eval()
        self.active_model = self.base_model
        self.active_adapter: Optional[str] = None

    def load_adapter(self, adapter_dir: str | Path) -> None:
        """Attach a LoRA adapter from a directory (already extracted)."""
        adapter_dir = str(adapter_dir)
        config_file = Path(adapter_dir) / "adapter_config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"No adapter_config.json in {adapter_dir}. "
                "Did you upload the wrong zip?"
            )
        self.active_model = PeftModel.from_pretrained(self.base_model, adapter_dir)
        self.active_model.eval()
        self.active_adapter = adapter_dir

    def unload_adapter(self) -> None:
        """Revert to base model."""
        self.active_model = self.base_model
        self.active_adapter = None

    @torch.no_grad()
    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Chat with the currently active model (base or base+adapter)."""
        ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.active_model.device)

        out = self.active_model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            out[0][ids.shape[-1]:], skip_special_tokens=True
        ).strip()