import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "adapters"))


def save_adapter(model, tokenizer, run_id: str, metadata: dict) -> Path:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    out = ADAPTER_DIR / run_id
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    zip_path = shutil.make_archive(str(out), "zip", str(out))
    return Path(zip_path)


def load_adapter_zip(zip_bytes: bytes, hf_token: str | None = None):
    """Load an adapter and the base model it was trained against."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tmp = tempfile.mkdtemp()
    zpath = Path(tmp) / "adapter.zip"
    zpath.write_bytes(zip_bytes)

    extract_dir = Path(tmp) / "extracted"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(extract_dir)

    meta_path = extract_dir / "training_metadata.json"
    if not meta_path.exists():
        raise ValueError(
            "Adapter archive is missing training_metadata.json. "
            "It was not produced by this platform."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    base_model_id = metadata["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(extract_dir), token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(extract_dir))
    model.eval()
    return model, tokenizer, metadata