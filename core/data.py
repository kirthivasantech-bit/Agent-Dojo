import json
from pathlib import Path
from datasets import Dataset

def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file with 'messages' format (industry standard)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            if "messages" not in rec:
                raise ValueError(
                    f"Line {line_num}: missing 'messages' key. "
                    f"Expected: {{\"messages\": [{{\"role\": \"user\", ...}}]}}"
                )
            records.append(rec)
    if not records:
        raise ValueError("Dataset is empty.")
    return Dataset.from_list(records)

def format_for_training(dataset: Dataset, tokenizer, max_len: int) -> Dataset:
    """Apply chat template and tokenize with proper masking."""
    def _format(batch):
        texts = []
        for msgs in batch["messages"]:
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            ))
        return {"text": texts}
    return dataset.map(_format, batched=True)

def validate_dataset(dataset: Dataset) -> dict:
    """Return validation stats for the UI."""
    roles = set()
    for msgs in dataset["messages"]:
        for m in msgs:
            roles.add(m.get("role"))
    return {
        "num_examples": len(dataset),
        "roles_present": sorted(roles),
        "has_system_prompt": "system" in roles,
        "has_assistant": "assistant" in roles,
    }