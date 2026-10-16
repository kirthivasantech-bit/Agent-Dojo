import torch
from transformers import AutoConfig, AutoTokenizer

# Modules we prefer to target when present. Covers Llama, Qwen, Mistral,
# Gemma, Phi, and most modern decoder-only transformers.
PREFERRED_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Fallback for older or non-standard architectures.
FALLBACK_TARGETS = [
    "query", "key", "value", "dense",
    "fc1", "fc2", "c_attn", "c_proj", "Wqkv",
]


def detect_target_modules(model) -> list[str]:
    """Inspect a loaded model and return the linear layers to adapt.

    Called after the model is loaded, so we see the real module names
    regardless of what the config claims.
    """
    linear_leaves = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_leaves.add(name.split(".")[-1])

    found = [n for n in PREFERRED_TARGETS if n in linear_leaves]
    if not found:
        found = [n for n in FALLBACK_TARGETS if n in linear_leaves]

    if not found:
        raise ValueError(
            "Could not detect LoRA target modules for this architecture. "
            f"Available linear layer names: {sorted(linear_leaves)}"
        )
    return found


def inspect_model(model_id: str, hf_token: str | None = None) -> dict:
    """Read a model's config without loading weights.

    Used for pre-flight UI feedback: architecture, rough parameter count,
    and estimated memory footprint.
    """
    try:
        config = AutoConfig.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )
    except Exception as e:
        raise ValueError(f"Could not load model config for '{model_id}': {e}")

    arch = (config.architectures or ["unknown"])[0]

    # Rough parameter estimate from common config fields. Not exact, but
    # accurate enough to warn about memory before the user commits.
    hidden = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    vocab = getattr(config, "vocab_size", None)
    inter = getattr(config, "intermediate_size", hidden * 4 if hidden else None)

    params_b = None
    if hidden and layers and vocab:
        # attention (4 * h^2) + MLP (3 * h * inter) per layer + embeddings
        per_layer = 4 * hidden * hidden + 3 * hidden * inter
        total = layers * per_layer + 2 * vocab * hidden
        params_b = round(total / 1e9, 2)

    return {
        "model_id": model_id,
        "architecture": arch,
        "parameters_billions": params_b,
        "hidden_size": hidden,
        "num_layers": layers,
        "vocab_size": vocab,
    }


def validate_tokenizer(model_id: str, hf_token: str | None = None) -> bool:
    """Check that a tokenizer can be loaded. Fails fast on gated repos."""
    try:
        AutoTokenizer.from_pretrained(model_id, token=hf_token)
        return True
    except Exception as e:
        raise ValueError(f"Tokenizer not available for '{model_id}': {e}")