import os
import time
import tempfile
import streamlit as st

from core.config import TrainConfig, DEFAULT_BASE_MODEL
from core.data import load_jsonl, validate_dataset
from core.eval import generate_response
from core.model_utils import inspect_model
from core.registry import save_adapter, load_adapter_zip
from core.trainer import train

st.set_page_config(page_title="Agent Dojo", layout="wide")
st.title("Agent Dojo")
st.caption("Bring your model. Train it. Take it home sharper.")
mode = st.sidebar.radio("Mode", ["Train", "Chat with a trained AI"])

# ------------------------------------------------------------------
# TRAIN MODE
# ------------------------------------------------------------------
if mode == "Train":

    st.header("1. Base model")
    st.write(
        "Specify the model you want to specialize. Any Hugging Face causal "
        "language model that supports 4-bit loading will work."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        base_model_id = st.text_input(
            "Hugging Face model ID",
            value=DEFAULT_BASE_MODEL,
            help="Example: meta-llama/Llama-3.2-1B-Instruct",
        )
    with col_b:
        hf_token = st.text_input(
            "HF token (optional)",
            type="password",
            help="Required only for gated or private repositories.",
        )

    if base_model_id and st.button("Inspect model"):
        with st.spinner("Reading model configuration..."):
            try:
                info = inspect_model(base_model_id, hf_token or None)
                st.session_state["model_info"] = info
            except Exception as e:
                st.error(str(e))
                st.session_state.pop("model_info", None)

    info = st.session_state.get("model_info")
    if info:
        st.success(f"Architecture: {info['architecture']}")
        if info.get("parameters_billions"):
            st.write(f"Approximate parameters: {info['parameters_billions']}B")
            if info["parameters_billions"] > 3:
                st.warning(
                    "This model is large. Training requires 24 GB or more "
                    "of GPU memory. Reduce batch size if you hit OOM."
                )

    st.header("2. Curriculum")
    uploaded = st.file_uploader(
        "Upload dataset (JSONL, messages format)", type=["jsonl"]
    )

    tmp_path = None
    if uploaded is not None:
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(
            tmp_dir, f"{int(time.time())}_{uploaded.name}"
        )
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            ds = load_jsonl(tmp_path)
            stats = validate_dataset(ds)
            st.success(
                f"{stats['num_examples']} examples. "
                f"Roles present: {', '.join(stats['roles_present'])}."
            )
            if not stats["has_assistant"]:
                st.error("Dataset must contain assistant messages.")
                st.stop()
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.header("3. Training configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("Epochs", 1, 10, 3)
        lr = st.number_input(
            "Learning rate", 1e-6, 1e-3, 2e-4, format="%.1e"
        )
    with c2:
        batch = st.number_input("Batch size", 1, 8, 2)
        grad_accum = st.number_input("Gradient accumulation", 1, 16, 4)
    with c3:
        lora_r = st.selectbox("LoRA rank", [8, 16, 32, 64], index=1)
        max_seq = st.selectbox(
            "Max sequence length", [512, 1024, 2048], index=1
        )

    ready = (uploaded is not None) and bool(base_model_id)

    if st.button("Start training", type="primary", disabled=not ready):
        cfg = TrainConfig(
            base_model=base_model_id,
            hf_token=hf_token or None,
            epochs=int(epochs),
            learning_rate=float(lr),
            batch_size=int(batch),
            grad_accum=int(grad_accum),
            lora_r=int(lora_r),
            lora_alpha=int(lora_r) * 2,
            max_seq_length=int(max_seq),
        )
        with st.spinner("Training..."):
            try:
                model, tokenizer, logs, metadata = train(tmp_path, cfg)
                run_id = f"run_{int(time.time())}"
                zip_path = save_adapter(
                    model, tokenizer, run_id, metadata
                )
                st.success("Training complete.")
                st.write(f"Base model: {metadata['base_model']}")
                st.write(
                    f"Target modules: {', '.join(metadata['target_modules'])}"
                )
                st.write(
                    f"Duration: {metadata['duration_seconds']} seconds"
                )
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "Download adapter archive",
                        data=f,
                        file_name=f"{run_id}_adapter.zip",
                        mime="application/zip",
                    )
            except Exception as e:
                st.error(f"Training failed: {e}")

# ------------------------------------------------------------------
# CHAT MODE
# ------------------------------------------------------------------
elif mode == "Chat with a trained AI":

    st.header("Load a trained adapter")
    st.write(
        "Upload an adapter archive produced by this platform. The base "
        "model recorded during training will be loaded automatically."
    )

    adapter_zip = st.file_uploader(
        "Adapter archive (.zip)", type=["zip"]
    )
    hf_token = st.text_input(
        "HF token (if the base model is private)", type="password"
    )

    if adapter_zip is not None and st.button("Load adapter"):
        with st.spinner("Loading base model and adapter..."):
            try:
                model, tokenizer, metadata = load_adapter_zip(
                    adapter_zip.getvalue(), hf_token or None
                )
                st.session_state["chat_model"] = model
                st.session_state["chat_tok"] = tokenizer
                st.session_state["chat_meta"] = metadata
                st.success(
                    f"Loaded. Base model: {metadata['base_model']}"
                )
            except Exception as e:
                st.error(str(e))

    if "chat_model" in st.session_state:
        prompt = st.chat_input("Ask your model")
        if prompt:
            st.chat_message("user").write(prompt)
            with st.spinner("Generating..."):
                reply = generate_response(
                    st.session_state["chat_model"],
                    st.session_state["chat_tok"],
                    prompt,
                )
            st.chat_message("assistant").write(reply)