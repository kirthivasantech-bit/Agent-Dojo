import torch

EVAL_PROMPTS = [
    "Hello, how are you?",
    "Explain your purpose in one sentence.",
    "What can you help me with?",
]


@torch.no_grad()
def generate_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 128
) -> str:
    msgs = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        msgs,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        out[0][input_len:], skip_special_tokens=True
    )


def run_sanity_check(base_model, trained_model, tokenizer):
    results = []
    for prompt in EVAL_PROMPTS:
        base_out = generate_response(base_model, tokenizer, prompt)
        tuned_out = generate_response(trained_model, tokenizer, prompt)
        results.append(
            {
                "prompt": prompt,
                "base": base_out.strip(),
                "tuned": tuned_out.strip(),
            }
        )
    return results