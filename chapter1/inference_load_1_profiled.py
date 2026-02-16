import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer, AutoModel
from torch.profiler import tensorboard_trace_handler

def add_scopes(model):
    """
    Wrap forward() of key modules with profiler scopes so we can attribute CUDA kernels
    to ATTENTION vs MLP.
    Works for Hugging Face BERT encoder models.
    """
    # Import classes lazily to avoid version issues
    from transformers.models.bert.modeling_bert import (
        BertSelfAttention,
        BertAttention,
        BertIntermediate,
        BertOutput,
        BertLayer,
    )

    def wrap_forward(module, scope_name: str):
        orig_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            with record_function(scope_name):
                return orig_forward(*args, **kwargs)

        module.forward = wrapped_forward

    for name, m in model.named_modules():
        # Self-attention core (QKV, attention scores, context)
        if isinstance(m, BertSelfAttention):
            wrap_forward(m, f"SCOPE/ATTENTION_CORE:{name}")

        # Attention output projection + residual + layernorm lives in BertAttention
        elif isinstance(m, BertAttention):
            wrap_forward(m, f"SCOPE/ATTENTION_BLOCK:{name}")

        # FFN up-projection (hidden -> intermediate)
        elif isinstance(m, BertIntermediate):
            wrap_forward(m, f"SCOPE/MLP_UP:{name}")

        # FFN down-projection + residual + layernorm
        elif isinstance(m, BertOutput):
            wrap_forward(m, f"SCOPE/MLP_DOWN:{name}")

        # Optional: whole layer scope (handy for structure)
        elif isinstance(m, BertLayer):
            wrap_forward(m, f"SCOPE/LAYER:{name}")

    return model


def main():
	#example cmd: BATCH=32 MAX_LENGTH=64  WARMUP=10 ITERS=50  python inference_load_1_profiled.py
    torch.set_grad_enabled(False)

    model_name = "bert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    model = add_scopes(model)

    # ---- knobs ----
    BATCH = int(os.environ.get("BATCH", "32"))
    MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
    WARMUP = int(os.environ.get("WARMUP", "10"))
    ITERS = int(os.environ.get("ITERS", "50"))
    # --------------

    base = "AI systems performance engineering is about end to end optimization. "
    # Ensure we can fill MAX_LENGTH
    texts = [(base * 100) for _ in range(BATCH)]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    # Warmup (not profiled)
    for _ in range(WARMUP):
        _ = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    # Profile
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    ) as prof:
        for _ in range(ITERS):
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()

    # Print a summary table (GPU time per op/scope)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=2000))

if __name__ == "__main__":
    main()
