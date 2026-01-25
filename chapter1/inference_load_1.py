import torch, time
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).cuda().eval()

BATCH = 30
SEQ_MULT = 50 # held constant
ITERS = 100 # held constant
WARMUP = 20 # held constant
MAX_LENGTH = 512   # <<< explicit sequence length control

base = "AI systems performance engineering is about end to end optimization. "
texts = [(base * SEQ_MULT) for _ in range(BATCH)]

# Tokenization with fixed sequence length
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH
)

total_bytes = 0
for k, v in inputs.items():
    b = v.numel() * v.element_size()
    print(k, v.shape, v.dtype, b/1024/1024, "MiB")
    total_bytes += b
print("TOTAL input bytes:", total_bytes/1024/1024, "MiB")

# Move inputs to GPU once
inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

# Warmup (not measured)
with torch.no_grad():
    for _ in range(WARMUP):
        outputs = model(**inputs)

torch.cuda.synchronize()
t0 = time.time()

# Measured region
with torch.no_grad():
    for _ in range(ITERS):
        outputs = model(**inputs)

torch.cuda.synchronize()
t1 = time.time()

print("Input shape:", inputs["input_ids"].shape)  # sanity check
print("Output type:", type(outputs))
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
print(f"Total GPU-forward wall time (iters={ITERS}): {t1 - t0:.4f}s")