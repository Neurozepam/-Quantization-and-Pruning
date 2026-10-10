import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to("cuda")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def stable(x):
    if len(x.flatten().unique()) == 1:
        return False
    
    return True

def summarize(x):
    return {
        "min": x.min().item(),
        "max": x.max().item(),
        "mean": x.mean().item(),
        "abs_max": x.abs().max().item(),
        "std": x.std(correction=0).item(),
        "p99": torch.quantile(x.abs().flatten(), 0.99).item(),
        "p99_9": torch.quantile(x.abs().flatten(), 0.999).item(),
        "kurtosis": torch.mean( ((x - x.mean()) / x.std(correction=0).item()) ** 4).item() if stable(x) else float("nan"),
    }

stats = {}
handlers = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        def inspect_hook(module, inputs, output, layer_name=name):
            stats[layer_name] = summarize(inputs[0])

        handle = module.register_forward_hook(inspect_hook)
        handlers.append(handle)

tokens = tokenizer(
    "The traveler reached a quiet village before sunset.",
    return_tensors="pt",
).to("cuda")

try:
    with torch.inference_mode():
        result = model(**tokens, use_cache=False)
finally:
    for handle in handlers:
        handle.remove()

print("Collected layers:", len(stats))
print("First q_proj:", stats.get("model.layers.0.self_attn.q_proj"))
print("LM head:", stats.get("lm_head"))