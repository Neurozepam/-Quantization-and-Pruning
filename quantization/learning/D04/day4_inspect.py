import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to("cuda")
model.eval()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if name.startswith("model.layers.0") or name.startswith("lm_head"):
            print(name, module.weight.shape)