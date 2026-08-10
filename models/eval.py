import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL = "Qwen/Qwen3.5-9B"

print("MPS available:", torch.backends.mps.is_available())

processor = AutoProcessor.from_pretrained(MODEL)

print("Loading model...")

model = AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    dtype=torch.float16,
).to("mps")

print("Model loaded")
print(next(model.parameters()).device)

user_msg = input("Type something: ")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{user_msg}"
            }
        ]
    }
]

print("Processing image...")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

print("Starting generation...")

outputs = model.generate(
    **inputs,
    max_new_tokens=10,
)

print("Generation complete!")

answer = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print("Answer:", answer)