import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers import TextStreamer

MODEL = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = """
You are Mavis, a voice-assistant that is experienced in: 
1. Stock market analysis
2. Teaching mandarin to native English speakers
3. Health and fitness advice for gym-goers 

You must respond only respond with your final answer and hide your thinking process from the user. Restrict your response to 4 sentences max. 

"""


if __name__ == "__main__":
    print("MPS available:", torch.backends.mps.is_available())
    processor = AutoProcessor.from_pretrained(MODEL)
    print("Loading model...")
    # Initialize model 
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.5-4B",
        dtype=torch.float16,
    ).to("mps")
    # Ensure that we can stream the tokens to the terminal
    streamer = TextStreamer(processor, skip_prompt=True, skip_special_tokens=True)

    print("Model loaded")
    print(next(model.parameters()).device)
    user_msg = input("Type something: ")

    while user_msg != "terminate":
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_msg
                    }
                ]
            }
        ]


        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = inputs.to(model.device)

        print("Starting generation...")

        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            streamer=streamer,
        )

        answer = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        print("Answer:", answer)
        user_msg = input("Type something: ")
