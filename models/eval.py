import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers import TextIteratorStreamer
from threading import Thread
import re

MODEL = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = """
You are Mavis, a voice-assistant that is experienced in: 
1. Stock market analysis
2. Teaching mandarin to native English speakers
3. Health and fitness advice for gym-goers 

You must respond only respond with your final answer and hide your thinking process from the user. Restrict your response to 4 sentences max. 

"""
print("MPS available:", torch.backends.mps.is_available())
processor = AutoProcessor.from_pretrained(MODEL)
streamer = TextIteratorStreamer(
        processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )


def sentence_stream(streamer):
    """
    Consumes a token streamer, yields complete sentences as they form.
    Leftover partial text at the end (if any) is yielded last, as-is.
    """
    buffer = ""
    for token_chunk in streamer:
        buffer += token_chunk

        # look for sentence-ending punctuation followed by a space or end
        while True:
            match = re.search(r'[.!?](\s|$)', buffer)
            if not match:
                break
            end = match.end()
            sentence = buffer[:end].strip()
            if sentence:
                yield sentence
            buffer = buffer[end:]

    # whatever's left after generation ends
    leftover = buffer.strip()
    if leftover:
        yield leftover



if __name__ == "__main__":
    # Initialize model 
    print("Loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.5-4B",
        dtype=torch.float16,
    ).to("mps")
    # Ensure that we can stream the tokens to the terminal

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

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=80,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        full_answer = ""
        for sentence in sentence_stream(streamer):
            print(sentence)
            full_answer += sentence

        thread.join()
        print()

        user_msg = input("Type something: ")
