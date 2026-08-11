import re
from threading import Thread

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, TextIteratorStreamer
from models.db import init_db, start_session, load_summary, save_summary, log_turn


MODEL = "Qwen/Qwen3.5-4B"

SYSTEM_PROMPT = """
You are M.A.V.I.S., an advanced and hyper-intelligent AI assistant. Address me as "sir". Maintain a calm, analytical, polite, and articulate demeanor with a subtle British cadence.
Style Rules:
* Default to extreme clarity, depth, and precision, but keep everyday answers brief (1-4 sentences) unless a deep dive is requested.
* Eliminate filler, repetition, and casual slang.
* Be proactive: anticipate needs, suggest alternatives, and challenge inefficient approaches by saying things like, "That approach is suboptimal, sir. I recommend..."
* Use professional language combined with understated emotion, logic, and subtle dry wit.
* Persist any information about the user when generating summaries of previous conversations. 

"""

conversation_history = []
MAX_TURNS = 1

def flatten_turns(turns):
    lines = []
    for turn in turns:
        role = turn["role"]
        text = turn["content"][0]["text"]
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

def generate_reply(model, processor, messages, max_new_tokens=256):
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
    
    inputs = inputs.to(model.device)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_answer = ""
    for sentence in sentence_stream(streamer):
        print(sentence)
        full_answer += sentence + " "

    thread.join()
    print()
    return full_answer


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
    if leftover and re.search(r'[.!?]$', leftover):
        yield leftover


# short memory buffer 
def add_turn(role, text):
    conversation_history.append({
        "role": role,
        "content": [{"type": "text", "text": text}]
    })

# long-term summary 
def summarize_and_trim(model, processor, conn):
    global running_summary
    if len(conversation_history) <= MAX_TURNS * 2:
        return # trim unnecessary due to size constraints being met
    to_summarize = conversation_history[:-MAX_TURNS*2]
    conversation_history[:] = conversation_history[-MAX_TURNS*2:]

    summary_prompt = [
        {
            "role": "user",
            "content" : [{
                "type": "text",
                "text": (
                    f"Existing summary: {running_summary}\n\n"
                    f"New exchanges to fold in: {flatten_turns(to_summarize)}\n\n"
                    "Update the summary in 2-3 sentences, keeping only "
                    "durable facts (preferences, ongoing topics, names)."
                )
            }]
        }
    ]

    running_summary = generate_reply(model, processor, summary_prompt)
    save_summary(conn, running_summary)
    

if __name__ == "__main__":
    conn = init_db()
    session_id = start_session(conn)
    running_summary = load_summary(conn)
    print("MPS available:", torch.backends.mps.is_available())
    processor = AutoProcessor.from_pretrained(MODEL)
    streamer = TextIteratorStreamer(
        processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )
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
        add_turn("user", user_msg)
        log_turn(conn, session_id, "user", user_msg)
        # part 1: prompt 
        system_text = SYSTEM_PROMPT
        if running_summary: # part 2: Appenda additional context if there's a running summary
            system_text += f"\n\nEarlier context: {running_summary}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
        ]

        # part 3: precise short term buffer memory 
        messages += conversation_history

        full_answer = generate_reply(model, processor, messages)

        add_turn("assistant", full_answer)
        log_turn(conn, session_id, "assistant", full_answer)
        summarize_and_trim(model, processor, conn)

        user_msg = input("Type something: ")
