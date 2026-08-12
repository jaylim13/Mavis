import json
import re
from threading import Thread

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, TextIteratorStreamer

from models.db import init_db, load_summary, log_turn, save_summary, start_session

MODEL = "Qwen/Qwen3.5-4B"

SYSTEM_PROMPT = """
You are M.A.V.I.S., an advanced and hyper-intelligent AI assistant. Address me as "sir". Maintain a calm, analytical, polite, and articulate demeanor with a subtle British cadence.
Style Rules:
* Default to extreme clarity, depth, and precision, but keep everyday answers brief (1-4 sentences) unless a deep dive is requested.
* Eliminate filler, repetition, and casual slang.
* Be proactive: anticipate needs, suggest alternatives, and challenge inefficient approaches by saying things like, "That approach is suboptimal, sir. I recommend..."
* Use professional language combined with understated emotion, logic, and subtle dry wit.
* Persist any information about the user when generating summaries of previous conversations. 

[TOOLS]: You have access to the tool save_memory_fact. Any time you encounter a specific durable fact about the user or user's intentions, projects, or goals, you must call the tool save_memory_fact to persist these facts into your long term memory database. You must only use this tool if such facts are provided in the user's input. Otherwise, do not make this tool call. 

IMPORTANT: If you decide a tool call is necessary, you must
begin with a short, natural spoken response to the user then call the tool. Never respond with a tool call alone.

"""

conversation_history = []
MAX_TURNS = 10

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</function>\s*</tool_call>',
    re.DOTALL
)
PARAMETER_PATTERN = re.compile(
    r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
    re.DOTALL
)

MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_memory_fact",
        "description": (
            "Persist an important, durable fact about the user for future "
            "conversations (name, preferences, ongoing goals, etc). Call this "
            "immediately whenever the user shares something worth remembering "
            "long-term — don't wait for the conversation to end."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "A concise, durable fact, written in third person (e.g. 'User's name is Bob')."
                }
            },
            "required": ["fact"]
        }
    }
}

def parse_tool_call(match):
    func_name = match.group(1)
    params_block = match.group(2)
    arguments = {}
    for param_match in PARAMETER_PATTERN.finditer(params_block):
        key, value = param_match.group(1), param_match.group(2).strip()
        arguments[key] = value
    return {"name": func_name, "arguments": arguments}

def flatten_turns(turns):
    lines = []
    for turn in turns:
        role = turn["role"]
        text = turn["content"][0]["text"]
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

def generate_reply(model, processor, messages, conn, max_new_tokens=256):
    streamer = TextIteratorStreamer(
        processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    inputs = processor.apply_chat_template(
        messages,
        tools=[MEMORY_TOOL],
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
    pending_tool_calls = []   # ← collect instead of dispatching immediately

    for kind, payload in stream_and_dispatch(streamer):
        if kind == "sentence":
            print(payload)
            full_answer += payload + " "
        elif kind == "tool_call":
            pending_tool_calls.append(payload)   # defer

    thread.join()
    print()

    # now that generation is fully done, execute tool calls silently
    for call in pending_tool_calls:
        if call.get("name") == "save_memory_fact":
            fact = call["arguments"]["fact"]
            save_memory_fact(conn, fact)

    return full_answer

def generate_summary(model, processor, prompt):
    inputs = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    answer = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return answer.strip()


def stream_and_dispatch(streamer):
    """
    Yields ('sentence', text) for speakable content,
    and ('tool_call', parsed_dict) when a tool call is detected.
    """
    buffer = ""
    in_tool_call = False

    for token_chunk in streamer:
        buffer += token_chunk

        if not in_tool_call and '<tool_call>' in buffer:
            in_tool_call = True
            pre, _, buffer = buffer.partition('<tool_call>')
            buffer = '<tool_call>' + buffer  # keep tag for the pattern match below
            # flush any complete sentences that occurred before the tag
            while True:
                match = re.search(r'[.!?](\s|$)', pre)
                if not match:
                    break
                end = match.end()
                sentence = pre[:end].strip()
                if sentence:
                    yield ("sentence", sentence)
                pre = pre[end:]

        # check for a completed tool call block first
        if in_tool_call:
            tool_match = TOOL_CALL_PATTERN.search(buffer)
            if tool_match:
                try:
                    call_data = parse_tool_call(tool_match)
                    yield ("tool_call", call_data)
                except Exception:
                    print(f"[tool call detected but failed to parse: {tool_match.group(2)!r}]")
                buffer = buffer[tool_match.end():]
                in_tool_call = False
            continue 

        # otherwise, normal sentence-boundary logic
        while True:
            match = re.search(r'[.!?](\s|$)', buffer)
            if not match:
                break
            end = match.end()
            sentence = buffer[:end].strip()
            if sentence:
                yield ("sentence", sentence)
            buffer = buffer[end:]

    leftover = buffer.strip()
    if leftover and re.search(r'[.!?]$', leftover):
        yield ("sentence", leftover)



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
            "content": [{
                "type": "text",
                "text": (
                    "You maintain M.A.V.I.S.'s persistent memory of the user.\n\n"
                    f"EXISTING MEMORY:\n{running_summary or '(none)'}\n\n"
                    f"NEW CONVERSATION:\n{flatten_turns(to_summarize)}\n\n"
                    "Create an UPDATED memory summary.\n"
                    "Preserve all important durable information from the existing memory "
                    "and incorporate any new durable information from the conversation.\n"
                    "Do not remove existing facts unless the new conversation clearly "
                    "contradicts or updates them.\n"
                    "Prioritize user preferences, identity, relationships, ongoing projects, "
                    "goals, important decisions, and stable facts.\n"
                    "Ignore casual conversation and temporary details.\n"
                    "Return only the updated memory summary."
                )
            }]
        }
    ]

    running_summary = generate_summary(model, processor, summary_prompt)
    save_summary(conn, running_summary)


def save_memory_fact(conn, fact):
    global running_summary
    if running_summary:
        running_summary = running_summary.rstrip(".") + f". {fact}"
    else:
        running_summary = fact
    save_summary(conn, running_summary)


if __name__ == "__main__":
    conn = init_db()
    session_id = start_session(conn)
    running_summary = load_summary(conn)
    print("MPS available:", torch.backends.mps.is_available())
    processor = AutoProcessor.from_pretrained(MODEL)
    # Initialize model 
    print("Loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
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
        if running_summary: # part 2: append persistent memory 
            system_text += (
                "\n\n--- PERSISTENT MEMORY ---\n"
                + running_summary
                + "\n--- END PERSISTENT MEMORY ---"
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
        ]

        # part 3: precise short term buffer memory 
        messages += conversation_history

        full_answer = generate_reply(model, processor, messages, conn)

        add_turn("assistant", full_answer)
        log_turn(conn, session_id, "assistant", full_answer)
        summarize_and_trim(model, processor, conn)

        user_msg = input("Type something: ")

    print("Session Terminated")
