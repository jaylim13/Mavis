import mlx_whisper
import sounddevice as sd 
import numpy as np 

SAMPLE_RATE = 16000
DURATION = 5 
WHISPER_MODEL = "mlx-community/whisper-small-mlx"

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Records audio directly from the user's mic and returns it
    Args: duration - how many seconds the audio message can be, sample_rate - how many snapshots of audio are captured per second  
    Returns: audio file 
    """
    print("Listening...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten() 

def transcribe(audio):
    """
    Takes the audio and puts it into words 
    Args: audio - the audio file 
    Returns: the transcription in text
    """
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=WHISPER_MODEL,
    )
    return result["text"].strip()

def listen_and_transcribe():
    """
    Calls helper functions to perform stt 
    Args: None
    Returns: the transcription in text 
    """
    audio = record_audio()
    text = transcribe(audio)
    return text

if __name__ == "__main__":
    print(listen_and_transcribe())