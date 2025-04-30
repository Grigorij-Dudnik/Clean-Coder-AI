import os
from src.utilities.print_formatters import print_formatted
from src.utilities.voice_utils import VoiceRecorder
import keyboard
import readline
import sys


recorder = VoiceRecorder()


def user_input(prompt=""):
    print_formatted(prompt + "Or use (m)icrophone to tell, or press Enter for multiline input:", color="cyan", bold=True)
    
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
        
    user_sentence = input()
    if user_sentence == "m":
        if not os.getenv("OPENAI_API_KEY"):
            print_formatted("Set OPENAI_API_KEY to use microphone feature.", color="red")
            user_sentence = input()
        elif recorder.libportaudio_available:
            transcription = record_voice_message()
            if os.getenv("EDIT_TRANSCRIPTION"):
                print_formatted("Edit text or hit Enter to proceed.\n", color="green")
                user_sentence = input_with_preinserted_text(transcription)
            else:
                print(transcription)
                user_sentence = transcription
        else:
            print_formatted(
                "Install 'sudo apt-get install libportaudio2' (Linux) or 'brew install portaudio' (Mac) to use microphone feature.",
                color="red",
            )
            user_sentence = input()
    elif user_sentence == '' or '\n' in user_sentence:  
        if user_sentence:  
            return user_sentence
        print_formatted("Enter your multiline text (end with Ctrl+D on Unix or Ctrl+Z on Windows):", color="green")
        try:
            user_sentence = sys.stdin.read().strip()
        except KeyboardInterrupt:
            return ""
        
    return user_sentence


def record_voice_message():
    recorder.start_recording()
    keyboard.wait("enter", suppress=True)
    recorder.stop_recording()
    print_formatted("Recording finished.", color="green")
    return recorder.transcribe_audio()


def input_with_preinserted_text(text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()

    readline.set_pre_input_hook(hook)
    result = input()
    readline.set_pre_input_hook()
    return result
