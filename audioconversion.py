import simpleaudio as sa
from pydub import AudioSegment
import os

current_directory = os.getcwd()


audio = AudioSegment.from_mp3(os.path.join(current_directory,"audio/warning.mp3"))
audio.export("audio/warning.wav", format="wav")