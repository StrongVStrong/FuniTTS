import torch
from TTS.api import TTS
from pydub import AudioSegment
import simpleaudio as sa

# Set device (CUDA if GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS with a pre-trained model (e.g., XTTS model)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Specify multiple reference audio samples (paths to your reference voice files)
reference_audio_paths = [
    "C:/Users/Megas/Documents/COquiTTS/TTS/TTS/bin/data/output1.wav",  # First reference audio file
    "C:/Users/Megas/Documents/COquiTTS/TTS/TTS/bin/data/output2.wav",  # Second reference audio file
    "C:/Users/Megas/Documents/COquiTTS/TTS/TTS/bin/data/output3.wav"   # Third reference audio file
    # Add as many samples as you need
]

# The text you want to synthesize
text = "Super Vegito"
language = "en"  # Choose the language corresponding to your reference audio

# Generate speech and save it to a file
output_path = "output_cloned_voice.wav"
tts.tts_to_file(
    text=text,
    speaker_wav=reference_audio_paths,  # Provide a list of reference audio files for cloning
    language=language,
    file_path=output_path
)

# Play the audio
print("Speech synthesis complete. Playing the 'output_cloned_voice.wav' file.")
audio = AudioSegment.from_wav(output_path)
play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
play_obj.wait_done()

print("Playback finished.")
