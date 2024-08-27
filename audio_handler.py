import io
import torch
import librosa
from transformers import pipeline

def convert_bytes_to_array(audio_bytes):
    """
       Converts audio bytes into a NumPy array.

       This function takes raw audio data in bytes, converts it into a stream using
       `io.BytesIO`, and then loads it into a NumPy array using `librosa`. The array
       represents the audio waveform, and `librosa` also infers the sample rate of the audio.

       Args:
           audio_bytes (bytes): Raw audio data in bytes.

       Returns:
           np.ndarray: A NumPy array representing the audio waveform.
    """
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    return audio

def transcribe_audio(audio_bytes):
    """
        Transcribes audio from bytes to text using a speech recognition model.

        This function uses a pre-trained automatic speech recognition model to convert
        audio bytes into a text transcription. It selects the appropriate device (CPU or GPU)
        for processing, loads the audio data, and uses the model to generate a transcription.

        Args:
            audio_bytes (bytes): Raw audio data in bytes.

        Returns:
            str: The transcribed text from the audio input.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array.copy(), batch_size=1)["text"]

    return prediction
