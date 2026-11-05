import os

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download a YouTube video's audio, convert it to .wav, return the file path."""
    # yt-dlp fills in %(title)s and %(ext)s itself
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",  # needs ffmpeg installed
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original_path = ydl.prepare_filename(info)  # e.g. downloads/Title.webm

    # swap whatever extension it had for .wav (that's what ffmpeg produced)
    return os.path.splitext(original_path)[0] + ".wav"


def convert_to_wav(input_path: str) -> str:
    """Convert any uploaded audio/video file (mp3, m4a, mp4...) to 16kHz mono .wav.

    16kHz mono is what Whisper uses internally, so this keeps things fast.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # "_converted" so we never overwrite the input if it's already a .wav
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split a long .wav into N-minute pieces, save each one, return their paths."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000  # pydub measures time in milliseconds

    base = os.path.splitext(wav_path)[0]
    chunks = []

    # range(0, total_length, chunk_ms) -> 0, 10min, 20min, ... (in ms)
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        piece = audio[start:start + chunk_ms]  # slicing works like a list
        chunk_path = f"{base}_chunk_{i}.wav"
        piece.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """Main entry point: takes a YouTube URL OR a local file path,
    returns a list of .wav chunk paths ready for Whisper."""
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)  # already a .wav
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
 
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks