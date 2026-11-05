import io
import os

import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()  # reads your .env so os.getenv can see the keys

# ---------- Whisper (local, free) ----------
# tiny / base / small / medium / large  (bigger = more accurate but slower)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# ---------- Sarvam AI (cloud, great for Hindi / Hinglish) ----------
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")

# Sarvam's REST API rejects audio longer than 30 seconds,
# so we cut each chunk into 25s pieces before sending.
SARVAM_MAX_SECONDS = 25

_model = None  # cache: Whisper is loaded once, then reused


def load_model():
    """Load the Whisper model the first time, then reuse it."""
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe one chunk locally with Whisper (best for English audio)."""
    model = load_model()

    # fp16=False avoids a warning when running on CPU
    result = model.transcribe(chunk_path, task="transcribe", fp16=False)
    return result["text"].strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Split audio into pieces of up to 25 seconds and transcribe in order."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = 25 * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    texts = []

    for index, start in enumerate(range(0, len(audio), piece_ms), start=1):
        # Slicing automatically keeps the final shorter piece.
        piece = audio[start:start + piece_ms]

        print(
            f"  Sarvam piece {index}/{total_pieces} "
            f"({len(piece) / 1000:.2f} seconds)..."
        )

        with io.BytesIO() as buffer:
            piece.export(buffer, format="wav")
            buffer.seek(0)

            response = requests.post(
                SARVAM_STT_TRANSLATE_URL,
                headers=headers,
                files={
                    "file": (
                        f"piece_{index}.wav",
                        buffer,
                        "audio/wav",
                    )
                },
                data={"model": SARVAM_MODEL},
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Sarvam error on piece {index}/{total_pieces}: "
                f"{response.status_code} — {response.text}"
            )

        text = response.json().get("transcript", "").strip()
        if text:
            texts.append(text)

    return " ".join(texts)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Router: picks the right engine based on the language."""
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe every chunk in order and join them into one transcript."""
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()


# Quick test: python -m core.transcriber   (run from the project root)
if __name__ == "__main__":
    from utils.audio_processor import process_input

    chunks = process_input("https://www.youtube.com/watch?v=7HSSR1n8dgc")
    print(transcribe_all(chunks, language="english"))