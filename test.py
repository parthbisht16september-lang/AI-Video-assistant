"""English video workflow and terminal Q&A.

Place this file in Video_agent/, beside .env, core/, and utils/.
Run: python testboard.py
Reuse saved text: python testboard.py --transcript test_outputs/run_XXX/transcript.txt

Expected RAG module: core/rag.py (the code defining build_rag_chain).
Checks prove that stages run and return text, not that their answers are accurate.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
VIDEO_URL = "https://www.youtube.com/watch?v=6i_T5kkfv4A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path,
                        help="Reuse a saved transcript instead of downloading/transcribing.")
    args = parser.parse_args()
    transcript_path = args.transcript.resolve() if args.transcript else None
    os.chdir(ROOT)
    output_root = ROOT / "test_outputs"
    output_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run_", dir=output_root))
    report = {"video_url": VIDEO_URL, "stages": {}, "chat": []}
    stage = "Setup"

    def save_report():
        (output / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def run_stage(name, function, *values, filename=None, **kwargs):
        nonlocal stage
        stage = name
        print(f"\n>>> {name}", flush=True)
        started = time.perf_counter()
        result = function(*values, **kwargs)
        if filename:
            if not isinstance(result, str) or not result.strip():
                raise ValueError(f"{name} returned empty text.")
            (output / filename).write_text(result, encoding="utf-8")
            print(result[:1500])
            if len(result) > 1500:
                print(f"[Full output saved in {filename}]")
        report["stages"][name] = {
            "status": "PASS", "seconds": round(time.perf_counter() - started, 2)
        }
        save_report()
        print(f"PASS: {name}", flush=True)
        return result

    try:
        print(f"Video: {VIDEO_URL}\nResults: {output}", flush=True)
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.getenv("GROQ_API_KEY", "").strip():
            raise RuntimeError("Set GROQ_API_KEY in the project-root .env file.")
        if not os.getenv("GROQ_MODEL", "").strip():
            os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
        print(f"Groq model: {os.environ['GROQ_MODEL']}")

        from core.summarize import summarize, generate_title
        from core.extractor import (
            extract_action_items, extract_key_decisions, extract_questions,
        )
        from core.rag_engine import build_rag_chain

        report["stages"]["Setup"] = {"status": "PASS"}

        if transcript_path:
            report["transcript_source"] = str(transcript_path)
            transcript = run_stage(
                "Load saved transcript", transcript_path.read_text,
                encoding="utf-8", filename="transcript.txt",
            )
        else:
            stage = "Audio setup"
            if not shutil.which("ffmpeg"):
                raise RuntimeError("FFmpeg not found. On Mac, run: brew install ffmpeg")
            from utils.audio_processing import process_input
            from core.transcriber import transcribe_all

            def prepare_audio():
                chunks = process_input(VIDEO_URL)
                if chunks is None or isinstance(chunks, (str, bytes, os.PathLike)):
                    raise TypeError("process_input() must return a list of audio paths.")
                chunks = list(chunks)
                if not chunks:
                    raise ValueError("No audio chunks were produced.")
                for chunk in chunks:
                    path = Path(chunk)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"Missing or empty audio chunk: {path}")
                print(f"Prepared {len(chunks)} chunks.")
                return chunks

            chunks = run_stage("Download and prepare audio", prepare_audio)
            transcript = run_stage(
                "English transcription (Whisper)", transcribe_all, chunks,
                language="english", filename="transcript.txt",
            )

        run_stage("Summary", summarize, transcript, filename="summary.txt")
        run_stage("Title", generate_title, transcript, filename="title.txt")
        run_stage("Action items", extract_action_items, transcript,
                  filename="action_items.txt")
        run_stage("Decisions", extract_key_decisions, transcript,
                  filename="decisions.txt")
        run_stage("Open questions", extract_questions, transcript,
                  filename="questions.txt")
        rag_chain = run_stage("Build vector store and RAG chain", build_rag_chain, transcript)

        stage = "Transcript Q&A"
        print("\nChat is ready. Ask complete questions about the transcript.")
        print("Each question is independent; this chain has no conversation memory.")
        print("Type exit or quit to finish. Answers are saved to chat.json.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nChat ended.")
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            try:
                answer = rag_chain.invoke(question)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("The RAG chain returned an empty answer.")
                print(f"\nAssistant: {answer}\n")
                report["chat"].append({"question": question, "answer": answer})
            except Exception as error:
                print(f"Question failed: {type(error).__name__}: {error}")
                print("You can try another question or type exit.\n")
                report["chat"].append({"question": question, "error": str(error)})
            (output / "chat.json").write_text(
                json.dumps(report["chat"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            save_report()

        answers = sum("answer" in turn for turn in report["chat"])
        errors = sum("error" in turn for turn in report["chat"])
        report["stages"][stage] = {
            "status": "FAIL" if errors else ("PASS" if answers else "NOT TESTED"),
            "answered": answers, "failed": errors,
        }
        report["status"] = "COMPLETED WITH CHAT ERRORS" if errors else "COMPLETED"
        print(f"\nWorkflow completed. Q&A: {answers} answered, {errors} failed.")
        if not report["chat"]:
            print("No questions were submitted, so Q&A was not tested.")
        print("Review the saved transcript and answers for factual accuracy.")
        return 1 if errors else 0

    except Exception as error:
        report["status"] = "FAIL"
        report["stages"][stage] = {"status": "FAIL", "error": str(error)}
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"\nFAIL at {stage}: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"Full traceback: {output / 'error.txt'}", file=sys.stderr)
        return 1
    finally:
        save_report()
        print(f"\nResults saved in: {output}")


if __name__ == "__main__":
    sys.exit(main())
