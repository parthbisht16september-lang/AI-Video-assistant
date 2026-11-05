# 🎥 AI Video Assistant (Multilingual RAG & Transcription)

An end-to-end AI-powered video ingestion, transcription, and interactive Question-Answering (QA) application. The tool processes YouTube videos in **English**, **Hindi**, or **Hinglish**, transcribes audio using dedicated speech models, chunks and indexes the content into a vector database, and enables conversational retrieval-augmented generation (RAG) for summaries and key insight extraction.

---

## 📸 Overview Architecture

-----------------------+
                                |   YouTube Video URL   |
                                +-----------+-----------+
                                            |
                                  [ Audio Extraction ]
                                            |
                                 +----------+----------+
                                 |                     |
                          (English Track)     (Hindi / Hinglish Track)
                                 |                     |
                           [ Whisper API ]      [ Sarvam AI API ]
                                 |                     |
                                 +----------+----------+
                                            |
                                    [ Raw Transcript ]
                                            |
                              [ Recursive Text Splitter ]
                                            |
                                 [ Vector Store Index ]
                                            |
                                 +----------+----------+
                                 |                     |
                       [ Dynamic Summary ]   [ Interactive RAG QA ]



                       --

## ✨ Features

- 🎧 **Multilingual Audio Processing:** Automatically transcribes YouTube videos across multiple languages:
  - **Whisper API** for precise English transcriptions.
  - **Sarvam AI API** for high-accuracy Hindi and mixed Hinglish audio streams.
- 🧩 **Smart Text Chunking & Indexing:** Implements token-aware text splitting to segment long transcripts into optimal vector payloads.
- ⚡ **RAG-Powered Conversational QA:** Queries are answered contextually using vector similarity searches backed by an LLM pipeline.
- 📝 **Automated Executive Summarization:** Generates structured, high-level key takeaways and topic breakdowns directly from the transcribed data.

---

## 🛠️ Tech Stack

- **Primary Language:** Python 3.10+
- **LLM & Frameworks:** LangChain, OpenAI / Gemini / Llama models
- **ASR & Speech Models:** OpenAI Whisper, Sarvam AI API
- **Vector Database:** ChromaDB / FAISS / Qdrant
- **Data Ingestion & Utils:** `yt-dlp`, PyPDF / LangChain Document Loaders
- **Frontend / UI:** Streamlit (Optional / Included)

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and install required dependencies:

```bash
git clone [https://github.com/parthbisht16september-lang/AI-Video-assistant.git](https://github.com/parthbisht16september-lang/AI-Video-assistant.git)
cd AI-Video-assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
