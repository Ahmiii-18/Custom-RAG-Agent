# Multimodal AI RAG Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://custom-rag-agent.streamlit.app/)
[![Watch Demo](https://img.shields.io/badge/🎥_Watch_Video_Demo-Google_Drive-red.svg)](https://drive.google.com/file/d/16woJIA1t3pKcKTVUSZM3eJ5w2rfbA3xx/view?usp=sharing)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced retrieval-augmented generation (RAG) agent capable of indexing and querying cross-modal assets—including text documents, PDFs, high-resolution images, audio files, and video recordings—with strict factual grounding and source-level citation tracking.

🌐 **Live Demo:** [custom-rag-agent.streamlit.app](https://custom-rag-agent.streamlit.app/)  
🎥 **Video Walkthrough:** [Google Drive Demo](https://drive.google.com/file/d/16woJIA1t3pKcKTVUSZM3eJ5w2rfbA3xx/view?usp=sharing)

---

## Key Features

* **Multimodal Data Ingestion**: Unified indexing for PDF, TXT, PNG, JPG, MP3, WAV, and MP4 formats.
* **Automated Media Contextualization**: Uses OpenAI Vision for detailed image analysis and Whisper API for automatic speech-to-text transcriptions of audio and video inputs.
* **Vector Search Indexing**: Employs `FAISS` and OpenAI `text-embedding-3-small` embeddings for localized similarity retrieval.
* **Source-Grounded Generation**: Strict prompt guardrails enforce response verification against context with file-level and page/timestamp citations.
* **Interactive Context Inspection**: Streamlit drawer interface lets users inspect retrieved vector chunks directly for auditing and provenance verification.

---

## System Architecture

1. **Asset Parsing**: Text documents and PDFs are split using `RecursiveCharacterTextSplitter`. Visual assets undergo Vision analysis, and media assets are transcribed via Whisper.
2. **Vector Indexing**: Document chunks and media transcriptions are embedded and stored in an in-memory `FAISS` vector database.
3. **Contextual Retrieval**: User queries pull the top-$k$ relevant chunks across text and transcribed media embeddings.
4. **Grounded Answering**: `gpt-4o` / `gpt-4o-mini` synthesizes contextual answers with citations pointing directly to source files.

---

## Installation & Setup

### Prerequisites

* Python 3.10+
* OpenAI API Key

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/Ahmiii-18/custom-rag-agent.git](https://github.com/Ahmiii-18/custom-rag-agent.git)
   cd custom-rag-agent