# AFP Major Daniel — AI Field Officer
### Offline AI Chat App for the Armed Forces of the Philippines

---

## Project Overview

Major Daniel is a fully **offline** AI assistant for AFP field personnel. When soldiers cannot reach their actual commanding officer, they can report their situation to Major Daniel and receive clear, step-by-step orders — covering vessel sightings, suspicious individuals, medical emergencies, disaster response, report drafting, and more.

**All data stays on the device. No internet required.**

---

## Project Structure

```
afp_ai_chat/
├── main.py                      # PyQt6 chat application (GUI)
├── rag.py                       # Offline RAG engine (ChromaDB + sentence-transformers)
├── Modelfile                    # Ollama custom model definition for Major Daniel
├── major_daniel_training.jsonl  # SFT training dataset (30 AFP scenarios)
├── docs/                        # AFP knowledge documents (auto-created on first run)
│   ├── wps_protocols.txt
│   ├── rules_of_engagement.txt
│   ├── field_medical.txt
│   ├── incident_report_formats.txt
│   └── philippine_military_law.txt
├── chroma_db/                   # Local vector database (auto-created after --ingest)
└── README.md
```

---

## Full Setup Guide

### Step 1 — Install Python dependencies
```bash
pip install PyQt6 ollama chromadb sentence-transformers
```

### Step 2 — Install Ollama
Download from https://ollama.com, then pull the base model:
```bash
ollama pull deepseek-r1:8b
```

### Step 3 — Build the Major Daniel custom model
```bash
ollama create major-daniel -f Modelfile
ollama list   # verify it appears
```

### Step 4 — Build the RAG knowledge base
```bash
python rag.py --ingest
```
Sample documents are auto-created in docs/ on first run.
Add your own .txt or .md AFP documents to docs/ and re-run --ingest.

### Step 5 — Run the app
```bash
python main.py
```

---

## RAG CLI Usage
```bash
python rag.py --query "I spotted a red vessel near our coast"
python rag.py --interactive
python rag.py --list-docs
python rag.py --ingest
```

---

## Tech Stack

| Component | Technology |
|---|---|
| GUI | PyQt6 |
| LLM | DeepSeek-R1 8B via Ollama |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (local, no server) |
| Language | Python 3.11+ |
| Internet required | None |
