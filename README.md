# DocuMind-RAG-PySpark

Scalable Document Intelligence using RAG, HuggingFace, FAISS, and PySpark.

## Problem Statement

Organizations deal with large volumes of unstructured documents. Traditional keyword search fails to capture semantic meaning.  
This project builds a scalable Retrieval-Augmented Generation (RAG) system to enable intelligent document querying using embeddings and LLMs.

## Architecture

Document → Chunking → Embedding (Hugging Face) → FAISS Index  
User Query → Embedding → FAISS Retrieval → LLM → Response

## Tech Stack

- PySpark (distributed processing)
- Hugging Face (embeddings)
- FAISS (vector search)
- FastAPI (API layer)
- Python

## Features

- Scalable document ingestion using PySpark
- Semantic search using vector embeddings
- Fast similarity search with FAISS
- Retrieval-Augmented Generation pipeline

## Setup & Run

```bash
git clone <your-repo-url>
cd DocuMind-RAG-PySpark
pip install -r requirements.txt
python src/main.py
