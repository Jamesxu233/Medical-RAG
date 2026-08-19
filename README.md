# 🩺 MedicalRAG

A modular Retrieval-Augmented Generation (RAG) framework for biomedical literature retrieval and medical question answering.

The project combines

- PubMed Full-text Retrieval
- MeSH-aware Query Expansion
- Multi-path Dense Retrieval
- Cross-Encoder Re-ranking
- Context Assembly
- Multi-stage Prompt Engineering
- Local LLM Generation (DeepSeek-R1 via Ollama)

to provide evidence-based medical answers from biomedical literature.

---

# Features

✔ Modular architecture

✔ PubMed full-text retrieval

✔ BGE dense embedding

✔ Chroma vector database

✔ MeSH terminology expansion

✔ Medical query understanding

✔ Hybrid retrieval pipeline

✔ Cross-Encoder reranking

✔ Token-aware context assembly

✔ Four-stage prompt engineering

✔ Local DeepSeek-R1 inference via Ollama

✔ JSON structured outputs

---

# Repository Structure

```
MedicalRAG
│
├── notebooks
│   ├── 任务2_字段完整性检查.ipynb
│   ├── 任务3_文档解析与分割.ipynb
│   ├── 任务4_向量化与索引构建.ipynb
│   ├── 任务5_RAG医学查询理解与增强.ipynb
│   ├── 任务6_多路检索.ipynb
│   ├── 任务7 8_生成模块与提示词工程.ipynb
│   ├── 医学词典构建.ipynb
│ 
│
├── src
│   └── medical_rag
│       └── __init__.py
│       ├── chroma_indexer.py
│       ├── config.py
│       ├── mesh_terminology.py
│       ├── query_processor.py
│       ├── retriever.py
│       ├── multipathretriever.py
│       ├── reranker.py
│       ├── context_assembler.py
│       ├── prompts.py
│       ├── llm_generator.py
│       ├── generation_pipeline.py
│       └── generation_utils.py
│       └── multiretrieval_pipeline.py
│       └── query_services.py
│       └── schemas.py
│       └── text_chunker.py
│
├── data
│       └── mesh
├── pyproject.toml
└── README.md
```

---

# System Architecture

```
Medical Query
        │
        ▼
Query Processing
        │
        ▼
MeSH Expansion
        │
        ▼
Dense Retrieval
        │
        ▼
Cross Encoder Reranker
        │
        ▼
Context Assembly
        │
        ▼
Prompt Engineering
        │
        ▼
DeepSeek-R1 (Ollama)
        │
        ▼
Evidence-based Medical Answer
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Jamesxu/Medical-RAG.git

cd MedicalRAG
```

---

## 2. Create Conda Environment

```bash
conda create -n medrag python=3.10

conda activate medrag
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

---

## 4. Install Ollama

Download Ollama

https://ollama.com/download

Verify installation

```bash
ollama --version
```

---

## 5. Download DeepSeek-R1

```bash
ollama pull deepseek-r1:7b
```

Verify

```bash
ollama list
```

---

## 6. Start Ollama

```bash
ollama serve
```

---

# Download Embedding Models

SentenceTransformer models are downloaded automatically.

Default model

```
BAAI/bge-small-en-v1.5
```

Alternative

```
BAAI/bge-base-en-v1.5
```

---

# Dataset

Download PubMed Open Access XML

https://ftp.ncbi.nlm.nih.gov/pub/pmc/

Convert XML to DataFrame

```
PMC XML

↓

Parquet

↓

Chunking

↓

Embedding

↓

Chroma
```

---

# Build Vector Database

```python
from medical_rag.chroma_indexer import PubMedChromaIndexer

indexer = PubMedChromaIndexer(...)

indexer.build_index(...)
```

---

# Load Existing Database

```python
indexer.load_collection()
```

---

# Retrieval Example

```python
results = retriever.search(
    query="Does metformin reduce cardiovascular mortality?"
)
```

---

# Generation Example

```python
answer = pipeline.generate(
    query=query,
    retrieved_docs=results
)
```

---

# Configuration

Modify

```
config.py
```

to customize

- embedding model
- reranker model
- chunk size
- retrieval parameters
- prompt templates

---

# Local Models

Embedding

```
BAAI/bge-small-en-v1.5
```

Reranker

```
BAAI/bge-reranker-base
```

Generator

```
deepseek-r1:7b
```

---

# Pipeline

```
Query

↓

Query Processor

↓

MeSH Expansion

↓

Retriever

↓

Cross Encoder

↓

Context Assembler

↓

Prompt Engineering

↓

LLM Generator

↓

Answer
```

---

# FAQ

### ModuleNotFoundError

Install package

```bash
pip install -e .
```

---

### Ollama returns empty response

Increase

```
max_tokens
```

to

```
1024
```

or higher.

---

### Cannot connect to Ollama

Check

```bash
ollama serve
```

is running.

---
