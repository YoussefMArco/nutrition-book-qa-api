# Health & Nutrition Assistant Using RAG 🌿

Your AI-powered healthcare knowledge companion! This chatbot answers questions about health, nutrition, and wellness using a **Retrieval-Augmented Generation (RAG)** approach. It combines the power of **Gemini AI** and **ChromaDB** to provide accurate, context-aware, and user-friendly responses.

---

## Features ✨

- **Conversational AI**: Understands context and follows up on questions.
- **Reliable Sources**: Answers are based on trusted health and nutrition references.
- **User-Friendly**: Designed for non-technical audiences.
- **Customizable**: Easily adapt the chatbot to new topics or datasets.

---

## Demo 🚀

Try the live demo on Hugging Face Spaces:  
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Saraay/Intelligent_Nutrition_assistant_Using_RAG)

---

## Setup 🛠️

## Requirements

- Python 3.8 or later

#### Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n rag-chroma python=3.8
```
3) Activate the environment:
```bash
$ conda activate mini-rag
```

### (Optional) Setup you command line interface for better readability

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `GEMINI_API_KEY` value.

