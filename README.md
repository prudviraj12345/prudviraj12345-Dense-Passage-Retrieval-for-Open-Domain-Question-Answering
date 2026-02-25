Dense Passage Retrieval for Open-Domain Question Answering
📌 Project Overview

This project implements a Dense Passage Retrieval (DPR) based Open-Domain Question Answering System.
It retrieves the most relevant passages from a given text corpus using dense vector embeddings and answers user questions based on semantic similarity.

The project is divided into:

Backend: Python + Flask + DPR + FAISS

Frontend: React + TypeScript + Vite (deployed on GitHub Pages)

  Problem Statement

Traditional keyword-based search fails to understand the meaning of a question.
This project solves that by using Dense Passage Retrieval, where both questions and passages are converted into dense vector representations, enabling semantic search and better relevance.

  How It Works

Passages are stored in a text file.

Each passage is converted into embeddings using a DPR context encoder.

Embeddings are indexed using FAISS.

User questions are converted into query embeddings.

FAISS retrieves the most similar passage.

The retrieved passage is returned as the answer.
