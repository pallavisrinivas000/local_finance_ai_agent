# local_finance_ai_agent

A local AI agent that answers questions about your monthly finance documents using vector search and embeddings.

## Overview

This project processes your bank transaction CSV (e.g., `aug_expenses.csv`), converts each transaction into a vector embedding, and stores them in a vector database (ChromaDB). You can then query your financial data using natural language, and the agent retrieves the most relevant transactions.

## How It Works

1. **Data Loading:** Reads your bank statement CSV using `pandas`.
2. **Embedding:** Converts each transaction into a vector using `OllamaEmbeddings` from `langchain-ollama`.
3. **Vector Database:** Stores the embeddings and metadata in a local ChromaDB instance using `langchain-chroma`.
4. **Retrieval:** Uses vector similarity search to find and return the most relevant transactions for your queries.

## Libraries Used

- **[pandas](https://pandas.pydata.org/):**  
  For reading and processing CSV files containing your transaction data.

- **[langchain](https://python.langchain.com/):**  
  Provides the framework for chaining together language model and vector database operations.

- **[langchain-ollama](https://github.com/langchain-ai/langchain-ollama):**  
  Supplies the `OllamaEmbeddings` class, which generates vector representations (embeddings) of your transaction text using local models.

- **[langchain-chroma](https://github.com/langchain-ai/langchain-chroma):**  
  Integrates ChromaDB with LangChain, allowing you to store and retrieve document embeddings efficiently.

## What is a Vector Database?

A vector database stores data as high-dimensional vectors (embeddings) instead of traditional rows and columns. This allows for efficient similarity search—finding items that are "close" in meaning or context. In this project, each transaction is embedded as a vector, enabling semantic search over your financial records.

### Why ChromaDB?

[ChromaDB](https://docs.trychroma.com/) is an open-source vector database designed for AI and LLM applications. It is used here to persist and search your transaction embeddings locally, ensuring privacy and fast retrieval.

## Project Structure
