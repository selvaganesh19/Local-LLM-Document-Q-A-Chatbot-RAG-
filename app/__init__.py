"""Local LLM Document Q&A Chatbot (RAG).

A fully local Retrieval-Augmented Generation service: hybrid (BM25 + dense)
retrieval over a ChromaDB HNSW index, cross-encoder reranking, and grounded
answer generation through a locally hosted Ollama model.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
