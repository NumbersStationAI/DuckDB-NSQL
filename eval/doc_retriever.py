"""Retrieve documentation for a given query."""

from pathlib import Path
from typing import Any
from rich.console import Console
from tqdm import tqdm
import numpy as np
from manifest import Manifest
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

console = Console(soft_wrap=True)

try:
    EMBEDDING_MODEL = Manifest(
        client_name="openaiembedding",
        cache_name="sqlite",
        cache_connection=".manifest.sqlite",
    )
except Exception as e:
    console.print(e)
    console.print(
        "Failed to load embedding model. Likely OPENAI API key is not set. Please set to run document retrieval.",
        style="bold red",
    )


def load_documentation(path: Path) -> dict[str, str]:
    """Load documentation from path."""
    content = {}
    for file in path.glob("**/*.md"):
        with open(file, "r") as f:
            data = f.read()
            key = str(file).replace(str(path), "")
            content[key] = data
    return content


def split_documents(content: dict[str, str]) -> dict[str, Any]:
    """Split documents into chunks."""
    md_splitted_docs = []
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=500, chunk_overlap=50, length_function=len
    )

    for file, raw_doc in content.items():
        splitted_text = markdown_splitter.split_text(raw_doc)
        for t in splitted_text:
            t.metadata["source"] = file
        md_splitted_docs.extend(splitted_text)

    docs = text_splitter.split_documents(md_splitted_docs)
    docs_as_dict = [doc.dict() for doc in docs]
    return docs_as_dict


def get_embeddings(text: str) -> np.ndarray:
    """Get embeddings."""
    return np.array(EMBEDDING_MODEL.run(text))


def embed_documents(
    chunked_docs: dict[str, Any], key: str = "page_content"
) -> tuple[dict[str, Any], np.ndarray]:
    """Embed documents."""
    all_embeddings = []
    for doc in tqdm(chunked_docs):
        emb = get_embeddings(doc[key])
        doc["embedding"] = emb
        all_embeddings.append(doc["embedding"])
    full_embedding_mat = np.vstack(all_embeddings)
    return chunked_docs, full_embedding_mat


def query_docs(
    query: str,
    docs: dict[str, Any],
    embedding_mat: np.ndarray,
    top_n: int = 10,
    key: str = "page_content",
) -> tuple[list[int], list[str]]:
    """Query documents."""
    query_embedding = get_embeddings(query)
    scores = embedding_mat.dot(query_embedding)
    sorted_indices = np.argsort(scores)[::-1]
    top_n_indices = sorted_indices[:top_n]
    top_n_indices_rev = top_n_indices[::-1]
    returned_docs = []
    for i in top_n_indices_rev:
        returned_docs.append(docs[i][key])
    return top_n_indices_rev.tolist(), returned_docs
