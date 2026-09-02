import os
import re
import json
import random
import tiktoken
from pypdf import PdfReader
import chromadb
import rag.embeddings_utils as embeddings_utils
from dotenv import load_dotenv
from rag.сhunk import Chunk
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


INDEXER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INDEXER_DIR.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return number of tokens for given text and model."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def read_corpus(model: str):
    data_corpus = []
    print(f"Reading corpus from {CORPUS_DIR}")
    files = os.listdir(CORPUS_DIR)
    for filename in files:
        if filename.endswith('.pdf'):
            reader = PdfReader(CORPUS_DIR / filename)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open(CORPUS_DIR / (filename.replace('.pdf', "") + '.meta.json'), "r", encoding="utf-8") as metafile:
                meta = json.load(metafile)

            data_corpus.append({"id": meta["id"], "title": meta["title"], "source_url": meta["source_url"],
                                "language": meta["language"], "version_date": meta["version_date"],
                                "text": text, "tokens_count": count_tokens(text, model)})
    print(f"Read {len(data_corpus)} documents from corpus")

    return data_corpus

def overlapping_chunks(text: str, chunk_size:int, overlap:int, doc_id: str, doc_title: str,
                       source_url: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(Chunk(text[i:i + chunk_size], chunk_id=f"id_{random.randint(100, 200)}", doc_id=doc_id,
                            doc_title=doc_title, source_url=source_url, tokens_count=chunk_size))
    return chunks

def structured_chunks(text: str, doc_id: str, doc_title: str, source_url: str) -> list[Chunk]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        length_function=count_tokens,
        is_separator_regex=False,
    )

    chunks: list[Chunk] = []
    pattern = r'Статья\s+\d+[.]\s+|Параграф\s\d+[.]\s+'
    chunk_texts = re.split(pattern, text)
    for chunk_text in chunk_texts:
        tokens_count = count_tokens(chunk_text)
        # print(f"Chunk length: {tokens_count}")
        article = chunk_text.split("\n")[0] if chunk_text else ""
        if tokens_count > 1200:
            # print(f"Chunk length: {tokens_count} - splitting")
            sub_chunks = text_splitter.split_text(chunk_text)
            for sub_chunk in sub_chunks:
                sub_tokens_count = count_tokens(sub_chunk)
                # print(f"Sub-chunk length: {sub_tokens_count}")
                chunk = Chunk(sub_chunk, chunk_id=f"id_{random.randint(100, 200)}", doc_id=doc_id,
                              doc_title=doc_title, source_url=source_url, tokens_count=sub_tokens_count,
                              article=article)
                chunks.append(chunk)
        elif tokens_count < 800:
            # print(f"Chunk length: {tokens_count} - merging")
            last_chunk = None
            if chunks:
                last_chunk = chunks[-1]
            if last_chunk is not None and last_chunk.tokens_count + tokens_count <= 1200:
                last_chunk.merge(Chunk(chunk_text, chunk_id=f"id_{random.randint(100, 200)}", doc_id=doc_id,
                                       doc_title=doc_title, source_url=source_url, tokens_count=tokens_count,
                                       article=article))
                # print(f"Merged with last chunk. New length: {last_chunk.tokens_count}")
            else:
                chunk = Chunk(chunk_text, chunk_id=f"id_{random.randint(100, 200)}", tokens_count=tokens_count,
                                  article=article, doc_id=doc_id, doc_title=doc_title, source_url=source_url)
                chunks.append(chunk)
        else:
            chunk = Chunk(chunk_text, chunk_id=f"id_{random.randint(100, 200)}", tokens_count=tokens_count,
                          article=article, doc_id=doc_id, doc_title=doc_title, source_url=source_url)
            chunks.append(chunk)

    return chunks

def create_chunks(strategy:str, data_corpus:list, chunk_size:int = 1100, overlap:int = 110):
    if data_corpus is None:
        return list()
    chunks: list[Chunk] = []
    print(f"Creating chunks using strategy: {strategy}")
    for document in data_corpus:
        if strategy == "structured":
            chunks.extend(structured_chunks(document.get("text"), doc_id=document.get("id"),
                                       doc_title=document.get("title"), source_url=document.get("source_url")))
        elif strategy == "overlapping":
            chunks.extend(overlapping_chunks(document.get("text"), chunk_size=chunk_size, overlap=overlap,
                                        doc_id=document.get("id"), doc_title=document.get("title"),
                                       source_url=document.get("source_url")))
    print(f"Created {len(chunks)} chunks")

    return chunks


def create_collection(client_db: chromadb.ClientAPI, name: str, chunks: list[Chunk]):
    print(f"Creating collection: {name} with {len(chunks)} chunks")
    collection = client_db.get_or_create_collection(
        name=name
    )
    embeddings = embeddings_utils.embed_texts([chunk.text for chunk in chunks])
    for chunk,embedding in zip(chunks, embeddings):
        collection.add(
            ids=[chunk.chunk_id],
            documents=[chunk.text],
            metadatas=[{"chunk_id": chunk.chunk_id, "source_url": chunk.source_url, "doc_title": chunk.doc_title, "doc_id": chunk.doc_id,
                        "tokens_count": chunk.tokens_count, "article": chunk.article}],
            embeddings=embedding,
        )
    print(f"Finished creating collection: {name}")

    return collection


def main():
    load_dotenv()

    model = "text-embedding-3-small"
    data_corpus = read_corpus(model)
    chunks_structured = create_chunks("structured", data_corpus)
    chunks_overlapping = create_chunks("overlapping", data_corpus)

    client_db = chromadb.PersistentClient(path="./chroma_data")
    structured_chunks_collection = create_collection(client_db, "structured_chunks",
                                                     chunks_structured)
    overlapping_chunks_collection = create_collection(client_db, "overlapping_chunks",
                                                      chunks_overlapping)




if __name__ == '__main__':
    main()