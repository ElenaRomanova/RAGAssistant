import json
from pathlib import Path
import tiktoken
import os
import sys
from time import perf_counter
from langfuse import get_client, propagate_attributes, Langfuse
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import rag.embeddings_utils as embeddings_utils
from rag.сhunk import Chunk

INDEXER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INDEXER_DIR.parent


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return number of tokens for given text and model."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def _retrieve_chunks(langfuse: Langfuse, collection: chromadb.Collection, query: str,
                    model: str = "text-embedding-3-small"):
    print(f"Retrieving chunks for query: {query}")
    with langfuse.start_as_current_observation(
        as_type="span",
        name="retrieval",
        input={
            "query": query,
        },
    ) as retrieval_span:
        started_at = perf_counter()
        query_embedding = embeddings_utils.embed_query(query)
        db_query_results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        chunks = []
        for i in range(0, len(db_query_results["documents"][0])):
            chunks.append(Chunk(text = db_query_results["documents"][0][i],
                                chunk_id = db_query_results["metadatas"][0][i].get("chunk_id"),
                                doc_id = db_query_results["metadatas"][0][i].get("doc_id"),
                                doc_title = db_query_results["metadatas"][0][i].get("doc_title"),
                                source_url = db_query_results["metadatas"][0][i]["source_url"],
                                tokens_count = db_query_results["metadatas"][0][i].get("tokens_count"),
                                article = db_query_results["metadatas"][0][i].get("article"))
                          )

        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

        retrieval_span.update(
            output={
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
            metadata={"retrieval_latency_ms": elapsed_ms},
        )
    print(f"Retrieved {len(chunks)} chunks for query: {query} in {elapsed_ms} ms")
    print(f"Retrieved documents: {[chunk.doc_id for chunk in chunks]}")

    return chunks, elapsed_ms


def load_system_prompt(langfuse: Langfuse):
    try:
        prompt = langfuse.get_prompt(
            "system",
            label="production",
            cache_ttl_seconds=300  # cache for 5 minutes
        ).compile()
        print(f"Loaded system prompt from Langfuse: {prompt}")
        return prompt
    except Exception as e:
        print(f"Langfuse unavailable, using fallback prompt: {e}")
        with open(PROJECT_ROOT / "prompts" / "system.md", "r", encoding="utf-8") as f:
            default_system_prompt = f.read()

        return default_system_prompt


def _prompt_assembly(langfuse: Langfuse, query:str, retrieved_results: list[Chunk]):
    print(f"Assembling prompt for query: {query} with {len(retrieved_results)} retrieved chunks")
    context_blocks = []
    context_tokens = 0

    with langfuse.start_as_current_observation(
        as_type="span",
        name="prompt_assembly",
        input={
            "query": query,
        },
    ) as prompt_span:
        for idx in range(0, len(retrieved_results)):
            context_block = (f"Фрагмент {idx}]\n "
                             f"Ссылка на источник: {retrieved_results[idx].source_url}\n"
                             f"Название документа: {retrieved_results[idx].doc_title}\n"
                             f"ID документа: {retrieved_results[idx].doc_id}\n"
                             f"Статья: {retrieved_results[idx].article}\n"
                             f"Индекс фрагмента: {retrieved_results[idx].chunk_id}\n"
                             f"Текст: {retrieved_results[idx].text}")
            block_tokens_count = _count_tokens(context_block)
            if context_tokens + block_tokens_count <= int(os.getenv("MAX_CONTEXT_TOKENS")):
                context_tokens += block_tokens_count
                context_blocks.append(context_block)
            else:
                break
        print(f"Context assembled with {len(context_blocks)} blocks and {context_tokens} tokens")

        context_text = "\n\n".join(context_blocks)

        system_prompt = load_system_prompt(langfuse)

        user_prompt = (
            f"Контекст:\n{context_text}\n\n"
            f"Вопрос пользователя: {query}\n\n"
        )

        messages =  [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt_span.update(
            output={
                "message_count": len(messages),
                "system_prompt": messages[0]["content"],
                "user_prompt_preview": messages[1]["content"][:800],
            }
        )
        print(f"Prompt assembled for query: {query} with {len(messages)} messages")

        return messages


def _generate_answer(openai_client: OpenAI, langfuse: Langfuse, prompt: list):
    print(f"Generating answer for prompt with {len(prompt)} messages")
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="generation",
        model="gpt-4o-mini",
        input=prompt,
        model_parameters={"temperature": 0.2},
    ) as generation:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.2
        )
        answer =  response.choices[0].message.content
        generation.update(
            output=answer,
            usage_details={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )
        print(f"Answer generated")

        return answer

def answer(question: str):
    load_dotenv()
    openai_client = OpenAI()
    langfuse = get_client()
    chroma_client = chromadb.PersistentClient(path=PROJECT_ROOT / "./chroma_data")
    collection = chroma_client.get_or_create_collection(
        name="structured_chunks" # another option is "overlapping_chunks"
    )
    strategy = "structured" # another option is "overlapping"

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-assistant",
        input={"query": question},
    ) as root_span:
        with propagate_attributes(metadata={"chunking_strategy": strategy,
                                            "embedding_model": "intfloat/multilingual-e5-large",
                                            "llm_model": "gpt-4o-mini",
                                            "max_context_tokens": int(os.getenv("MAX_CONTEXT_TOKENS")),
                                            "top_k": 3,
                                            "system_prompt_version": "1.0",
                                            "judge_prompt_version": "1.0"
                                            }):
            retrieved_chunks, retrieval_latency_ms = _retrieve_chunks(langfuse, collection, question)
            prompt = _prompt_assembly(langfuse, question, retrieved_chunks)
            full_response_text = _generate_answer(openai_client, langfuse, prompt)
            if full_response_text:
                result_json= json.loads(full_response_text)

            root_span.update(
                output={
                    "answer": result_json.get("answer"),
                    "language": result_json.get("language"),
                    "source_list": [retrieved_chunks[idx].doc_id for idx in range(len(retrieved_chunks))],
                    "retrieval_latency_ms": retrieval_latency_ms,
                }
            )

    langfuse.flush()
    return result_json

if __name__ == '__main__':
    if len(sys.argv) > 1:
        question_text = sys.argv[1]
        result = answer(question_text)
        print(result)
