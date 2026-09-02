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


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return number of tokens for given text and model."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def retrieve_chunks(openai_client: OpenAI, langfuse: Langfuse, collection: chromadb.Collection, query: str,
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
    print(f"Retrieved chunks: {chunks}")

    return chunks, elapsed_ms

def prompt_assembly(langfuse: Langfuse, query:str, retrieved_results: list[Chunk]):
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
            block_tokens_count = count_tokens(context_block)
            if context_tokens + block_tokens_count <= int(os.getenv("MAX_CONTEXT_TOKENS")):
                context_tokens += block_tokens_count
                context_blocks.append(context_block)
            else:
                break

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "Ты ассистент по нормативно-правовым актам Республики Казахстан. Вопросы могут быть заданы на одном из "
            "языков: ru, kz, или en. Отвечай на том же языке, на котором задан вопрос. Если язык вопроса не определен "
            "или не входит в список, отвечай на русском языке. Если ответ не может быть найден в предоставленном "
            "контексте, честно признай это, не изобретай ответ. В ответе должен быть список: названий документов "
            " и статей, на которые ссылается ответ. "
        )

        user_prompt = (
            f"Контекст:\n{context_text}\n\n"
            f"Вопрос пользователя: {query}\n\n"
            "Дай краткий и точный ответ, ссылаясь на полученный контекст."
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


def generate_answer(openai_client: OpenAI, langfuse: Langfuse, prompt: list):
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
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    collection = chroma_client.get_or_create_collection(
        name="structured_chunks" # another option is "overlapping_chunks"
    )

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-assistant",
        input={"query": question},
    ) as root_span:
        with propagate_attributes(metadata={"chunker": "structured", "embeddings": "intfloat/multilingual-e5-large"}):
            retrieved_chunks, retrieval_latency_ms = retrieve_chunks(openai_client, langfuse, collection, question)
            prompt = prompt_assembly(langfuse, question, retrieved_chunks)
            full_response_text = generate_answer(openai_client, langfuse, prompt)

            root_span.update(
                output={
                    "answer": full_response_text,
                    "source_list": [retrieved_chunks[idx].doc_id for idx in range(len(retrieved_chunks))],
                    "retrieval_latency_ms": retrieval_latency_ms,
                }
            )

    langfuse.flush()
    return full_response_text

if __name__ == '__main__':
    if len(sys.argv) > 1:
        question_text = sys.argv[1]
        result = answer(question_text)
        print(result)
