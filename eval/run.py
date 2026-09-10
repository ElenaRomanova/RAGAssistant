import os
import json
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import get_client, Langfuse
from rag.rag_traced import answer


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR.parent / "eval" / "questions.jsonl"
RESULTS_DIR = BASE_DIR.parent / "results"


def load_questions(path: Path):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs


def load_judge_prompt(langfuse: Langfuse, question, answer):
    try:
        prompt = langfuse.get_prompt(
            "judge",
            label="production",
            cache_ttl_seconds=300  # cache for 5 minutes
        )
        return prompt.compile(
            user_question = question.get("question"),
            assistant_answer = answer.get("answer"),
            sources = answer.get("sources"),
            expected_sources = question.get("expected_sources"),
            expected_answer_points = question.get("expected_answer_points")
        )
    except Exception as e:
        print(f"Langfuse unavailable, using fallback prompt: {e}")
        with open(BASE_DIR.parent / "prompts" / "judge.md", "r", encoding="utf-8") as f:
            default_judge_prompt = f.read()

        return default_judge_prompt.replace(
            "{{user_question}}", question.get("question")
        ).replace("{{assistant_answer}}", answer.get("answer")
        ).replace("{{sources}}", json.dumps(answer.get("sources"))
        ).replace("{{expected_sources}}", json.dumps(question.get("expected_sources"))
        ).replace("{{expected_answer_points}}", json.dumps(question.get("expected_answer_points")))


def llm_judge(openai_client: OpenAI, judge_prompt: str):
    user_msg = (
        "Оцени ответ ассистента. Используй инструкции в системном сообщении. "
    )
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_msg},
    ]

    resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    eval_json = json.loads(resp.choices[0].message.content)
    return eval_json


def run_evaluation():
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is required for OpenAI-only judging.")
        sys.exit(1)
    # ensure SDK picks up key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    openai_client = OpenAI()
    langfuse = get_client()

    questions = load_questions(QUESTIONS_PATH)

    evaluation_results = []
    for question in questions:
        qid = question.get("id")
        qtext = question.get("question")
        with langfuse.start_as_current_observation(as_type="span", name="eval_query", input={"query": qtext}):
            ans_json = answer(qtext)
            judge_prompt = load_judge_prompt(langfuse, question, ans_json)
            model_answer_text = ans_json.get("answer")
            refused = (ans_json.get("in_scope") == "false")
            expected_refusal = (question.get("type") == "out_of_scope")
            print(f"Refused: {refused}, Expected refusal: {expected_refusal}")
            refusal_correctness = not (refused ^ expected_refusal)
            language_match = (ans_json.get("language") == question.get("language"))
            judge_result = llm_judge(openai_client, judge_prompt) # score,  context_recall, answer_correctness
            score = judge_result.get("score")
            context_recall = judge_result.get("context_recall")
            answer_correctness = judge_result.get("answer_correctness")
            evaluation_results.append({
                "strategy": "structured", #another option is "overlapping"
                "id": qid,
                "question": qtext,
                "context_recall": context_recall,
                "answer_score": score,
                "answer_correctness": answer_correctness,
                "model_answer": model_answer_text,
                "refusal_correctness": refusal_correctness,
                "language_match": language_match,
            })


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"eval_{ts}.json"
    md_path = RESULTS_DIR / f"eval_{ts}.md"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(evaluation_results, jf, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(f"# Evaluation results {ts}\n\n")
        for result in evaluation_results:
            mf.write(
                f" Strategy: {result.get("strategy")}\n"
                f" Question id: {result.get("id")}\n"
                f" Language match: {result.get("language_match")}\n"
                f" Refusal correctness: {result.get("refusal_correctness")}\n"
                f" Answer score: {result.get("answer_score")}\n"
                f" Context recall: {result.get("context_recall")}\n"
                f" Answer correctness: {result.get("answer_correctness")}\n"
                f" Question: {result.get("question")}\n"
                f" Model answer: {result.get("model_answer")}\n"
                f"\n---\n\n"
            )

    print(f"Saved results to {json_path} and {md_path}.")


if __name__ == '__main__':
    run_evaluation()
