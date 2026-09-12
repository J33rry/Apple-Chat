import json
import re
import time
from collections import defaultdict
from tqdm import tqdm
from agent import SupportAgent, get_best_provider, LocalProvider, VALID_INTENTS


def load_golden_dataset(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class SimpleAgent(SupportAgent):
    """Baseline agent with no RAG — relies purely on LLM pre-trained knowledge."""
    def retrieve_context(self, tweet_text: str) -> str:
        return ""  # No RAG


def compute_rouge_scores(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 4),
            "rouge2": round(scores['rouge2'].fmeasure, 4),
            "rougeL": round(scores['rougeL'].fmeasure, 4),
        }
    except ImportError:
        print("Warning: rouge-score not installed. Run: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def judge_reply(judge_provider, customer_tweet: str, drafted_reply: str, brand_tweet: str, max_retries: int = 2) -> int:
    """Use LLM-as-judge with retry and validation. Returns score 1-5 or 0 on failure."""
    if not drafted_reply or not drafted_reply.strip():
        return 0

    judge_prompt = f"""You are an expert customer service evaluator.
Customer Tweet: "{customer_tweet}"
Agent Drafted Reply: "{drafted_reply}"
Actual Brand Historical Reply: "{brand_tweet}"

Rate the drafted reply on a scale of 1 to 5 based on helpfulness, relevance, and tone.
1 = Completely unhelpful or irrelevant
2 = Somewhat relevant but missing key information
3 = Adequate response with room for improvement
4 = Good, helpful response
5 = Excellent, matches or exceeds brand quality

Respond ONLY with a single integer digit (1, 2, 3, 4, or 5). Do not write anything else.
"""
    for attempt in range(max_retries + 1):
        try:
            score_str = judge_provider.generate(judge_prompt).strip()
            match = re.search(r'\b([1-5])\b', score_str)
            if match:
                return int(match.group(1))
            if attempt < max_retries:
                time.sleep(1)
        except Exception:
            if attempt < max_retries:
                time.sleep(1)
    return 0


def run_evaluation():
    golden_threads = load_golden_dataset("data/golden_dataset_labeled.json")

    # Use best available provider for agents
    agent_provider = get_best_provider()
    provider_name = type(agent_provider).__name__
    print(f"Using {provider_name} for agent inference.")

    # Use best available provider for judging (prefer Gemini for quality)
    judge_provider = get_best_provider()
    judge_name = type(judge_provider).__name__
    print(f"Using {judge_name} for LLM-as-judge.")

    # RAG Agent
    rag_agent = SupportAgent(agent_provider, "data/knowledge_base.json")
    # Simple Agent (no RAG)
    simple_agent = SimpleAgent(agent_provider, "data/knowledge_base.json")

    results = []
    # Accumulators for aggregate metrics
    intent_correct = {"rag": 0, "simple": 0}
    escalation_correct = {"rag": 0, "simple": 0}
    intent_total = 0

    for i, data in enumerate(tqdm(golden_threads, desc="Evaluating")):
        customer_tweet = data["customer_tweet"]
        brand_tweet = data["actual_reply"]
        true_intent = data.get("true_intent", "general_inquiry")
        true_escalate = data.get("true_escalate", True)

        # 1. Trivial Baseline
        trivial_output = {
            "intent": "general_inquiry",
            "drafted_reply": "Please DM us so we can look into this.",
            "escalate": True,
            "escalation_reason": "Trivial baseline always escalates"
        }

        # 2. Simple Baseline (No RAG)
        simple_output = simple_agent.handle_tweet(customer_tweet)

        # 3. RAG Agent
        rag_output = rag_agent.handle_tweet(customer_tweet)

        # Judge scores
        rag_judge = judge_reply(judge_provider, customer_tweet, rag_output['drafted_reply'], brand_tweet)
        simple_judge = judge_reply(judge_provider, customer_tweet, simple_output['drafted_reply'], brand_tweet)
        trivial_judge = judge_reply(judge_provider, customer_tweet, trivial_output['drafted_reply'], brand_tweet)

        # ROUGE scores
        rag_rouge = compute_rouge_scores(rag_output['drafted_reply'], brand_tweet)
        simple_rouge = compute_rouge_scores(simple_output['drafted_reply'], brand_tweet)
        trivial_rouge = compute_rouge_scores(trivial_output['drafted_reply'], brand_tweet)

        # Intent accuracy
        if rag_output['intent'] == true_intent:
            intent_correct['rag'] += 1
        if simple_output['intent'] == true_intent:
            intent_correct['simple'] += 1

        # Escalation accuracy
        if rag_output['escalate'] == true_escalate:
            escalation_correct['rag'] += 1
        if simple_output['escalate'] == true_escalate:
            escalation_correct['simple'] += 1
        intent_total += 1

        results.append({
            "tweet": customer_tweet,
            "actual_reply": brand_tweet,
            "true_intent": true_intent,
            "true_escalate": true_escalate,
            "rag": {"output": rag_output, "judge_score": rag_judge, "rouge": rag_rouge},
            "simple": {"output": simple_output, "judge_score": simple_judge, "rouge": simple_rouge},
            "trivial": {"output": trivial_output, "judge_score": trivial_judge, "rouge": trivial_rouge},
        })

        # Rate limiting for API providers
        if provider_name == "GeminiProvider":
            time.sleep(1.5)

    # ── Aggregate Metrics ──
    n = len(results)
    print(f"\n{'='*60}")
    print(f"Evaluation Complete — {n} queries")
    print(f"Provider: {provider_name} | Judge: {judge_name}")
    print(f"{'='*60}")

    def calc_metrics(key):
        valid_scores = [r[key]['judge_score'] for r in results if r[key]['judge_score'] > 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        esc_rate = sum(1 for r in results if r[key]['output']['escalate']) / n if n else 0
        avg_rouge1 = sum(r[key]['rouge']['rouge1'] for r in results) / n if n else 0
        avg_rouge2 = sum(r[key]['rouge']['rouge2'] for r in results) / n if n else 0
        avg_rougeL = sum(r[key]['rouge']['rougeL'] for r in results) / n if n else 0
        valid_count = len(valid_scores)
        return avg_score, esc_rate, avg_rouge1, avg_rouge2, avg_rougeL, valid_count

    header = f"{'Agent':<20} {'Judge Avg':>10} {'Valid':>6} {'Esc%':>6} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}"
    print(header)
    print("-" * len(header))

    for key, label in [('rag', 'RAG Agent'), ('simple', 'Simple Baseline'), ('trivial', 'Trivial Baseline')]:
        avg, esc, r1, r2, rL, valid = calc_metrics(key)
        print(f"{label:<20} {avg:>10.2f} {valid:>6}/{n} {esc*100:>5.1f}% {r1:>8.4f} {r2:>8.4f} {rL:>8.4f}")

    # Intent & Escalation accuracy (only for RAG and Simple)
    print(f"\n{'='*60}")
    print("Intent & Escalation Accuracy (vs. ground truth labels)")
    print(f"{'='*60}")
    if intent_total > 0:
        print(f"  RAG    — Intent: {intent_correct['rag']}/{intent_total} ({intent_correct['rag']/intent_total*100:.1f}%)  |  Escalation: {escalation_correct['rag']}/{intent_total} ({escalation_correct['rag']/intent_total*100:.1f}%)")
        print(f"  Simple — Intent: {intent_correct['simple']}/{intent_total} ({intent_correct['simple']/intent_total*100:.1f}%)  |  Escalation: {escalation_correct['simple']}/{intent_total} ({escalation_correct['simple']/intent_total*100:.1f}%)")

    # Per-intent breakdown
    print(f"\n{'='*60}")
    print("Per-Intent Judge Score Breakdown (RAG Agent)")
    print(f"{'='*60}")
    intent_scores = defaultdict(list)
    for r in results:
        intent = r['rag']['output']['intent']
        if r['rag']['judge_score'] > 0:
            intent_scores[intent].append(r['rag']['judge_score'])

    for intent in sorted(intent_scores.keys()):
        scores = intent_scores[intent]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {intent:<25} avg={avg:.2f}  n={len(scores)}")

    # Save results
    output = {
        "metadata": {
            "n_queries": n,
            "agent_provider": provider_name,
            "judge_provider": judge_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "aggregate": {
            "rag": {"avg_judge": calc_metrics('rag')[0], "escalation_rate": calc_metrics('rag')[1],
                    "rouge1": calc_metrics('rag')[2], "rouge2": calc_metrics('rag')[3], "rougeL": calc_metrics('rag')[4],
                    "intent_accuracy": intent_correct['rag'] / intent_total if intent_total else 0,
                    "escalation_accuracy": escalation_correct['rag'] / intent_total if intent_total else 0},
            "simple": {"avg_judge": calc_metrics('simple')[0], "escalation_rate": calc_metrics('simple')[1],
                       "rouge1": calc_metrics('simple')[2], "rouge2": calc_metrics('simple')[3], "rougeL": calc_metrics('simple')[4],
                       "intent_accuracy": intent_correct['simple'] / intent_total if intent_total else 0,
                       "escalation_accuracy": escalation_correct['simple'] / intent_total if intent_total else 0},
            "trivial": {"avg_judge": calc_metrics('trivial')[0], "escalation_rate": calc_metrics('trivial')[1],
                        "rouge1": calc_metrics('trivial')[2], "rouge2": calc_metrics('trivial')[3], "rougeL": calc_metrics('trivial')[4]},
        },
        "per_query": results,
    }

    with open("data/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to data/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()
