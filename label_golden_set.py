import json
import re
import time
from tqdm import tqdm
from agent import SupportAgent, get_best_provider, VALID_INTENTS


# Keyword-based heuristic labeling as a fallback/alternative to LLM labeling
INTENT_PATTERNS = {
    "battery_issue": [
        r'\bbatter(y|ies)\b', r'\bcharging?\b', r'\bpower\b', r'\bdrain(ing|s|ed)?\b',
        r'\bdead\b.*\b(phone|device)\b', r'\bwon\'?t charge\b',
    ],
    "software_update_bug": [
        r'\bupdate[ds]?\b', r'\bios\s*\d', r'\bbug(s|gy)?\b', r'\bglitch\b',
        r'\bfreez(e|ing|es)\b', r'\bslow\b.*\b(after|since|update)\b',
        r'\bbroken\b.*\bupdate\b', r'\bupdate\b.*\bbroken\b',
    ],
    "app_crash": [
        r'\bcrash(es|ing|ed)?\b', r'\bapp\b.*\b(not|won\'?t)\b.*\b(open|work|load)\b',
        r'\bforce\s*clos', r'\bkeeps?\s*(closing|stopping|crashing)\b',
    ],
    "account_help": [
        r'\baccount\b', r'\blogin\b', r'\bpassword\b', r'\bapple\s*id\b',
        r'\blocked\s*out\b', r'\bicloud\b', r'\bsign\s*in\b', r'\btwo.?factor\b',
        r'\bverif(y|ication)\b',
    ],
    "hardware_damage": [
        r'\bscreen\b.*\b(crack(ed)?|broke(n)?|shatter(ed)?)\b', r'\bbroken\b.*\b(screen|phone|device)\b',
        r'\bwater\s*damage\b', r'\bdrop(ped)?\b', r'\bphysical\b', r'\bdent\b',
        r'\brepair\b', r'\bhardware\b',
    ],
}


def heuristic_classify(text: str) -> str:
    """Classify intent using keyword patterns. Returns best match or general_inquiry."""
    text_lower = text.lower()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[intent] = score
    if scores:
        return max(scores, key=scores.get)
    return "general_inquiry"


def heuristic_should_escalate(text: str, intent: str) -> bool:
    """Heuristic escalation based on frustration signals and intent type."""
    text_lower = text.lower()
    # Always escalate hardware damage
    if intent == "hardware_damage":
        return True
    # Frustration signals
    frustration_patterns = [
        r'\b(worst|terrible|horrible|awful|unacceptable|ridiculous)\b',
        r'\b(hate|angry|furious|disgusted|frustrated)\b',
        r'[!]{2,}',  # Multiple exclamation marks
        r'[A-Z]{5,}',  # Shouting (all caps)
        r'\b(lawsuit|sue|legal|refund|compensation)\b',
    ]
    frustration_score = sum(1 for p in frustration_patterns if re.search(p, text_lower if '!!' not in p else text))
    return frustration_score >= 2 or intent == "account_help"


def label_golden_set():
    with open("data/golden_dataset.json", 'r', encoding='utf-8') as f:
        threads = json.load(f)

    # Prefer Gemini for higher-quality labeling
    provider = get_best_provider()
    provider_name = type(provider).__name__
    print(f"Using {provider_name} for LLM-based labeling.")

    agent = SupportAgent(provider, "data/knowledge_base.json")

    labeled = []

    print("Labeling golden dataset...")
    # Label up to 100 for golden set (increased from 50)
    for thread in tqdm(threads[:100], desc="Labeling"):
        customer_tweet = None
        brand_tweet = None
        for t in thread:
            if t['inbound'] and not customer_tweet:
                customer_tweet = t['text']
            elif t['author_id'] == 'AppleSupport' and customer_tweet and not brand_tweet:
                brand_tweet = t['text']
                break

        if not customer_tweet or not brand_tweet:
            continue

        # LLM-based classification
        llm_intent = agent.classify(customer_tweet)
        llm_escalation = agent.decide_escalation(customer_tweet, llm_intent)

        # Heuristic-based classification (independent signal)
        heuristic_intent = heuristic_classify(customer_tweet)
        heuristic_escalate = heuristic_should_escalate(customer_tweet, heuristic_intent)

        # Use LLM intent if it's valid and non-empty; otherwise fall back to heuristic
        if llm_intent and llm_intent != "general_inquiry":
            final_intent = llm_intent
        elif heuristic_intent != "general_inquiry":
            final_intent = heuristic_intent
        else:
            final_intent = llm_intent  # Both agree on general_inquiry

        # For escalation, use heuristic as tiebreaker
        llm_esc = llm_escalation.get("escalate", True)
        final_escalate = llm_esc if llm_esc == heuristic_escalate else heuristic_escalate

        labeled.append({
            "thread": thread,
            "customer_tweet": customer_tweet,
            "actual_reply": brand_tweet,
            "true_intent": final_intent,
            "true_escalate": final_escalate,
            "labeling_detail": {
                "llm_intent": llm_intent,
                "heuristic_intent": heuristic_intent,
                "llm_escalate": llm_esc,
                "heuristic_escalate": heuristic_escalate,
            }
        })

        # Rate limiting for API providers
        if provider_name == "GeminiProvider":
            time.sleep(1)

    with open("data/golden_dataset_labeled.json", 'w', encoding='utf-8') as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)

    # Print labeling statistics
    from collections import Counter
    intent_dist = Counter(d['true_intent'] for d in labeled)
    esc_count = sum(1 for d in labeled if d['true_escalate'])
    agreement = sum(1 for d in labeled if d['labeling_detail']['llm_intent'] == d['labeling_detail']['heuristic_intent'])

    print(f"\nLabeled {len(labeled)} examples.")
    print(f"Intent distribution: {dict(sorted(intent_dist.items()))}")
    print(f"Escalation rate: {esc_count}/{len(labeled)} ({esc_count/len(labeled)*100:.1f}%)")
    print(f"LLM-Heuristic intent agreement: {agreement}/{len(labeled)} ({agreement/len(labeled)*100:.1f}%)")


if __name__ == "__main__":
    label_golden_set()
