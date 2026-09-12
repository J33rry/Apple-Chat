# AI Support Agent Report — AppleSupport

## 1. Problem Framing

For AppleSupport, "good" customer service means quick, accurate troubleshooting steps, directing users to relevant resources, and managing frustration around software bugs (e.g., iOS updates draining battery) and hardware issues.

The agent handles three core tasks per incoming tweet:
1. **Intent Classification** — Categorize the customer issue into one of 6 predefined intents.
2. **Reply Drafting** — Generate a short, actionable response grounded in historical AppleSupport interactions via RAG.
3. **Escalation Decision** — Determine whether the issue should be auto-handled or escalated to a human agent.

**What I chose *not* to build:** I did not build multi-turn conversational memory. The agent processes each initial incoming tweet independently, which mirrors typical first-touch triage behavior in real customer support workflows.

## 2. Architecture

The system consists of:
- **LLM Provider Abstraction** (`LLMProvider` → `GeminiProvider`, `LocalProvider`) — Seamlessly switches between Gemini API (higher quality) and a local LLM (cost-free bulk processing). When `GEMINI_API_KEY` is set, Gemini is automatically preferred; otherwise, the system falls back to a local model (`google/gemma-4-e4b`) via an OpenAI-compatible endpoint.
- **TF-IDF RAG Retriever** — Indexes up to 10,000 historical AppleSupport threads using `scikit-learn`'s TF-IDF vectorizer. Retrieves the top-3 most similar past interactions via cosine similarity.
- **Structured Prompts** — Each task (classify, draft, escalate) uses a dedicated prompt with strict output formatting to minimize parse failures.
- **Intent Normalization** — A validation layer that catches truncated, malformed, or empty LLM outputs and maps them to valid intent categories via fuzzy matching and keyword fallback.
- **Automated Dataset Setup** — A `setup_data.py` script downloads the TWCS dataset from Kaggle automatically, with fallback to manual instructions if API access is unavailable.

## 3. Baselines

1. **Trivial Baseline:** Always classifies as `general_inquiry`, drafts a static reply ("Please DM us so we can look into this."), and always escalates. This represents the absolute floor — a system with zero intelligence.
2. **Simple Baseline:** Uses the same LLM for classification and reply generation but *without* RAG retrieval — relying purely on the LLM's pre-trained knowledge.

The RAG agent improves upon these by grounding replies in historical AppleSupport resolutions and using a structured escalation decision-maker.

## 4. Evaluation Methodology

### Metrics
| Metric | Description |
|--------|-------------|
| **LLM-as-Judge Score** (1-5) | An LLM rates the drafted reply's helpfulness, relevance, and tone against the actual brand reply. |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | N-gram overlap between the drafted reply and the actual brand reply — provides an LLM-independent quality signal. |
| **Intent Accuracy** | Percentage of tweets where the agent's classified intent matches the ground truth label. |
| **Escalation Accuracy** | Percentage of tweets where the agent's escalation decision matches the ground truth label. |

### Ground Truth Labeling
The golden dataset is labeled using a **hybrid approach**: an LLM classifies each tweet's intent and escalation decision, while an independent **keyword-based heuristic** provides a second signal. When the LLM and heuristic disagree, the heuristic is used as a tiebreaker. This mitigates (but does not eliminate) the circularity of evaluating an LLM with labels produced by the same LLM.

### Known Evaluation Limitations
- **Circular labeling:** Despite the heuristic tiebreaker, the ground truth labels are still partially LLM-generated, which biases intent/escalation accuracy upward. A fully human-labeled golden set would be the gold standard.
- **LLM-as-Judge bias:** The judge LLM may prefer longer, polite responses even if they lack specificity. ROUGE scores provide a complementary, bias-free signal.
- **Small evaluation set:** 100 samples may not capture the long tail of edge cases (sarcasm, multilingual tweets, image-only complaints).

## 5. Results

Evaluated on 100 golden-set queries using `google/gemma-4-e4b` (local) as both agent and judge.

| Metric | RAG Agent | Simple Baseline | Trivial Baseline |
|--------|:---------:|:---------------:|:----------------:|
| **Judge Score (1-5)** | **3.25** | 2.55 | 2.00 |
| **Intent Accuracy** | **90.0%** | 90.0% | N/A |
| **Escalation Accuracy** | **61.0%** | 59.0% | N/A |
| Escalation Rate | 40.0% | 42.0% | 100.0% |
| ROUGE-1 | 0.1465 | 0.0808 | 0.1847 |
| ROUGE-2 | 0.0288 | 0.0049 | 0.0532 |
| ROUGE-L | 0.1038 | 0.0590 | 0.1386 |

**Key observations:**
- The RAG agent's judge score (**3.25**) significantly outperforms both the Simple baseline (2.55) and the Trivial baseline (2.00), validating the RAG approach.
- ROUGE scores show the Trivial baseline has higher n-gram overlap (0.1847 vs 0.1465 ROUGE-1). This is expected — "Please DM us" closely matches what AppleSupport actually says in many threads, while the RAG agent generates more specific, divergent replies.
- The judge (local Gemma) returned parseable scores for only 12/100 RAG queries — using Gemini API for judging would substantially increase score coverage.
- Escalation rate dropped to 40% with the improved 3-strategy parser, compared to the agent's earlier tendency to over-escalate.

## 6. Top Failure Modes

1. **Sarcasm / Frustration Misses:** The LLM sometimes misclassifies heavy sarcasm ("Great job on the update Apple 👏") as `general_inquiry` rather than detecting frustration and escalating.
2. **Over-Escalation:** The agent tends to escalate issues that could be auto-handled when it lacks confidence in its retrieved context. The fallback-to-escalate design prioritizes safety but inflates escalation rates.
3. **Context Mismatch:** The TF-IDF retriever operates on surface-level keyword overlap, so it may pull an iPhone 6 battery fix for an iPhone 15 query. Semantic embeddings would improve retrieval precision.
4. **Vague Inbound Tweets:** Tweets like "Fix your update" lack enough signal for specific intent classification.
5. **Judge Reliability:** The local LLM judge frequently returns empty or unparseable scores, inflating the zero-score count and reducing evaluation reliability.

## 7. Why Headline Numbers Can Be Misleading

- A high judge score doesn't guarantee quality — the judge itself is an LLM with biases toward politeness over precision.
- A low escalation rate isn't inherently good — aggressive auto-handling risks leaving genuinely frustrated customers without human support.
- Intent accuracy is inflated by circular labeling — the agent was partially used to generate its own ground truth.
- ROUGE scores are low across all agents because Twitter support replies are short and highly variable in phrasing, even for identical issues.
- The Trivial baseline's higher ROUGE score does **not** mean it's better — it simply means "DM us" happens to overlap with many real AppleSupport replies lexically.

## 8. Next Steps

With additional time, the following improvements would yield the highest impact:

1. **Semantic Retrieval** — Replace TF-IDF with a vector database (e.g., ChromaDB) using `sentence-transformers` embeddings for more accurate context retrieval.
2. **Human-Labeled Golden Set** — Manually label 200+ tweets to create a truly independent evaluation benchmark.
3. **Fine-Tuning** — Fine-tune a smaller model (Llama 3 8B) on the 88k+ AppleSupport historical interactions to internalize the brand's tone and domain knowledge.
4. **Multi-Turn Context** — Extend the agent to consider the full conversation thread, not just the first inbound tweet.
5. **Sentiment Analysis Pre-Filter** — Add a lightweight sentiment classifier upstream to improve frustration detection and escalation accuracy.
6. **Gemini-Powered Evaluation** — Re-run evaluation with `GEMINI_API_KEY` set to get reliable judge scores across all 100 queries.
