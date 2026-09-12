# Decision Log

1. **Brand Selection:** Chose `AppleSupport` because it has a massive volume of tweets and very clear, repeated issue categories (battery, software update, crashes).
2. **Golden Dataset Size:** Settled on 300 threads to provide a robust evaluation set while remaining manageable to manually review and process with local/API limits.
3. **Provider Abstraction:** Created `LLMProvider` abstract class to seamlessly switch between Gemini API (for high-quality testing/judging) and a Local LLM (for bulk processing without costs). Added `get_best_provider()` factory that auto-selects Gemini when an API key is available.
4. **RAG vs Fine-tuning:** Opted for a RAG architecture to ground replies in historical data, as fine-tuning requires significant compute and time for a short assignment.
5. **Data Pipeline DFS:** Used an iterative DFS approach for conversation thread reconstruction instead of recursion to prevent `RecursionError` on deeply nested Twitter threads.
6. **Escalation Logic:** Separated the escalation decision into a distinct JSON-structured prompt to ensure the agent gives a concrete reason, rather than burying it in the drafted text. Added multi-strategy JSON parsing (direct parse → regex extraction → keyword detection) to handle malformed LLM outputs gracefully.
7. **Intent Classes:** Defined intents narrowly (e.g., `battery_issue`, `software_update_bug`) based on manual inspection of AppleSupport data, rather than using a generic dataset like Banking77. Added a normalization layer that validates LLM output against the valid intent set, with fuzzy matching for truncated outputs and keyword fallback.
8. **LLM-as-Judge Setup:** The judge returns an integer (1-5) with retry logic to reduce parse failures. Added ROUGE-1/2/L scores as an LLM-independent complementary metric.
9. **Rate Limiting:** Added provider-aware rate limiting — `time.sleep()` is applied only when using Gemini API, avoiding unnecessary delays with local inference.
10. **Fallback to Escalate:** Designed the escalation parser to default to `True` (escalate) if all parsing strategies fail, ensuring no customer is left hanging due to a code error.
11. **Hybrid Ground Truth Labeling:** Used both LLM classification and keyword-based heuristic patterns to generate ground truth labels. When the two signals disagree, the heuristic (deterministic) is used as the tiebreaker, reducing circularity in evaluation.
12. **Top-3 Context Retrieval:** Changed RAG retrieval from single best match to top-3 most similar historical responses, providing richer context to the reply drafting prompt.
13. **Thread Deduplication:** Added deduplication in the data pipeline to prevent the same root tweet from generating multiple overlapping threads, which would bias the knowledge base.
14. **Missing Dependencies:** Added `scikit-learn` (TF-IDF vectorizer) and `rouge-score` (evaluation metrics) to `requirements.txt` — both were used in code but not listed.
15. **Automated Dataset Setup:** Added `setup_data.py` to download the TWCS dataset from Kaggle automatically. Uses a 3-strategy fallback (kagglehub → kaggle CLI → manual instructions) and skips download if `data/twcs.csv` already exists, making first-time setup a single command.
