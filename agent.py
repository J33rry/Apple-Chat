import json
import abc
import os
import re
import time

VALID_INTENTS = [
    "battery_issue",
    "software_update_bug",
    "app_crash",
    "account_help",
    "hardware_damage",
    "general_inquiry",
]


def get_best_provider():
    """Return GeminiProvider if API key is available, otherwise LocalProvider."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            provider = GeminiProvider()
            return provider
        except Exception:
            pass
    return LocalProvider()


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt == 2:
                    print(f"Gemini API error after 3 retries: {e}")
                    return ""
                time.sleep(2 ** attempt)


class LocalProvider(LLMProvider):
    def __init__(self):
        self.api_url = "http://127.0.0.1:1234/v1/chat/completions"
        self.model = "google/gemma-4-e4b"

    def generate(self, prompt: str) -> str:
        import urllib.request
        import urllib.error

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    text = result['choices'][0]['message']['content'].strip()
                    if text:
                        return text
                    # Empty response — retry
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return ""
            except urllib.error.URLError as e:
                if attempt == 2:
                    print(f"LocalProvider error after 3 retries: {e}")
                    return "Failed to generate response locally. Ensure LM Studio is running."
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    print(f"LocalProvider unexpected error: {e}")
                    return ""
                time.sleep(2 ** attempt)
        return ""


class SupportAgent:
    def __init__(self, llm_provider: LLMProvider, knowledge_base_path: str):
        self.llm = llm_provider
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self._build_index()

    def _build_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.kb_documents = []
        self.kb_responses = []
        # Sample up to 10k threads to prevent memory issues
        sample_kb = self.knowledge_base[:10000] if len(self.knowledge_base) > 10000 else self.knowledge_base
        for thread in sample_kb:
            customer_tweet = None
            for t in thread:
                if t['inbound'] and not customer_tweet:
                    customer_tweet = t['text']
                elif t['author_id'] == 'AppleSupport' and customer_tweet:
                    self.kb_documents.append(customer_tweet)
                    self.kb_responses.append(t['text'])
                    break
        if self.kb_documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.kb_documents)
        else:
            self.tfidf_matrix = None

    def _load_knowledge_base(self, path: str):
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            return []

    def _normalize_intent(self, raw_intent: str) -> str:
        """Validate and normalize intent output. Falls back to general_inquiry."""
        cleaned = raw_intent.strip().lower().replace('"', '').replace("'", "")
        # Remove any markdown or extra text
        cleaned = cleaned.split('\n')[0].strip()
        # Try exact match first
        if cleaned in VALID_INTENTS:
            return cleaned
        # Try partial/fuzzy match for truncated outputs like "software_update_b"
        for valid in VALID_INTENTS:
            if valid.startswith(cleaned) and len(cleaned) >= 5:
                return valid
        # Try keyword matching as last resort
        keyword_map = {
            "battery": "battery_issue",
            "software": "software_update_bug",
            "update": "software_update_bug",
            "crash": "app_crash",
            "account": "account_help",
            "hardware": "hardware_damage",
            "damage": "hardware_damage",
        }
        for keyword, intent in keyword_map.items():
            if keyword in cleaned:
                return intent
        return "general_inquiry"

    def classify(self, tweet_text: str) -> str:
        prompt = f"""Classify the following customer support tweet into exactly one of these intents:
- battery_issue
- software_update_bug
- app_crash
- account_help
- hardware_damage
- general_inquiry

Tweet: "{tweet_text}"
Respond ONLY with the exact intent name from the list above. No explanation, no markdown, no punctuation.
Intent:"""
        raw = self.llm.generate(prompt).strip()
        return self._normalize_intent(raw)

    def retrieve_context(self, tweet_text: str) -> str:
        default_context = "Historically, we suggest restarting the device, checking for updates, or visiting an Apple Store."
        if getattr(self, 'tfidf_matrix', None) is None or not self.kb_documents:
            return default_context

        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([tweet_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        # Get top-3 most similar and combine context
        top_indices = sims.argsort()[-3:][::-1]
        contexts = []
        for idx in top_indices:
            if sims[idx] > 0.1:
                contexts.append(self.kb_responses[idx])
        if contexts:
            return " | ".join(contexts)
        return default_context

    def draft_reply(self, tweet_text: str, intent: str, context: str) -> str:
        prompt = f"""You are an Apple Support agent replying on Twitter.

Customer Tweet: "{tweet_text}"
Identified Intent: {intent}
Historical Context from similar past resolutions: {context}

Instructions:
- Draft a short (1-3 sentences), helpful, and polite reply.
- Ground your answer in the historical context provided.
- Include a specific actionable step the customer can take.
- Use a warm, professional tone consistent with Apple Support.
- Do NOT use hashtags or marketing language.

Reply:"""
        return self.llm.generate(prompt).strip()

    def decide_escalation(self, tweet_text: str, intent: str) -> dict:
        prompt = f"""Analyze the following customer tweet and determine if it should be escalated to a human agent or if it can be auto-handled.
Escalate if the issue involves hardware damage, high frustration, or complex account issues.

Tweet: "{tweet_text}"
Intent: {intent}

Respond strictly in valid JSON format with keys "escalate" (boolean) and "reason" (string). Do not add markdown blocks or conversational text.
Example: {{"escalate": true, "reason": "Customer is highly frustrated."}}
"""
        response = self.llm.generate(prompt)
        return self._parse_escalation(response)

    def _parse_escalation(self, response: str) -> dict:
        """Robustly parse escalation decision with multiple fallback strategies."""
        if not response or not response.strip():
            return {"escalate": True, "reason": "Empty LLM response — defaulting to escalation."}

        # Strategy 1: Direct JSON parse
        try:
            cleaned = response.strip()
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1].split('```')[0]
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, IndexError):
            pass

        # Strategy 2: Regex extraction
        try:
            esc_match = re.search(r'"escalate"\s*:\s*(true|false)', response, re.IGNORECASE)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
            if esc_match:
                escalate = esc_match.group(1).lower() == 'true'
                reason = reason_match.group(1) if reason_match else "Parsed via regex fallback."
                return {"escalate": escalate, "reason": reason}
        except Exception:
            pass

        # Strategy 3: Keyword detection
        response_lower = response.lower()
        if 'not escalat' in response_lower or 'no escalat' in response_lower or '"escalate": false' in response_lower:
            return {"escalate": False, "reason": f"Keyword-detected non-escalation. Raw: {response[:100]}"}

        return {"escalate": True, "reason": f"Failed to parse escalation decision. Raw: {response[:100]}"}

    def handle_tweet(self, tweet_text: str) -> dict:
        intent = self.classify(tweet_text)
        context = self.retrieve_context(tweet_text)
        reply = self.draft_reply(tweet_text, intent, context)
        escalation = self.decide_escalation(tweet_text, intent)

        return {
            "intent": intent,
            "drafted_reply": reply,
            "escalate": escalation.get("escalate", True),
            "escalation_reason": escalation.get("reason", "Unknown")
        }
