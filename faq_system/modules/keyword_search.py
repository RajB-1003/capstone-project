"""
keyword_search.py — Phase 2: BM25 Keyword Retrieval

Responsibilities:
  - Tokenize FAQ text using a custom regex tokenizer that PRESERVES structured
    tokens such as course codes (CS-202, ENG-404) as single units
  - Build a BM25Okapi index over the tokenized FAQ corpus
  - Score and rank documents using pure BM25 arithmetic — zero reliance on
    embeddings, vectors, or cosine similarity

Architecture mandate (Phased Agent Prompt Chain Design.docx):
  - BM25 (Best Matching 25) algorithm via rank_bm25.BM25Okapi
  - Handles alphanumeric identifiers that semantic models fail on (sub-word
    tokenizers decompose "CS-202" into meaningless fragments)
  - Independent service — no coupling to the embedding pipeline
  - For short FAQ snippets, b (length normalization) set to 0.5 to avoid
    penalizing concise, high-value answers (architecture doc note)

Tokenizer design (critical — must preserve structured codes):
  - Regex alternation: course-code pattern is matched FIRST, before \w+
  - Matched tokens are lowercased: "CS-202" → "cs-202" (stable BM25 key)
  - Domain stopwords removed from non-code tokens only
  - Course codes from metadata are injected into document text so they score
    even if mentioned only in metadata and not in the body

Public API:
  tokenize_preserve_codes(text)                         -> list[str]
  build_bm25_index(faq_docs)                            -> tuple[BM25Okapi, list[list[str]]]
  search_keyword(query, faq_docs, bm25_index, top_k=5)  -> dict
"""

import re
from rank_bm25 import BM25Okapi


# ──────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────

# Matches structured codes like CS-202, ENG-404, MATH-101 as ONE token.
# The alternation order is intentional: codes are captured before \w+ splits them.
_CODE_PATTERN = re.compile(r'[A-Z]{2,6}-\d{3,4}', re.IGNORECASE)

# Full tokenizer: course codes first, then regular words (len > 1 filtered later)
_TOKEN_RE = re.compile(r'[A-Z]{2,6}-\d{3,4}|\b\w+\b', re.IGNORECASE)

# Stopwords: domain-neutral function words that add no retrieval signal.
# Important: course codes are NEVER filtered as stopwords (checked via _CODE_PATTERN).
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "what", "which", "who", "whom",
    "this", "that", "these", "those", "of", "in", "to", "for", "on", "at",
    "by", "from", "with", "about", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "same", "so", "than",
    "too", "very", "just", "but", "and", "or", "if", "while", "because",
    "although", "until", "since", "any", "also",
    # additional function words
    "without", "take", "get", "go", "up", "down", "its", "been", "upon",
    "whether", "whose", "per", "cannot", "dont", "doesnt", "didnt",
    "however", "therefore", "hence", "thus", "yet", "still",
})


# ──────────────────────────────────────────────────────────────
# 1. Custom tokenizer
# ──────────────────────────────────────────────────────────────

def tokenize_preserve_codes(text: str) -> list[str]:
    """
    Tokenize text while preserving structured codes as single tokens.

    Why a custom tokenizer?
        Standard BM25 implementations use .split() or basic regex \w+.
        These break "CS-202" into ["CS", "202"] — two separate tokens with
        no relation to the original code. BM25 then scores them as independent
        terms, destroying exact-match retrieval for identifiers.

    Strategy:
        1. _TOKEN_RE alternation matches course-code pattern FIRST:
           "CS-202" is captured as one token before \w+ sees it.
        2. All tokens are lowercased for BM25 consistency.
        3. Stopwords are removed — but NEVER if the token is a structured code.
        4. Tokens shorter than 2 chars are discarded (noise like "a", "I").

    Args:
        text: Raw text string (FAQ question, answer, or query).

    Returns:
        list[str]: Cleaned, lowercase tokens with codes preserved as single units.

    Examples:
        "CS-202 prerequisites"   → ["cs-202", "prerequisites"]
        "ENG-404 summer term"    → ["eng-404", "summer", "term"]
        "What is the fee?"       → ["fee"]
    """
    raw_tokens = _TOKEN_RE.findall(text)

    result = []
    for token in raw_tokens:
        token_lower = token.lower()

        # Always keep valid structured codes — never stopword-filter them
        if _CODE_PATTERN.fullmatch(token):
            result.append(token_lower)
            continue

        # Discard single-character noise
        if len(token_lower) < 2:
            continue

        # Remove domain stopwords from regular words
        if token_lower not in _STOPWORDS:
            result.append(token_lower)

    return result


# ──────────────────────────────────────────────────────────────
# 2. BM25 index builder
# ──────────────────────────────────────────────────────────────

def build_bm25_index(
    faq_docs: list[dict],
) -> tuple[BM25Okapi, list[list[str]]]:
    """
    Build a BM25Okapi index from the FAQ corpus.

    Document text = question + answer + metadata course_code (if present).
    Injecting the metadata course_code ensures that even if the body mentions
    the course only once, BM25 term frequency reflects the document's relevance
    to that specific code.

    BM25 parameters:
        k1 = 1.5  (term frequency saturation — default, appropriate for medium docs)
        b  = 0.5  (length normalization factor — reduced from 0.75 default to avoid
                   penalizing concise, high-value FAQ answers; per architecture doc)
        epsilon = 0.25 (floor for IDF to prevent negative scores on common terms)

    Args:
        faq_docs: List of FAQ dicts loaded from faqs.json.

    Returns:
        bm25_index      — BM25Okapi instance ready for get_scores().
        tokenized_corpus — list[list[str]], parallel to faq_docs.
                           Row i is the token list for faq_docs[i].
                           Returned so callers can inspect exact index contents.
    """
    tokenized_corpus = []

    for doc in faq_docs:
        # Base text: question + answer
        text = f"{doc['question']} {doc['answer']}"

        # Inject metadata course_code if present — boosts exact code matching
        course_code = doc.get("metadata", {}).get("course_code", "")
        if course_code:
            # Append twice to give it higher TF weight without hardcoding scores
            text += f" {course_code} {course_code}"

        tokens = tokenize_preserve_codes(text)
        tokenized_corpus.append(tokens)

    # k1=1.5, b=0.5 per architecture doc guidance for short FAQ snippets
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.5, b=0.5)

    return bm25_index, tokenized_corpus


# ──────────────────────────────────────────────────────────────
# 3. Keyword search (pure BM25)
# ──────────────────────────────────────────────────────────────

def search_keyword(
    query: str,
    faq_docs: list[dict],
    bm25_index: BM25Okapi,
    top_k: int = 5,
) -> dict:
    """
    Retrieve the top-k FAQ documents using pure BM25 keyword scoring.

    No embeddings. No cosine similarity. No semantic models.
    Ranking is determined entirely by BM25 term statistics:
      - Term frequency (TF) with saturation parameter k1
      - Inverse document frequency (IDF) — rare terms score higher
      - Document length normalization (b) — shorter docs not penalized as heavily

    Steps:
      1. Tokenize the query using tokenize_preserve_codes().
      2. Call bm25_index.get_scores(query_tokens) → raw BM25 score per document.
      3. Sort descending; take top_k.
      4. Pack into the shared data contract structure.

    Args:
        query:       Raw natural language or keyword query string.
        faq_docs:    Original FAQ records from faqs.json (parallel to BM25 index).
        bm25_index:  BM25Okapi index built by build_bm25_index().
        top_k:       Number of top results to return (default 5).

    Returns:
        dict matching the shared data contract:
        {
            "query":             str,
            "detected_entities": list[str],  # structured codes found in query
            "route_decision":    str,         # always "keyword" from this function
            "retrieved_docs":    list[dict],  # top-k FAQ records + bm25_score
            "scores":            list[float], # BM25 scores, descending
            "rationale":         str
        }

    Note on zero-score documents:
        BM25 returns 0.0 for documents sharing no tokens with the query.
        These are excluded from retrieved_docs — only positive-score matches
        are returned. This is correct behavior: a 0-score document has no
        keyword overlap and should not be surfaced by keyword search.
    """
    # ── Step 1: tokenize query ─────────────────────────────────
    query_tokens = tokenize_preserve_codes(query)

    # Detect structured codes in query for rationale + detected_entities
    detected_codes = [
        t.upper()
        for t in query_tokens
        if _CODE_PATTERN.fullmatch(t)
    ]

    # Edge case: empty token list after preprocessing
    if not query_tokens:
        return {
            "query":             query,
            "detected_entities": [],
            "route_decision":    "keyword",
            "retrieved_docs":    [],
            "scores":            [],
            "rationale":         (
                "Keyword search activated but query produced no valid tokens "
                "after preprocessing (all tokens were stopwords or noise)."
            ),
        }

    # ── Step 2: BM25 scoring ───────────────────────────────────
    # get_scores returns a numpy array of shape (N,) with float64 BM25 scores
    raw_scores = bm25_index.get_scores(query_tokens)  # shape: (N,)

    # ── Step 3: rank — sort by descending score ────────────────
    actual_k = min(top_k, len(faq_docs))
    ranked_indices = sorted(
        range(len(raw_scores)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )[:actual_k]

    # ── Step 4: filter zero-score and pack results ─────────────
    retrieved_docs = []
    scores = []

    for idx in ranked_indices:
        score = float(raw_scores[idx])
        if score <= 0.0:
            continue   # no keyword overlap — do not include

        doc = faq_docs[idx].copy()
        doc["bm25_score"] = round(score, 4)
        retrieved_docs.append(doc)
        scores.append(round(score, 4))

    # ── Step 5: build rationale ────────────────────────────────
    if detected_codes:
        rationale = (
            f"Keyword (BM25) search activated. "
            f"Detected structured code(s): {detected_codes}. "
            f"Query tokens: {query_tokens}. "
            f"Retrieved {len(retrieved_docs)} document(s) with positive BM25 score. "
            f"Top BM25 score: {scores[0]:.4f}."
            if scores else
            f"Keyword (BM25) search activated. "
            f"Detected code(s): {detected_codes} but no documents matched."
        )
    else:
        rationale = (
            f"Keyword (BM25) search activated. "
            f"Query tokens: {query_tokens}. "
            f"Retrieved {len(retrieved_docs)} document(s) with positive BM25 score. "
            f"Top BM25 score: {scores[0]:.4f}."
            if scores else
            "Keyword (BM25) search activated but no documents matched query tokens."
        )

    # ── Step 6: return shared data contract ───────────────────
    return {
        "query":             query,
        "detected_entities": detected_codes,
        "route_decision":    "keyword",
        "retrieved_docs":    retrieved_docs,
        "scores":            scores,
        "rationale":         rationale,
    }
