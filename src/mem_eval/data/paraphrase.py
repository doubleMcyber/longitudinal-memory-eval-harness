"""Paraphrase families for lexical-gap scenarios (PRD §1 retrieval quality).

To test *memory* rather than *string overlap*, some scenarios state a fact and
query it with words that share MEANING but not TOKENS (a lexical gap). Lexical
bag-of-tokens retrieval then cannot tell the target from same-entity distractors;
a semantic retriever that knows these synonyms can.

`PARAPHRASE_FAMILIES` is generic world knowledge (synonym sets of common
relations), NOT gold: it contains no fact ids, values, or answers. The generator
draws fact/query surface forms from these families, and the offline semantic
embedding (`SynonymHashEmbedding`) uses the same families as its lexical
knowledge — exactly as a real embedding model would know that "relocated" and
"lives" are about residence. Labels remain in the structured layer (§8).
"""

from __future__ import annotations

# concept -> {"fact": [relation phrases], "query": [question templates with {e}]}.
# INVARIANT: the content tokens used in `fact` phrasings and `query` phrasings are
# DISJOINT, so every fact×query pairing has a true lexical gap (no shared content
# token). The generator additionally guards this at construction time.
PARAPHRASE_FAMILIES = {
    "reside": {
        "tokens": [
            # fact-side relation tokens
            "relocated", "moved", "settled",
            # query-side relation tokens
            "reside", "residing", "living", "lives", "live", "residence", "based",
        ],
        "fact": ["relocated to", "moved to", "settled in"],
        "query": ["Where does {e} reside these days?", "Which city is {e} living in?",
                  "What place is {e} residing in?"],
    },
    "employer": {
        "tokens": ["employer", "works", "employed", "job", "workplace", "hired", "staffed"],
        "fact": ["was hired by", "joined the staff of", "works at"],
        "query": ["Who is {e}'s employer?", "What is {e}'s workplace?", "Which firm employs {e}?"],
    },
}

# token -> concept id (the semantic embedding's lexical knowledge)
TOKEN_CONCEPT: dict[str, str] = {
    tok: concept for concept, fam in PARAPHRASE_FAMILIES.items() for tok in fam["tokens"]
}


__all__ = ["PARAPHRASE_FAMILIES", "TOKEN_CONCEPT"]
