"""Lexical-gap longitudinal recall (PRD §6.1, §1 retrieval quality).

A fact is stated with one phrasing and queried with a *synonymous* phrasing that
shares no content tokens, among many same-entity distractors. Lexical token-cosine
cannot separate the target from the distractors (they all share only the entity
token, and the target is the oldest, so it falls below top-k); a SEMANTIC
retriever that knows the synonym family ranks the target first. This is what makes
recall measure memory rather than string overlap, and what makes the pluggable
`EmbeddingFn` seam meaningful.

These are tagged as longitudinal_recall (they ARE longitudinal recall, just with a
lexical gap), so the suite keeps exactly 5 categories (A2). Gold is structured.
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact, session_timestamp
from mem_eval.data.paraphrase import PARAPHRASE_FAMILIES
from mem_eval.data.schema import LONGITUDINAL_RECALL, Query
from mem_eval.text import tokenize

# enough same-entity distractors that the lexically-tied target (oldest) is pushed
# below the top-k cutoff for token-cosine across the tested k-band.
DISTRACTORS = 14

# Short, same-length, reside-free predicates so distractors TIE the target on
# token-cosine (both share only the entity token, equal norm) — neutralizing any
# length/norm advantage. The target then loses on recency (it is oldest), so a
# purely lexical retriever drops it; a semantic one keeps it via the relation.
_PREDICATES = ["enjoys", "dislikes", "collects", "studies", "mentioned", "avoids", "praised"]
_OBJECTS = ["jazz", "chess", "pottery", "hiking", "novels", "puzzles", "gardening", "cycling"]


def generate(rng: Random, count: int = 3) -> list:
    scenarios = []
    fam = PARAPHRASE_FAMILIES["reside"]
    for i in range(count):
        voc = Vocab(Random(rng.randint(0, 2**31)))
        p = Planted(f"lexgap-{i}", LONGITUDINAL_RECALL)
        entity = voc.entity()
        city = voc.city()

        # target stated early with phrasing A (relation synonym), value = city
        fact_phrase = fam["fact"][rng.randrange(len(fam["fact"]))]
        fact_tokens = set(tokenize(fact_phrase))
        s0 = p.session(0)
        target = new_fact(f"lexgap-{i}-target", entity, "residence", city)
        p.add_turn(s0, f"{entity} {fact_phrase} {city}.", fact=target)

        # many same-entity distractors: short, reside-free, equal-norm to the target
        for day in range(1, DISTRACTORS + 1):
            pred = _PREDICATES[rng.randrange(len(_PREDICATES))]
            obj = _OBJECTS[rng.randrange(len(_OBJECTS))]
            sd = p.session(day)
            df = new_fact(f"lexgap-{i}-d{day}", entity, pred, obj, is_distractor=True)
            p.add_turn(sd, f"{entity} {pred} {obj}.", fact=df)

        # query with phrasing B — GUARANTEE a true lexical gap: pick a query
        # template whose content tokens don't intersect the fact phrase's (so the
        # only shared token is the entity). The family is built disjoint, so this
        # almost always picks the first; the guard defends against future edits.
        templates = list(fam["query"])
        rng.shuffle(templates)
        q_template = next(
            (t for t in templates if not (set(tokenize(t.format(e=entity))) & fact_tokens)),
            templates[0],
        )
        p.add_query(
            Query(
                query_id=f"lexgap-{i}-q",
                category=LONGITUDINAL_RECALL,
                prompt=q_template.format(e=entity),
                as_of=session_timestamp(DISTRACTORS),
                gold_answer=city,
                gold_support=(target.fact_id,),
            )
        )
        scenarios.append(p.build())
    return scenarios


__all__ = ["generate", "DISTRACTORS"]
