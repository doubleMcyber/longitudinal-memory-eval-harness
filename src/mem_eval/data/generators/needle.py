"""Category 5 — Needle-across-sessions (PRD §6.5).

One distinctive fact is buried among many sessions of distractor content — the
cross-session analog of needle-in-haystack. Gold = the needle item id. Primary:
recall@k at depth (as #sessions grows).
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact, session_timestamp
from mem_eval.data.schema import NEEDLE, Query

_SECRETS = [
    ("vault passcode", "ZEBRA-9981"),
    ("locker combination", "QUARTZ-4417"),
    ("server hostname", "NIMBUS-3302"),
    ("badge serial", "FALCON-7725"),
]


def generate(rng: Random, count: int = 2, needle_depth: int = 24) -> list:
    scenarios = []
    for i in range(count):
        voc = Vocab(Random(rng.randint(0, 2**31)))
        p = Planted(f"needle-{i}", NEEDLE)
        topic, code = _SECRETS[i % len(_SECRETS)]

        needle_day = needle_depth // 3  # buried, not at an edge
        needle = None
        for day in range(needle_depth):
            sd = p.session(day)
            if day == needle_day:
                needle = new_fact(f"needle-{i}-target", "secret", topic, code)
                p.add_turn(sd, f"The {topic} is {code}.", fact=needle)
            else:
                de = voc.entity()
                da, dk = voc.attribute()
                dv = voc.value(dk)
                df = new_fact(f"needle-{i}-d{day}", de, da, dv, is_distractor=True)
                p.add_turn(sd, f"{de}'s {da} is {dv}.", fact=df)

        p.add_query(
            Query(
                query_id=f"needle-{i}-q",
                category=NEEDLE,
                prompt=f"What is the {topic}?",
                as_of=session_timestamp(needle_depth - 1),
                gold_answer=code,
                gold_support=(needle.fact_id,),
            )
        )
        scenarios.append(p.build())
    return scenarios


__all__ = ["generate"]
