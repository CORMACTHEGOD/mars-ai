"""Mars-2 research data: original examples and meaningful held-out splits.

This project does not scrape third-party books, chats, websites, or personal data.
Scores from paraphrased *known* facts are NOT the same as true novel knowledge.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .corpus import DIALOGUES, TOPICS


@dataclass
class Mars2Splits:
    train: list[tuple[str, str]]
    validation: list[tuple[str, str]]
    novel_test: list[tuple[str, str]]


def make_mars2_splits(seed: int = 42) -> Mars2Splits:
    train: list[tuple[str, str]] = []
    validation: list[tuple[str, str]] = []
    novel_test: list[tuple[str, str]] = []

    # Each familiar fact has training questions in three styles, then a
    # fourth, unseen instruction style in validation. Validation answers
    # are familiar; only their *phrasing* is new.
    for question, answer in DIALOGUES:
        question = question.strip()
        train.extend((
            (question, answer),
            (f"Please answer: {question}", answer),
            (f"Hey Mars, {question}", answer),
        ))
        validation.append((f"Could you answer this question: {question}", answer))

    # Reserve three entire concepts from training: the model has no right
    # to claim it knows these facts merely from an unfamiliar question.
    heldout_concepts = {"sensor", "texture", "robot"}
    for topic, definition in TOPICS:
        answer = f"A {topic} is {definition}."
        questions = (
            f"What is a {topic}?",
            f"Define {topic}.",
            f"Explain {topic}.",
        )
        if topic in heldout_concepts:
            novel_test.append((questions[0], answer))
            continue
        train.extend((question, answer) for question in questions)
        validation.append((f"Could you explain a {topic}?", answer))

    # This task has a verifiable answer for unseen NUMBER PAIRS, unlike
    # conversational claims of general intelligence.
    # All values are non-sensitive, independently generated arithmetic.
    for a in range(0, 13):
        for b in range(0, 13):
            answer = f"{a + b}."
            examples = (
                (f"What is {a} plus {b}?", answer),
                (f"Calculate {a} + {b}.", answer),
                (f"Add {a} and {b}.", answer),
            )
            # Same arithmetic template, held-out operand combinations.
            if (a * 17 + b * 11) % 5 == 0:
                novel_test.append(examples[0])
            else:
                train.extend(examples)
                validation.append((f"Please work out {a} + {b}.", answer))

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(novel_test)

    # Ensure no exact prompt overlaps. This does NOT mean the train and
    # validation answers are independent; they deliberately often overlap.
    train_prompts = {q.lower() for q, _ in train}
    val_prompts = {q.lower() for q, _ in validation}
    test_prompts = {q.lower() for q, _ in novel_test}
    assert len(train_prompts) == len(train), "Training prompt collision"
    assert not (train_prompts & val_prompts)
    assert not (train_prompts & test_prompts)
    assert not (val_prompts & test_prompts)
    return Mars2Splits(train, validation, novel_test)


def format_summary(split: Mars2Splits) -> str:
    return (f"Train {len(split.train)} | unseen phrasing validation "
            f"{len(split.validation)} | held-out task examples {len(split.novel_test)}")
