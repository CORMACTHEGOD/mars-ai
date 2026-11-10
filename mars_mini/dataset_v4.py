"""Mars Data V4: original, deterministic, compositional practice data.

A large *synthetic* data expansion, NOT an internet-scale language corpus.
Each example is generated locally from authored text or clear algorithms;
no pretrained model, third-party API, scraping, or private conversations.

Families, not just identical prompts, are split into train/validation/test.
Validation is for checkpoint selection; test must be used only after training.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .corpus import DIALOGUES
from .dataset_v3 import (
    DEFINITIONS, WORDS, UNSEEN_FACTS, DataPack, Example,
    normalize_prompt, stable_bucket,
)

NAMES = (
    "Ada", "Ben", "Cleo", "Dara", "Eli", "Faye", "Gus", "Hana", "Iris",
    "Jules", "Kira", "Leo", "Mina", "Nico", "Omar", "Pia", "Ravi", "Sana",
)
ITEMS = (
    "blue book", "red scarf", "green cup", "silver key", "small bag",
    "paper map", "yellow hat", "wooden box", "orange ball", "glass jar",
    "purple pen", "striped sock", "toy rocket", "white mug",
)
PLACES = (
    "kitchen", "library", "attic", "garage", "studio",
    "bedroom", "hallway", "classroom", "garden shed", "living room",
)

# Short facts authored for this project. These give some language practice,
# but do not turn a tiny model into a general knowledge assistant.
EXTRA_FACTS = (
    ("a bicycle", "A bicycle is a vehicle powered by a rider using pedals."),
    ("a thermometer", "A thermometer measures temperature."),
    ("a compass", "A compass helps show direction."),
    ("a map", "A map represents places and their positions."),
    ("a recipe", "A recipe gives instructions for preparing food."),
    ("a library", "A library provides books and other resources."),
    ("a battery", "A battery stores energy for electrical devices."),
    ("a keyboard", "A keyboard lets people type letters and commands."),
    ("a sensor", "A sensor detects a change in its surroundings."),
    ("an algorithm", "An algorithm is a sequence of steps for a task."),
    ("a circuit", "A circuit is a path for electric current."),
    ("a triangle", "A triangle has three straight sides."),
    ("a rectangle", "A rectangle has four right angles."),
    ("an adjective", "An adjective describes a noun."),
    ("a verb", "A verb often describes an action or a state."),
    ("a noun", "A noun names a person, place, thing, or idea."),
    ("a paragraph", "A paragraph groups sentences about a topic."),
    ("a summary", "A summary presents the main points briefly."),
    ("a question", "A question asks for information."),
    ("a pattern", "A pattern is a repeated or predictable arrangement."),
    ("a prediction", "A prediction describes what might happen next."),
    ("a habitat", "A habitat is a place where an organism lives."),
    ("a seed", "A seed can grow into a new plant."),
    ("a root", "A root anchors a plant and absorbs water."),
    ("a cloud", "A cloud contains tiny water droplets or ice crystals."),
    ("a rainbow", "A rainbow forms when light is separated into colors."),
    ("a shadow", "A shadow forms when something blocks light."),
    ("a telescope", "A telescope helps us observe distant objects."),
    ("a microphone", "A microphone converts sound into an electrical signal."),
    ("a speaker", "A speaker turns electrical signals into sound."),
    ("a file", "A file is a named collection of computer data."),
    ("a folder", "A folder helps organize files."),
    ("an update", "An update changes software or information."),
    ("an error", "An error is a mistake or an unexpected result."),
    ("a test", "A test checks whether something works as expected."),
    ("a model", "A model is a simplified representation of something."),
)

CHAT_VARIATIONS = (
    (("Hello.", "Hi Mars.", "Hey there, Mars.", "Hello, can we chat?"),
     "Hello! I am Mars, a small experimental AI."),
    (("Who are you?", "Tell me your name.", "What is your name?", "Introduce yourself."),
     "I am Mars, an experimental AI being developed from scratch."),
    (("Are you a human?", "Are you a real person?", "Are you a person or a program?"),
     "I am a computer program, not a person."),
    (("Can you make mistakes?", "Are your answers always correct?", "Do you ever get things wrong?"),
     "Yes. I can be wrong, so important information should be checked."),
    (("Thank you.", "Thanks for helping.", "Thanks, Mars.", "I appreciate the help."),
     "You're welcome."),
    (("Goodbye.", "See you later.", "Bye for now.", "Talk later."),
     "Goodbye!"),
    (("What can you do?", "What kinds of things can you help explain?", "How can you help?"),
     "I can try short explanations and simple tasks, but I am still learning."),
    (("What should I do if I am confused?", "I don't understand a topic. What next?"),
     "Try breaking the problem into smaller steps and ask a specific question."),
    (("How can I check an answer?", "What if an answer seems wrong?"),
     "Compare it with a trusted source and check the steps."),
    (("Can we go step by step?", "Please explain in small steps."),
     "Yes. Tell me what you are working on."),
    (("Can you answer every question?", "Do you know everything?"),
     "No. My training and abilities are limited."),
    (("Why should we test software?", "Why are tests useful?"),
     "Tests help find mistakes before people rely on software."),
    (("What makes a good question?", "How do I ask a clear question?"),
     "Include the goal, relevant details, and what is confusing."),
    (("How do I learn to code?", "What is a good way to practice programming?"),
     "Start with small programs, test them, and build more difficult projects."),
)


def build_pack(seed: int = 42) -> DataPack:
    """Generate tens of thousands of structured examples with disjoint families."""
    splits: dict[str, list[Example]] = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()

    def add_family(family: str, task: str, pairs: tuple[tuple[str, str], ...],
                   provenance: str = "original_rules") -> None:
        bucket = stable_bucket(family, seed, modulo=20)
        split = "train" if bucket < 16 else "validation" if bucket < 18 else "test"
        # Keep the *entire task input family* in exactly one split.
        for i, (prompt, answer) in enumerate(pairs if split == "train" else pairs[:1]):
            key = normalize_prompt(prompt)
            if not key or not answer.strip() or key in seen:
                raise ValueError(f"Empty or repeated V4 prompt: {prompt!r}")
            seen.add(key)
            splits[split].append(Example(
                id=f"v4:{family}:{i}", prompt=prompt, answer=answer,
                task=task, family=family, provenance=provenance,
            ))

    # General arithmetic: unseen operand pairs, not merely unseen sentences.
    for a in range(81):
        for b in range(a, 81):
            add_family(f"v4-add:{a}:{b}", "addition", (
                (f"What is {a} plus {b}?", f"{a+b}."),
                (f"Calculate {b} + {a}.", f"{a+b}."),
            ))
            if a != b:
                add_family(f"v4-compare:{a}:{b}", "comparison", (
                    (f"Which is greater, {a} or {b}?", f"{b}."),
                    (f"Which is smaller, {b} or {a}?", f"{a}."),
                ))
        for b in range(81):
            add_family(f"v4-sub:{a}:{b}", "subtraction", (
                (f"Calculate {a} minus {b}.", f"{a-b}."),
                (f"What is {a} - {b}?", f"{a-b}."),
            ))

    for a in range(16):
        for b in range(a, 16):
            add_family(f"v4-multiply:{a}:{b}", "multiplication", (
                (f"What is {a} times {b}?", f"{a*b}."),
                (f"Calculate {b} * {a}.", f"{a*b}."),
            ))

    for n in range(251):
        answer = "even." if n % 2 == 0 else "odd."
        add_family(f"v4-parity:{n}", "parity", (
            (f"Is {n} even or odd?", answer),
            (f"Tell me the parity of {n}.", answer),
        ))

    # A short reading-comprehension task with combinatorial story variations.
    # A held-out triple is different, but its names and words may have appeared
    # in training; the evaluation is compositional, NOT unseen vocabulary.
    for name in NAMES:
        for item in ITEMS:
            for place in PLACES:
                story = f"{name} left the {item} in the {place}."
                add_family(f"v4-story:{name}:{item}:{place}", "reading", (
                    (f"Read: {story} Where did {name} leave the {item}?",
                     f"In the {place}."),
                    (f"Read: {story} What did {name} leave in the {place}?",
                     f"The {item}."),
                ))

    for word in sorted(set(WORDS) | {
        "apple", "banana", "camera", "dolphin", "engine", "forest",
        "guitar", "island", "jacket", "kitten", "ladder", "mountain",
        "notebook", "ocean", "pencil", "question", "robot", "station",
        "tablet", "umbrella", "village", "window", "yogurt",
    }):
        add_family(f"v4-word:{word}", "word_rules", (
            (f"Reverse the word {word}.", f"{word[::-1]}."),
            (f"Write {word} in uppercase letters.", f"{word.upper()}."),
            (f"Write {word} in lowercase letters.", f"{word.lower()}."),
        ))

    # Short, original natural-language examples: far fewer than algorithmic
    # examples. Keep this limitation visible in the research report.
    for i, (questions, answer) in enumerate(CHAT_VARIATIONS):
        add_family(f"v4-chat:{i}", "conversation",
                   tuple((q, answer) for q in questions), "original_dialogue")
    for i, (question, answer) in enumerate(DIALOGUES):
        add_family(f"v4-original-chat:{i}", "conversation", (
            (question, answer),
            (f"Please answer: {question}", answer),
        ), "original_mars_dialogue")

    # Avoid reintroducing the explicitly unseen V3 concepts into the V3
    # research benchmark. V4's tests are still their own independent splits.
    for concept, answer in DEFINITIONS:
        if concept in UNSEEN_FACTS:
            continue
        add_family(f"v4-definition:{concept}", "definition", (
            (f"Define {concept} in one sentence.", answer),
            (f"What does {concept} mean?", answer),
        ), "original_short_definition")
    for i, (concept, answer) in enumerate(EXTRA_FACTS):
        add_family(f"v4-extra-fact:{i}", "definition", (
            (f"What is {concept}?", answer),
            (f"Explain {concept} briefly.", answer),
        ), "original_short_fact")

    rng = random.Random(seed)
    for records in splits.values():
        rng.shuffle(records)

    pack = DataPack(
        train=tuple(splits["train"]),
        validation=tuple(splits["validation"]),
        test=tuple(splits["test"]),
    )
    verify_pack(pack)
    return pack


def verify_pack(pack: DataPack) -> None:
    if not pack.train or not pack.validation or not pack.test:
        raise ValueError("Empty data split")
    seen_ids, seen_prompts = set(), set()
    family_splits: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for example in getattr(pack, split_name):
            if example.id in seen_ids or normalize_prompt(example.prompt) in seen_prompts:
                raise ValueError("Duplicate example ID or question")
            seen_ids.add(example.id)
            seen_prompts.add(normalize_prompt(example.prompt))
            previous = family_splits.setdefault(example.family, split_name)
            if previous != split_name:
                raise ValueError(f"Cross-split leakage: {example.family}")
            if len(f"User: {example.prompt}\nMars: {example.answer}\n".encode()) > 240:
                raise ValueError(f"Very long example needs review: {example.id}")


def summary(pack: DataPack) -> dict:
    return {
        "dataset": "Mars Data V4",
        "type": "synthetic structured practice + small authored dialogue/fact collection",
        "training_method": "no external corpus, no pretrained AI",
        "warning": "High row count does not mean broad language variety. Arithmetic and templated stories dominate.",
        "splits": {
            name: {
                "rows": len(getattr(pack, name)),
                "tasks": dict(sorted(Counter(e.task for e in getattr(pack, name)).items())),
            } for name in ("train", "validation", "test")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/mars-data-v4")
    args = parser.parse_args()
    pack = build_pack()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        with (path / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for ex in getattr(pack, name):
                out.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
    (path / "summary.json").write_text(json.dumps(summary(pack), indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(summary(pack), indent=2), flush=True)


if __name__ == "__main__":
    main()
