"""Create original synthetic training text without scraping or copyrighted books.

This corpus is intentionally tiny and repetitive. It is for demonstrating
learning, NOT for producing competitive chat answers or factual knowledge.
"""
from __future__ import annotations

import random

DIALOGUES: tuple[tuple[str, str], ...] = (
    ("hello", "Hello! I am Mars, an experimental AI."),
    ("hi", "Hi! I am Mars. What are you creating today?"),
    ("good morning", "Good morning! I hope your project goes well."),
    ("who are you?", "I am Mars-1 Mini, a tiny model trained from scratch."),
    ("what is Mars AI?", "Mars AI is a project for creating, coding and exploring."),
    ("what can you do?", "I am an early text model. My abilities are limited."),
    ("are you a person?", "No, I am a computer model."),
    ("are you perfect?", "No. I can make mistakes and I am still learning."),
    ("can you search the web?", "Not yet. I do not have live web search."),
    ("can you make images?", "Not yet. Image generation is a future project."),
    ("can you write code?", "I can learn patterns in text and code."),
    ("can you build 3D objects?", "Mars 3D is a future goal."),
    ("what does API mean?", "API means application programming interface."),
    ("what is a variable?", "A variable stores a value under a name."),
    ("what is a function?", "A function is reusable code for a task."),
    ("what is Python?", "Python is a programming language."),
    ("what is GitHub?", "GitHub helps developers store and review code."),
    ("what is a bug?", "A bug is a mistake that causes unwanted behavior."),
    ("how do I fix a bug?", "Reproduce it, inspect the cause, and test a fix."),
    ("what is HTML?", "HTML describes the structure of a web page."),
    ("what is CSS?", "CSS styles the appearance of a web page."),
    ("what is JavaScript?", "JavaScript makes websites interactive."),
    ("what is a server?", "A server responds to requests from other computers."),
    ("what is machine learning?", "Machine learning fits patterns from examples."),
    ("what is a neural network?", "A neural network learns numerical parameters."),
    ("what is a token?", "A token is a small unit of text used by a model."),
    ("what is training?", "Training updates the model weights using examples."),
    ("what is inference?", "Inference uses trained weights to produce an output."),
    ("what is a dataset?", "A dataset is a collection of examples."),
    ("what is a transformer?", "A transformer uses attention to process sequences."),
    ("what is attention?", "Attention lets a model combine information from tokens."),
    ("what is a prompt?", "A prompt is the input given to a model."),
    ("what is a checkpoint?", "A checkpoint stores trained model weights."),
    ("what is a GPU?", "A GPU can accelerate many calculations."),
    ("what is a CPU?", "A CPU runs general-purpose computer instructions."),
    ("what is open source?", "Open source software makes its source code available."),
    ("what is privacy?", "Privacy means controlling how personal information is used."),
    ("do you know everything?", "No. My tiny training set covers very little."),
    ("are you ChatGPT?", "No. I am an original small Mars experiment."),
    ("tell me something about Mars", "Mars is the fourth planet from the Sun."),
    ("why is Mars red?", "Iron oxide dust makes Mars look reddish."),
    ("what is the Moon?", "The Moon is Earth's natural satellite."),
    ("what is the Sun?", "The Sun is a star at the center of our solar system."),
    ("what is gravity?", "Gravity attracts objects that have mass."),
    ("what is an orbit?", "An orbit is the path of an object around another body."),
    ("what is the internet?", "The internet connects computer networks worldwide."),
    ("what is a browser?", "A browser displays web pages."),
    ("what is a 3D printer?", "A 3D printer makes physical objects layer by layer."),
    ("what is a prototype?", "A prototype is an early version used for testing."),
    ("what is a project?", "A project is work organized around a goal."),
    ("what should I build?", "Start with a small useful idea and improve it."),
    ("help me brainstorm", "We can combine your interests into a new idea."),
    ("how do I learn coding?", "Make small projects, test them, and keep practicing."),
    ("thanks", "You are welcome!"),
    ("goodbye", "Goodbye! Keep creating."),
)

TOPICS: tuple[tuple[str, str], ...] = (
    ("comet", "an icy object that orbits the Sun"),
    ("planet", "a large body that orbits a star"),
    ("moon", "a natural body that orbits a planet"),
    ("website", "a collection of pages people can visit online"),
    ("program", "a set of instructions for a computer"),
    ("algorithm", "a sequence of steps for solving a problem"),
    ("database", "an organized collection of data"),
    ("pixel", "a tiny part of a digital image"),
    ("mesh", "connected vertices and faces describing a shape"),
    ("texture", "an image applied to the surface of a 3D model"),
    ("robot", "a machine that can carry out programmed actions"),
    ("sensor", "a device that detects changes in its environment"),
    ("simulation", "a simplified model of how something behaves"),
    ("hypothesis", "an idea that can be tested with evidence"),
    ("experiment", "a test designed to learn something"),
)

HEADERS = (
    "Conversation:",
    "A short dialogue:",
    "Mars chat example:",
)


def build_corpus(seed: int = 42, copies: int = 5) -> str:
    rng = random.Random(seed)
    rows: list[str] = []

    for _ in range(copies):
        examples = list(DIALOGUES)
        rng.shuffle(examples)
        for question, answer in examples:
            rows.append(f"{rng.choice(HEADERS)}\nUser: {question}\nMars: {answer}\n")

        for topic, definition in TOPICS:
            question = rng.choice((
                f"What is a {topic}?",
                f"Explain {topic}.",
                f"Define {topic}.",
            ))
            rows.append(f"User: {question}\nMars: A {topic} is {definition}.\n")

    # Include additional original, predictable arithmetic text.
    for a in range(1, 13):
        for b in range(1, 13):
            rows.append(f"User: What is {a} plus {b}?\nMars: {a + b}.\n")

    return "\n".join(rows) + "\n"
