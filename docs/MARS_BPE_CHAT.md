# Connecting our original Mars BPE tokenizer to chat messages

**Research preview:** The normal Mars-1 model and the older Mars-2 Lab remain available, unchanged. Our BPE research model is not a general-purpose chatbot.

## What the BPE Lab does

The browser now supports a separate **Mars BPE Lab** chat route:

**https://cormacthegod.github.io/mars-ai/?model=bpe**

You can also use the **Choose Model** links in the Mars website sidebar.

When you type and send a message in BPE Lab:

1. The website loads our original trained `mars-bpe-mini.tokenizer.json` vocabulary.
2. `mars-bpe-browser.mjs` encodes your message into learned token IDs.
3. ONNX Runtime Web runs our own small Transformer from `mars-bpe-mini.onnx` **directly on your device**.
4. The model predicts token IDs and the same BPE tokenizer decodes them to text.
5. The answer appears in the chat history, just like the older Mars models.

No paid model provider, pretrained weights, secret API key, or server-side AI inference is used. This does require downloading the *public model file* to your browser automatically, which is normal; **you don't need to manually download anything**.

## How the model is published

The [Publish original Mars BPE browser experiment workflow](https://github.com/CORMACTHEGOD/mars-ai/actions/workflows/publish-mars-bpe-browser.yml):

- Trains our original BPE Transformer from random weights on Mars Data V3's **training split only**.
- Saves a checkpoint with the exact vocabulary used while training it.
- Exports its own weights to ONNX and tests numerical equivalence with the PyTorch model.
- Verifies that Python and JavaScript BPE encoding return **identical token IDs** for sample prompts.
- Publishes the model and tokenizer together under `models/`, with no credentials embedded in either.
- Triggers the real browser test which sends a message to the optional BPE Lab.

GitHub's Actions logs contain the training and verification results. Downloading artifacts is optional.

## Important limitations

This is the same small-scale BPE model family tested in our earlier byte/BPE experiment. The tokenizer produced much shorter token sequences and somewhat lower validation prediction loss, but **generated answers to unfamiliar questions were still mostly wrong or nonsense**. We are connecting real neural-network inference to the message interface, **not** claiming that the model suddenly understands language.

Each message is currently handled independently as a short prompt. The website saves a conversation list for display, but the BPE model **has not learned reliable multi-turn conversation or long-term memory**. Adding old chat text to a 160-token model without training for conversations would not make it understand context.

## What stays safe

- Default site `/mars-ai/` still uses Mars-1, with its existing local chat history.
- `?model=mars2` still uses the original Mars-2 research model.
- `?model=bpe` has **separate saved chat history** and displays a research warning.
- No one is required to manually configure, download or install any AI tools.
- Browser tests check the actual published model and its vocabulary before treating the experiment as working.

The next development step is a properly trained multi-turn model with more licensed, diverse training text and a longer context window—not simply adding more messages to a weak model.
