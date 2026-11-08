# Mars AI 🪐

**Imagine beyond.** Mars is a work-in-progress AI platform for chat, research, coding, plugins, creative tools, and future 3D modeling.

## What's built

- A dark, responsive **single-chat** website with the motto **YOUR IDEAS, UNLIMITED**
- Conversation history saved in the user's browser
- A custom Mars scrolling indicator and auto-growing message input
- A **real text-chat backend**, available after the owner configures and deploys it on Netlify
- A private beta access code, so the AI provider's credits are not exposed to every visitor by default

**Important:** The [GitHub Pages preview](https://cormacthegod.github.io/mars-ai/) is *static* and still shows a preview message. The real AI chat only works from a Netlify deployment after the model token and beta settings are supplied.

## Activate real AI chat

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)** for browser-only GitHub + Netlify + Hugging Face setup, without a terminal.

We use an existing external text model for the first release. Mars does **not** yet have its own trained model. Costs, model limits and service requirements apply.

## Files

- `index.html` — unified chat interface
- `styles.css` — responsive dark Mars design
- `app.js` — chat history, UI behavior and AI requests
- `netlify.toml` — publishes only frontend files, deploys server functions
- `netlify/functions/chat.mjs` — secure private text AI API
- `docs/AI_SETUP.md` — activation guide
- `docs/TOOL_ROUTING.md` — future automatic tool-routing architecture

## Mars-1 Mini — our original model

**We are also building an AI model from scratch!** See **[Mars-1 Mini training guide](docs/MARS1_MINI.md)**.

The first experiment is a very small original byte-token Transformer, trained with PyTorch on original example text. It starts with random weights; it does not call an external model provider. Its output will be limited and often unreliable at first.

To train without using a terminal, open **[GitHub Actions](https://github.com/CORMACTHEGOD/mars-ai/actions)** and manually run **Train original Mars-1 Mini (manual)**. The workflow gives you a downloadable checkpoint and sample text.

**This training project is separate from the existing Netlify/Hugging Face backend.** The public website still needs separately hosted inference before Mars-1 Mini can answer visitors.

## Roadmap

1. **Real AI text chat** (backend code available, deployment configuration pending)
2. User accounts and proper per-user limits before the general public uses it
3. Web search with citations and image understanding/generation
4. Mars Studio: ToolBloom (builder), ProofLab (tester), and FutureFork (alternatives)
5. Permission-controlled GitHub and other plugins
6. Mars 3D modeling and exports

## Security

Never put AI-provider tokens, passwords, credentials, or private user data in the public repository. Keep credentials in server-side environment variables. The early beta access code is **not** a substitute for proper production authentication or abuse protection.

Mars is an early-stage project, not a finished AI service.
