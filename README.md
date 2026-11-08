# Mars AI 🪐

**Imagine beyond.** Mars is a work-in-progress AI platform for chat, research, coding, creative tools, plugins, and future 3D generation.

## Website preview (v0.1)

This repository contains the **public frontend**, built with plain HTML, CSS and JavaScript.

Included in the first preview:
- Responsive, dark Mars workspace
- Chat interface and prompt suggestions
- Conversation history stored **only in the current browser** using localStorage
- Preview pages for Explore, Mars Studio, and Plugins
- Planned tools clearly labeled rather than pretending the AI is already connected

**Important:** The preview does **not** have a working AI model, live web search, image generation, server accounts, or a backend. Image selection only displays the filename locally; it does not upload or analyze images.

## Publish with GitHub Pages

Once the frontend pull request is merged:

1. Open this repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Choose branch **main** and folder **/(root)**, then **Save**.
4. GitHub will display the published site address once deployment succeeds.

Expected site URL after Pages is enabled:
https://cormacthegod.github.io/mars-ai/

GitHub Pages serves static files only. The production AI will need a separately hosted, secure backend.

## File layout

- `index.html` — interface and sections
- `styles.css` — responsive Mars visual design
- `app.js` — local preview interactions and chat history

## Roadmap

1. Secure backend + genuine AI chat
2. Search, citations, image understanding, and image generation
3. Mars Studio: ToolBloom, ProofLab, and FutureFork
4. Safe permissions-based plugins and developer API
5. Mars 3D: generate, edit, texture, and export models

## Security

**Never commit API keys, tokens, passwords, or other secrets here.**
Use private backend environment variables. The public static website must not directly embed secret provider credentials.

This project is an early prototype, not a finished AI service.
