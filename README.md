# Mars AI 🪐

**Imagine beyond.** Mars is a work-in-progress AI platform for chat, research, coding, creative tools, plugins, and future 3D generation.

## Website preview (v0.2)

The public frontend uses plain HTML, CSS and JavaScript, suitable for GitHub Pages.

Included in this preview:
- A responsive, dark **single-chat** interface
- No Search, Image, Studio, or Plugins tabs: users will describe what they want naturally
- Conversation history saved **only in the current browser** using localStorage
- Prompt suggestions for different future use cases
- Local image file selection (only the filename is displayed; files are not uploaded)

**Important:** The preview does not have a working AI model, live search, image generation, image understanding, a coding agent, server accounts, or automatic tool routing yet. The preview replies are honest placeholders. The planned architecture for routing requests through AI tools is described in [docs/TOOL_ROUTING.md](docs/TOOL_ROUTING.md).

## Publish with GitHub Pages

After merging the frontend pull request:

1. Open **Settings → Pages** in this repository.
2. Set **Source** to **Deploy from a branch**.
3. Choose **main** and **/(root)**, then click **Save**.
4. GitHub will display the published site URL when deployment completes.

Expected URL: https://cormacthegod.github.io/mars-ai/

GitHub Pages serves static files. A real AI needs a separately hosted backend.

## Files

- `index.html` — unified chat interface
- `styles.css` — responsive dark design
- `app.js` — browser-local chat preview and history
- `docs/TOOL_ROUTING.md` — design for automatic AI tool selection

## Long-term goals

- Real AI chat, web search with citations, and image understanding/generation
- Mars Studio: ToolBloom (build), ProofLab (test), and FutureFork (alternatives) behind the same chat
- Permission-controlled plugins and a developer API
- Mars 3D for generating, editing and exporting models

## Security

Never commit API keys, tokens, passwords, or other credentials. Use private server environment variables and keep paid model calls on the backend. Public UI code must not contain secret provider credentials.

This project is an early prototype, not a finished AI service.
