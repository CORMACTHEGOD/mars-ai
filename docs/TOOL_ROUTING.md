# Mars AI: automatic tool routing design

## The product rule

**One conversation, not separate feature tabs.** Users ask naturally; Mars infers when to use capabilities such as search, vision, image generation, code generation, testing, or plugins. Tool decisions must happen on the **secure AI backend**, never by trusting a simple browser keyword detector.

## Intended request flow

1. Frontend sends the chat conversation and any user-approved attachments to a secure endpoint such as `POST /api/chat` on a separately hosted backend.
2. Backend authenticates the user, enforces rate limits and usage budgets, and sends conversation context plus tool specifications to the AI model.
3. The model either answers directly or requests one or more tool calls through structured tool calling.
4. Backend validates tool names, arguments, permissions, quotas, and output sizes before executing them.
5. The tool results return to the model; the model prepares a final answer, linking to grounded sources and displaying generated artifacts appropriately.
6. Frontend renders text, citations, images, code, or downloads **inside the same conversation**.

## Planned tools

| Tool | Example request | Expected result |
| --- | --- | --- |
| `web_search` | “Find today's Roblox update and cite sources” | Fresh search results with URLs and citations |
| `image_generate` | “Draw a cinematic Martian landscape” | Generated image asset |
| `image_analyze` | “What's in my uploaded photo?” | Answer grounded in the uploaded image |
| `code_write` | “Make a Roblox NPC script” | Copyable code and explanation |
| `project_build` (ToolBloom) | “Build this app for me” | Project files and a working preview |
| `project_test` (ProofLab) | “Test it and fix the bugs” | Actual test results and remaining failures |
| `project_fork` (FutureFork) | “Show two design alternatives” | Multiple isolated versions and comparison |
| `plugin_call` | “Use my connected developer tools” | Permission-scoped service operation |
| `model_3d` (future) | “Make a printable 3D model of this object” | Exportable geometry with validation |

A single request may use multiple tools (for example, search for references, generate code, and run a test). The model can ask for clarification if an intent is ambiguous.

## Security and quality guardrails

- Never expose AI-provider credentials or private backend URLs containing secrets in the public GitHub Pages frontend.
- Require user permission before sharing attachments with external models/services or performing consequential third-party actions.
- Sandbox all code execution, with restricted filesystem, network, CPU, and time limits.
- Validate plugin permissions and keep a clear account connection/disconnection mechanism.
- Enforce per-user quotas, provider cost ceilings, upload size/type validation, and abuse prevention.
- Cite real search results; never invent links, tool executions, test results, or generated assets.
- Label incomplete capabilities honestly, as the current frontend preview does.

## Current status

**Design only.** The `main` website currently runs no AI routing or backend. This document defines how to implement it once the secure model service is connected.
