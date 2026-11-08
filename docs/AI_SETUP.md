# Mars AI: Real text chat (private beta)

Mars AI can now send text messages to an actual model **after the owner activates a secure Netlify backend**. The public GitHub Pages address remains a static preview.

## Architecture

- `index.html`, `styles.css`, `app.js`: one unified chat interface; no AI provider credentials in the browser.
- `netlify/functions/chat.mjs`: private backend endpoint at `/api/chat` when deployed on Netlify.
- `netlify.toml`: Netlify publishes only `dist/` (copied HTML/CSS/JS), not server source.
- Hugging Face Inference Providers: text model API (currently defaults to `openai/gpt-oss-20b:cheapest`). This is an existing model, **not** a model trained by Mars.
- `MARS_BETA_CODE`: private pilot gate so a random visitor cannot immediately consume your inference credits.
- Search, images, uploads, plugins, model training and 3D are **not** implemented yet.

## No-terminal setup

These are website dashboard steps that the owner must complete. ChatGPT cannot register for external accounts or create secret API tokens on your behalf.

1. Go to [Netlify](https://app.netlify.com/) and choose **Add new project → Import an existing project → GitHub**.
2. Select `CORMACTHEGOD/mars-ai`. Netlify should pick up the `netlify.toml` configuration. Use `main` as the production branch.
3. On [Hugging Face](https://huggingface.co/settings/tokens), generate a **fine-grained token** permitting **Make calls to Inference Providers**, following the site's account requirements and terms.
4. In your **Netlify project → Project configuration → Environment variables**, add the following variables. Use the dashboard's secret-value controls; **never paste real values into GitHub or ChatGPT**:

   | Variable | Value |
   | --- | --- |
   | `HF_TOKEN` | Your private Hugging Face inference token |
   | `MARS_BETA_CODE` | A private, long random tester access code (recommend 20+ characters) |
   | `MARS_ENABLE_AI` | `true` |
   | `MARS_MODEL` (optional) | `openai/gpt-oss-20b:cheapest` |

5. **Redeploy** the Netlify site after setting variables, so the backend receives the new configuration.
6. Open the **Netlify-provided website address**, **not** the `github.io` preview. Send a text question; on the first request the page asks for the **beta access code**, not your Hugging Face API token.
7. When ready, you may move a purchased custom domain to the Netlify project. Do not point DNS to a service you have not successfully deployed.

**If Netlify says 'AI setup needed':** ensure the three required variables are present and redeploy.

**If the AI replies about missing credits:** the Hugging Face free allowance may be exhausted. Check the provider dashboard, rather than repeatedly retrying.

## Important security limitations

This is a **private prototype**, not a safe open-to-the-public paid service yet.

- The beta code is a small barrier, **not proper user authentication**; shared codes can leak.
- The backend caps prompt length, recent history and answer length, but **does not implement per-user rate limiting, abuse monitoring or mature moderation**.
- Do not publicly distribute your tester code. Never enter your AI provider token in the Mars website's chat or access-code prompt.
- Your messages go to the selected AI inference provider. Avoid sending passwords, addresses, financial details or other sensitive information.
- Before a public launch add accounts, robust rate limiting, usage budgeting, content safety checks, privacy policies, and logging that avoids exposing personal information. Services serving young users need additional safeguards.
- An external AI provider has its own terms and pricing, even when some usage is free.
- GitHub Pages is static and cannot host Netlify functions. Its old preview response is intentional until the app moves to a server-capable host.

## Running costs and age requirements

Hugging Face currently advertises a **small free monthly inference credit** ($0.10 for free accounts as of this document), not unlimited messages. Netlify also has usage limits. See the providers' current terms and pricing before switching to a paid plan.

If account eligibility, billing or commercial terms require an adult account holder, ask a parent/guardian to manage it; don't misrepresent your age.

## What is working once configured

- Actual multi-turn text conversation, not a canned reply
- Secure server-side model token
- Private-beta access code
- Honest 'AI not configured' and usage-limit messages
- Local browser conversation history
- Auto-growing message field and custom Mars scroll indicator

Other tools will be wired into this same chat later.
