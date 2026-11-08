// Mars AI private-beta chat endpoint.
// Runs ONLY as a Netlify Function. Never put model tokens in browser code.
import { createHash, timingSafeEqual } from "node:crypto";

const HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions";
const DEFAULT_MODEL = "openai/gpt-oss-20b:cheapest";
const MAX_REQUEST_BYTES = 16_000;
const MAX_PROMPT_CHARS = 2_500;
const MAX_MESSAGES = 10;
const MAX_CONTEXT_CHARS = 8_000;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    }
  });
}

function configured() {
  return process.env.MARS_ENABLE_AI === "true" &&
    Boolean(process.env.HF_TOKEN) &&
    Boolean(process.env.MARS_BETA_CODE);
}

function verifyCode(input, expected) {
  if (typeof input !== "string" || input.length > 200 || !expected) return false;
  const a = createHash("sha256").update(input).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b);
}

function parseMessages(payload) {
  if (!payload || !Array.isArray(payload.messages) || payload.messages.length === 0 ||
      payload.messages.length > MAX_MESSAGES) return null;
  const filtered = [];
  let total = 0;
  for (const item of payload.messages) {
    if (!item || (item.role !== "user" && item.role !== "assistant") ||
        typeof item.content !== "string" || !item.content.trim() ||
        item.content.length > MAX_PROMPT_CHARS) return null;
    const content = item.content.trim();
    total += content.length;
    filtered.push({ role: item.role, content });
  }
  if (total > MAX_CONTEXT_CHARS || filtered.at(-1).role !== "user") return null;
  return filtered;
}

const SYSTEM_MESSAGE = [
  "You are Mars AI, a helpful, clear and creative assistant.",
  "Be accurate, explain uncertainty, and respond in a friendly, concise way.",
  "You are a text-chat model in an early beta. You do not yet have live search,",
  "image generation, image analysis, code execution, or connected plugins.",
  "Never claim to have searched, viewed an attachment, generated an image,",
  "executed code, or accessed a project unless a working tool actually did so.",
  "You may write and explain code as text.",
  "Keep responses age-appropriate and avoid facilitating dangerous activities."
].join(" ");

export default async function handler(request) {
  if (request.method === "GET") {
    return json({ service: "mars-chat", configured: configured() });
  }
  if (request.method !== "POST") {
    return json({ error: "Method not allowed." }, 405);
  }
  if (!configured()) {
    return json({
      error: "Mars AI has not been activated by its owner yet.",
      code: "not_configured"
    }, 503);
  }
  const suppliedCode = request.headers.get("x-mars-beta-code");
  if (!verifyCode(suppliedCode, process.env.MARS_BETA_CODE)) {
    return json({ error: "Invalid Mars beta access code.", code: "unauthorized" }, 401);
  }

  const size = Number(request.headers.get("content-length"));
  if (Number.isFinite(size) && size > MAX_REQUEST_BYTES) {
    return json({ error: "Message is too large." }, 413);
  }

  let payload;
  try {
    const body = await request.text();
    if (body.length > MAX_REQUEST_BYTES) return json({ error: "Message is too large." }, 413);
    payload = JSON.parse(body);
  } catch {
    return json({ error: "Invalid request." }, 400);
  }

  const messages = parseMessages(payload);
  if (!messages) {
    return json({ error: "Send a shorter message and try again." }, 400);
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 22_000);
  try {
    const upstream = await fetch(HF_ENDPOINT, {
      method: "POST",
      signal: controller.signal,
      headers: {
        Authorization: "Bearer " + process.env.HF_TOKEN,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: process.env.MARS_MODEL || DEFAULT_MODEL,
        messages: [{ role: "system", content: SYSTEM_MESSAGE }, ...messages],
        max_tokens: 600,
        stream: false
      })
    });

    if (!upstream.ok) {
      // Do not leak provider errors, request headers, or server credentials.
      if ([402, 429].includes(upstream.status)) {
        return json({
          error: "Mars has reached its current AI usage allowance. Please try later.",
          code: "limit"
        }, 429);
      }
      return json({
        error: "The AI provider could not complete this request. Please try again.",
        code: "provider"
      }, 502);
    }

    const result = await upstream.json();
    const reply = result?.choices?.[0]?.message?.content;
    if (typeof reply !== "string" || !reply.trim()) {
      return json({ error: "Mars received an empty answer. Please try again." }, 502);
    }
    return json({ reply: reply.slice(0, 10_000), model: "Mars Chat beta" });
  } catch {
    return json({ error: "Mars could not reach its AI provider right now." }, 504);
  } finally {
    clearTimeout(timeout);
  }
}

export const config = {
  path: "/api/chat"
};
