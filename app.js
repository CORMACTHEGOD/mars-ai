/* Mars AI public frontend preview: no AI requests or API secrets are used here. */
"use strict";

const STORAGE_KEY = "mars-ai-preview-chats-v1";
const historyList = document.getElementById("history");
const emptyState = document.getElementById("empty-state");
const messagesList = document.getElementById("messages");
const input = document.getElementById("prompt");
const chatForm = document.getElementById("chat-form");
const toast = document.getElementById("toast");
const imageInput = document.getElementById("image-input");
const attachmentPreview = document.getElementById("attachment-preview");
const sidebar = document.getElementById("sidebar");
const menuButton = document.getElementById("menu-button");
const aiStatus = document.getElementById("ai-status");
const previewDisclaimer = document.getElementById("preview-disclaimer");
const composerNote = document.getElementById("composer-note");
const sidebarStatus = document.getElementById("sidebar-status");
const sendButton = document.querySelector(".send-button");
const ACCESS_STORAGE_KEY = "mars-ai-beta-access-session";
let aiEnabled = false;
let isSending = false;

async function detectAiBackend() {
  // GitHub Pages is static: no server-side AI endpoint is hosted there.
  if (location.hostname.endsWith(".github.io")) return false;
  try {
    const response = await fetch("/api/chat", { cache: "no-store" });
    if (!response.ok) return false;
    const status = await response.json();
    if (status.service !== "mars-chat") return false;

    if (!status.configured) {
      aiStatus.lastChild.textContent = " AI setup needed";
      sidebarStatus.textContent = "The secure AI server is deployed, but the owner has not activated it yet.";
      composerNote.textContent = "Mars needs a server-side model token and private beta access code to start answering.";
      return false;
    }
    aiEnabled = true;
    aiStatus.classList.add("live");
    aiStatus.lastChild.textContent = " AI beta";
    previewDisclaimer.textContent = "Mars can now answer text questions. Search, images and other tools are still coming.";
    sidebarStatus.textContent = "Mars Chat beta is available here. Other tools are in development.";
    composerNote.textContent = "Private beta: your first AI message asks for a tester access code. Messages are sent to an AI provider.";
    return true;
  } catch {
    return false;
  }
}

function readAccessCode() {
  try { return sessionStorage.getItem(ACCESS_STORAGE_KEY) || ""; }
  catch { return ""; }
}
function rememberAccessCode(code) {
  try { sessionStorage.setItem(ACCESS_STORAGE_KEY, code); } catch { /* session storage may be disabled */ }
}
function forgetAccessCode() {
  try { sessionStorage.removeItem(ACCESS_STORAGE_KEY); } catch { /* ignore */ }
}
function getAccessCode() {
  const existing = readAccessCode();
  if (existing) return existing;
  // This is a shared beta *access code*, never an AI provider API token.
  const code = window.prompt("Mars AI private beta: enter your access code (NOT a provider API key):");
  if (!code || !code.trim()) return null;
  rememberAccessCode(code.trim());
  return code.trim();
}

function buildModelHistory(chat) {
  // Strip old placeholder replies and any temporary thinking indicator.
  const messages = chat.messages
    .filter(entry => !entry.pending && ["user", "assistant"].includes(entry.role))
    .filter(entry => !(entry.role === "assistant" && entry.text.startsWith("Mars AI preview:")))
    .slice(-8)
    .map(entry => ({ role: entry.role, content: entry.text.slice(0, 2500) }));
  let total = messages.reduce((count, entry) => count + entry.content.length, 0);
  while (total > 7500 && messages.length > 1) {
    const oldest = messages.shift();
    total -= oldest.content.length;
  }
  return messages;
}


/* The Mars scrollbar is a separate on-screen indicator. Its position follows
   the real scroll offset instead of relying on browser-specific scrollbar CSS. */
function makeMarsScrollIndicator(scroller, host, options = {}) {
  const track = document.createElement("div");
  track.className = "mars-scroll-track";
  track.setAttribute("aria-hidden", "true");

  const dot = document.createElement("span");
  dot.className = "mars-scroll-dot";
  track.appendChild(dot);
  host.appendChild(track);

  function update() {
    const rect = scroller.getBoundingClientRect();
    const hostRect = host.getBoundingClientRect();
    const maximumScroll = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    const inset = options.inset ?? 10;
    const length = Math.max(0, Math.round(rect.height - inset * 2));
    const visible = length > 20 && rect.width > 0 &&
      (maximumScroll > 2 || options.alwaysShow === true);

    track.classList.toggle("is-visible", visible);
    if (!visible) return;

    track.style.top = Math.round(rect.top - hostRect.top + inset) + "px";
    track.style.right = Math.round(hostRect.right - rect.right + (options.right ?? 10)) + "px";
    track.style.height = length + "px";

    const range = Math.max(0, length - 12);
    const fraction = maximumScroll > 0
      ? Math.min(1, Math.max(0, scroller.scrollTop / maximumScroll))
      : 0;
    dot.style.transform = "translateY(" + Math.round(range * fraction) + "px)";
  }

  scroller.addEventListener("scroll", update, { passive: true });
  return update;
}

const refreshChatMarker = makeMarsScrollIndicator(
  document.querySelector(".chat-area"),
  document.getElementById("view-chat"),
  { alwaysShow: true, right: 12 }
);
const refreshHistoryMarker = makeMarsScrollIndicator(
  historyList,
  sidebar,
  { right: 8 }
);
const refreshComposerMarker = makeMarsScrollIndicator(
  input,
  chatForm,
  { right: 9, inset: 5 }
);

function refreshScrollIndicators() {
  refreshChatMarker();
  refreshHistoryMarker();
  refreshComposerMarker();
}

function resizeComposer() {
  // Do not let browsers show a native resize grip or scrollbar.
  // The field grows with text until 180px, then scrolls invisibly.
  const minimum = window.matchMedia("(max-width: 650px)").matches ? 56 : 60;
  const maximum = 180;
  input.style.resize = "none";
  input.style.scrollbarWidth = "none";
  input.style.height = "auto";
  const contentHeight = input.scrollHeight;
  input.style.height = Math.min(Math.max(contentHeight, minimum), maximum) + "px";
  input.style.overflowY = contentHeight > maximum ? "auto" : "hidden";
  refreshComposerMarker();
}

let chats = loadChats();
let activeChatId = null;
let selectedImageName = null;
let toastTimer = null;

function loadChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(saved)) return [];
    return saved.filter(c => c && typeof c.id === "string" && Array.isArray(c.messages));
  } catch {
    return [];
  }
}

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch {
    notify("Browser storage is unavailable. These chats may not be saved.");
  }
}

function notify(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4800);
}

function makeId() {
  if (globalThis.crypto && crypto.randomUUID) return crypto.randomUUID();
  return String(Date.now()) + "-" + Math.random().toString(36).slice(2);
}

function findActiveChat() {
  return chats.find(chat => chat.id === activeChatId) || null;
}

function renderHistory() {
  historyList.replaceChildren();
  if (!chats.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "Your conversations will appear here.";
    historyList.appendChild(empty);
    requestAnimationFrame(refreshHistoryMarker);
    return;
  }

  for (const chat of chats) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item" + (chat.id === activeChatId ? " selected" : "");
    button.textContent = String(chat.title || "Untitled conversation");
    button.title = button.textContent;
    button.addEventListener("click", () => {
      activeChatId = chat.id;
      renderChat();
      renderHistory();
      closeSidebar();
    });
    historyList.appendChild(button);
  }
  requestAnimationFrame(refreshHistoryMarker);
}

function makeMessage(role, text) {
  const message = document.createElement("div");
  message.className = "message " + (role === "user" ? "user" : "assistant");

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "YOU" : "M";
  avatar.setAttribute("aria-hidden", "true");

  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = text;
  message.append(avatar, body);
  return message;
}

function renderChat() {
  const chat = findActiveChat();
  const hasMessages = Boolean(chat && chat.messages.length);
  emptyState.hidden = hasMessages;
  messagesList.hidden = !hasMessages;
  messagesList.replaceChildren();

  if (hasMessages) {
    for (const entry of chat.messages) {
      messagesList.appendChild(makeMessage(entry.role, entry.text));
    }
    requestAnimationFrame(() => {
      const area = document.querySelector(".chat-area");
      area.scrollTop = area.scrollHeight;
      refreshChatMarker();
    });
  } else {
    requestAnimationFrame(refreshChatMarker);
  }
}

function closeSidebar() {
  sidebar.classList.remove("open");
  menuButton.setAttribute("aria-expanded", "false");
}

function clearAttachment() {
  selectedImageName = null;
  imageInput.value = "";
  attachmentPreview.hidden = true;
  attachmentPreview.textContent = "";
}

function newChat() {
  activeChatId = null;
  input.value = "";
  resizeComposer();
  clearAttachment();
  renderChat();
  renderHistory();
  closeSidebar();
  input.focus();
}

async function submitChat() {
  const text = input.value.trim();
  if (!text) return notify("Write a message first.");
  if (isSending) return notify("Mars is already answering your message.");

  isSending = true;
  sendButton.disabled = true;
  try {
    const online = await aiReadyPromise;
    let accessCode = null;
    if (online) {
      accessCode = getAccessCode();
      if (!accessCode) return notify("An access code is required for this private AI beta.");
    }

    let chat = findActiveChat();
    if (!chat) {
      chat = {
        id: makeId(),
        title: text.replace(/\s+/g, " ").slice(0, 36),
        messages: []
      };
      chats.unshift(chat);
      activeChatId = chat.id;
    }

    const attachmentNote = selectedImageName
      ? "\n\n[Image selected for preview: " + selectedImageName + " — not uploaded or analyzed]"
      : "";
    chat.messages.push({ role: "user", text: text + attachmentNote });
    persist();
    input.value = "";
    resizeComposer();
    clearAttachment();

    if (!online) {
      chat.messages.push({
        role: "assistant",
        text: "Mars AI status: Mars-1 Mini was trained successfully on GitHub, but its model weights are not hosted or connected to this website yet. GitHub Pages only serves the interface, so it cannot run our Python model. We're working toward hosting Mars-1 Mini behind a secure API; its first text generation is still experimental."
      });
      persist();
      renderChat();
      renderHistory();
      return;
    }

    const modelMessages = buildModelHistory(chat);
    const thinking = { role: "assistant", text: "Mars is thinking…", pending: true };
    chat.messages.push(thinking);
    if (activeChatId === chat.id) renderChat();
    renderHistory();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Mars-Beta-Code": accessCode
        },
        body: JSON.stringify({ messages: modelMessages })
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 401) {
          forgetAccessCode();
          throw new Error("The beta code was incorrect. Send another message to enter it again.");
        }
        throw new Error(result.error || "Mars could not answer right now.");
      }
      if (typeof result.reply !== "string" || !result.reply.trim()) {
        throw new Error("Mars received an empty AI answer.");
      }
      thinking.text = result.reply;
    } catch (error) {
      thinking.text = "Mars couldn't finish that message. " +
        (error instanceof Error ? error.message : "Please try again.");
    }
    delete thinking.pending;
    persist();
    if (activeChatId === chat.id) renderChat();
    renderHistory();
  } finally {
    isSending = false;
    sendButton.disabled = false;
    input.focus();
  }
}

input.addEventListener("input", resizeComposer);
window.addEventListener("resize", () => {
  resizeComposer();
  refreshScrollIndicators();
});

chatForm.addEventListener("submit", event => {
  event.preventDefault();
  submitChat();
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.getElementById("new-chat").addEventListener("click", newChat);

document.getElementById("clear-history").addEventListener("click", () => {
  if (!chats.length) return notify("There are no saved chats to clear.");
  if (!confirm("Delete all Mars preview conversations saved in this browser?")) return;
  chats = [];
  persist();
  newChat();
  notify("Local conversation history cleared.");
});

document.getElementById("attach-button").addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    clearAttachment();
    return notify("Please select an image file.");
  }
  selectedImageName = file.name.slice(0, 100);
  attachmentPreview.textContent = "Image selected: " + selectedImageName + " (preview only; not uploaded)";
  attachmentPreview.hidden = false;
});

menuButton.addEventListener("click", () => {
  const open = sidebar.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(open));
});

document.addEventListener("keydown", event => {
  const target = event.target;
  const typing = target instanceof HTMLElement &&
    (target.matches("input, textarea") || target.isContentEditable);
  if (!typing && event.key.toLowerCase() === "n" && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault();
    newChat();
  }
  if (event.key === "Escape") closeSidebar();
});

renderHistory();
renderChat();
resizeComposer();
requestAnimationFrame(refreshScrollIndicators);
const aiReadyPromise = detectAiBackend();
if ("ResizeObserver" in window) {
  const markerObserver = new ResizeObserver(refreshScrollIndicators);
  markerObserver.observe(document.querySelector(".chat-area"));
  markerObserver.observe(historyList);
  markerObserver.observe(chatForm);
}
