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
let isSending = false;
let marsMiniModule = null;

async function detectAiBackend() {
  // Mars-1 Mini is our own model and runs directly on the visitor's device.
  // We only download public ONNX weights and the numerical inference runtime.
  // No inference API, user credentials or Hugging Face access code are needed.
  aiStatus.lastChild.textContent = " Loading Mars-1 Mini…";
  sidebarStatus.textContent = "Starting the original Mars-1 Mini model in your browser.";
  try {
    const browserModel = await import("./mars-mini-browser.js?v=mars-dialogue-v3");
    await browserModel.loadMarsMini();
    marsMiniModule = browserModel;
    aiStatus.classList.add("live");
    aiStatus.lastChild.textContent = " Mars-1 Mini · Local";
    previewDisclaimer.textContent = "Mars-1 Mini is running in your browser. This early model may produce nonsense.";
    sidebarStatus.textContent = "Our original tiny model runs on your device. It does not use a paid AI API.";
    composerNote.textContent = "Mars-1 Mini: 260,736 parameters, trained on a small set of example dialogues. Text stays on your device.";
    return true;
  } catch (error) {
    console.warn("Mars-1 Mini could not start:", error);
    aiStatus.lastChild.textContent = " Model unavailable";
    previewDisclaimer.textContent = "Mars-1 Mini has been trained, but its browser model couldn't be loaded here.";
    sidebarStatus.textContent = "The browser model is unavailable. Check your connection and refresh the page.";
    composerNote.textContent = "Our model file may still be publishing, or the inference library may not have loaded. Refresh to retry.";
    return false;
  }
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
  if (isSending) return notify("Mars is already generating a message.");

  isSending = true;
  sendButton.disabled = true;
  try {
    const ready = await aiReadyPromise;
    if (!ready || !marsMiniModule) {
      notify("Mars-1 Mini isn't available yet. The trained model may still be publishing. Refresh to retry.");
      return;
    }
    if (selectedImageName) {
      notify("Mars-1 Mini can only read text for now. Remove the image and try again.");
      return;
    }

    let chat = findActiveChat();
    if (!chat) {
      chat = {
        id: makeId(),
        title: text.replace(/\\s+/g, " ").slice(0, 36),
        messages: []
      };
      chats.unshift(chat);
      activeChatId = chat.id;
    }

    // This experimental model accepts one prompt, not multi-turn conversations.
    // History is saved for display, but it is not sent to a remote service.
    chat.messages.push({ role: "user", text });
    persist();
    input.value = "";
    resizeComposer();
    clearAttachment();

    const thinking = { role: "assistant", text: "Mars-1 Mini is generating locally…", pending: true };
    chat.messages.push(thinking);
    if (activeChatId === chat.id) renderChat();
    renderHistory();

    try {
      const reply = await marsMiniModule.generateMarsMiniReply(text, {
        maxNewBytes: 90,
        onProgress: (n, max) => {
          thinking.text = `Mars-1 Mini is generating locally… ${n}/${max} bytes`;
          if (activeChatId === chat.id) {
            const last = messagesList.querySelector(".message:last-child .message-content");
            if (last) last.textContent = thinking.text;
          }
        }
      });
      thinking.text = reply;
    } catch (error) {
      console.warn("Mars-1 Mini generation error:", error);
      thinking.text = "Mars-1 Mini couldn't generate text in this browser. " +
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
