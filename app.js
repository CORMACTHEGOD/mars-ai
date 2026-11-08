/* Mars AI frontend preview. No server calls or third-party AI APIs are made here. */
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
      showView("chat");
      renderChat();
      renderHistory();
    });
    historyList.appendChild(button);
  }
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
    });
  }
}

function newChat() {
  activeChatId = null;
  input.value = "";
  selectedImageName = null;
  imageInput.value = "";
  attachmentPreview.hidden = true;
  attachmentPreview.textContent = "";
  showView("chat");
  renderChat();
  renderHistory();
  input.focus();
}

function submitChat() {
  const text = input.value.trim();
  if (!text) {
    notify("Write a message first. AI chat will be connected in a later build.");
    return;
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
    ? "\n\n[Selected locally: " + selectedImageName + " — image processing not connected]"
    : "";
  chat.messages.push({ role: "user", text: text + attachmentNote });
  chat.messages.push({
    role: "assistant",
    text: "Mars AI preview: Your message has been saved in this browser. The real AI model and secure backend aren't connected yet, so I can't answer or process images in this preview."
  });
  persist();
  input.value = "";
  selectedImageName = null;
  imageInput.value = "";
  attachmentPreview.hidden = true;
  attachmentPreview.textContent = "";
  renderChat();
  renderHistory();
  input.focus();
}

function showView(view) {
  const valid = ["chat", "explore", "studio", "plugins"];
  if (!valid.includes(view)) view = "chat";
  for (const section of document.querySelectorAll(".view")) {
    const isActive = section.id === "view-" + view;
    section.classList.toggle("active", isActive);
    section.hidden = !isActive;
  }
  for (const button of document.querySelectorAll("[data-view]")) {
    button.classList.toggle("active", button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  document.getElementById("page-title").textContent = ({
    chat: "Chat", explore: "Explore", studio: "Mars Studio", plugins: "Plugins"
  })[view];
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("menu-button").setAttribute("aria-expanded", "false");
}

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

for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => showView(button.dataset.view));
}

for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
}

for (const button of document.querySelectorAll("[data-soon]")) {
  button.addEventListener("click", () => notify(button.dataset.soon));
}

document.getElementById("attach-button").addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    imageInput.value = "";
    notify("Please select an image file.");
    return;
  }
  selectedImageName = file.name.slice(0, 100);
  attachmentPreview.textContent = "Image selected: " + selectedImageName + " (local preview only; not uploaded)";
  attachmentPreview.hidden = false;
});

document.getElementById("menu-button").addEventListener("click", () => {
  const sidebar = document.getElementById("sidebar");
  const open = sidebar.classList.toggle("open");
  document.getElementById("menu-button").setAttribute("aria-expanded", String(open));
});

document.addEventListener("keydown", event => {
  const target = event.target;
  const typing = target instanceof HTMLElement &&
    (target.matches("input, textarea") || target.isContentEditable);
  if (!typing && event.key.toLowerCase() === "n" && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault();
    newChat();
  }
  if (event.key === "Escape") {
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("menu-button").setAttribute("aria-expanded", "false");
  }
});

renderHistory();
renderChat();
showView("chat");