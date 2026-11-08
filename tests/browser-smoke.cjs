/* Check that the ACTUAL Mars-1 Mini model produces text in Chromium.
 * This script runs inside GitHub Actions, not on the public website.
 * It uses the website, ONNX file and browser model code from the repository.
 */
const { spawn } = require("node:child_process");
const { chromium } = require("playwright");

async function waitForServer(url) {
  for (let i = 0; i < 60; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error("Local HTTP server did not start");
}

async function main() {
  const server = spawn("python3", ["-m", "http.server", "9876", "--bind", "127.0.0.1"], {
    stdio: "ignore",
    cwd: process.cwd()
  });
  let browser;
  const errors = [];
  try {
    await waitForServer("http://127.0.0.1:9876/");
    browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
    const page = await browser.newPage();
    page.on("console", msg => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", err => errors.push(err.message));

    await page.goto("http://127.0.0.1:9876/", { waitUntil: "load" });
    await page.waitForFunction(
      () => document.querySelector("#ai-status")?.classList.contains("live"),
      null,
      { timeout: 120000 }
    ).catch(async error => {
      const status = await page.locator("#ai-status").innerText().catch(() => "(missing)");
      throw new Error("Mars model failed to load. Status: " + status +
        "\nBrowser errors: " + errors.join("\n") + "\n" + error.message);
    });
    console.log("Mars-1 Mini weights and browser inference runtime loaded successfully");

    await page.locator("#prompt").fill("hello");
    await page.locator(".send-button").click();
    await page.waitForFunction(
      () => {
        const el = document.querySelector(".message.assistant .message-content");
        if (!el) return false;
        const text = el.textContent || "";
        return text.length > 0 && !text.includes("generating locally");
      },
      null,
      { timeout: 120000 }
    );
    const reply = await page.locator(".message.assistant .message-content").last().innerText();
    if (!reply.trim() || reply.includes("couldn't generate") || reply.includes("not available")) {
      throw new Error("Mars did not generate genuine text: " + reply +
        "\nBrowser errors: " + errors.join("\n"));
    }
    console.log("Mars-1 Mini generated actual browser text:", JSON.stringify(reply));
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
