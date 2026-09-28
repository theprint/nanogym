import { BrowserGPT, CharTokenizer } from "./model.js";

const output = document.querySelector("#output");
const outputPlaceholder = document.querySelector("#output-placeholder");
const outputText = document.querySelector("#output-text");
const generateButton = document.querySelector("#generate");
const writeButton = document.querySelector("#write");
const modelDot = document.querySelector("#model-dot");
const modelStats = document.querySelector("#model-stats");
const buttonLabel = generateButton.querySelector(".button-label");
const themeToggle = document.querySelector("#theme-toggle");
const themeLabel = themeToggle.querySelector(".theme-label");
const themeIcon = themeToggle.querySelector(".theme-icon");
const writingMode = document.querySelector("#writing-mode");
const writingPrompt = document.querySelector("#writing-prompt");
const writingInput = document.querySelector("#writing-input");
const exitWritingButton = document.querySelector("#exit-writing");
const saveWritingButton = document.querySelector("#save-writing");

const THEME_KEY = "tiny-prompt-writer-theme";
const ADJECTIVES = [
  "fresh", "new", "special", "custom", "tantalizing", "creative", "curious", "unexpected",
  "playful", "vivid", "strange", "daring", "intriguing", "offbeat", "electric", "original",
];
let model;
let tokenizer;
let currentPrompt = "";
let isBusy = false;

function setModelState(state) {
  modelDot.classList.toggle("is-ready", state === "ready");
  modelDot.classList.toggle("is-error", state === "error");
  const label = state === "ready" ? "Model ready" : state === "error" ? "Model unavailable" : "Loading model";
  modelDot.title = label;
  modelDot.setAttribute("aria-label", label);
}

const placeholderAdjective = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
outputPlaceholder.textContent = `Click 'Generate' to get a ${placeholderAdjective} writing prompt.`;

function setStatus(message, mode = "") {
  // Status is intentionally kept out of the visual layout; the green dot below
  // the prompt is the quiet model indicator.
  void message;
  void mode;
}

function getStoredTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch {
    return null;
  }
}

function applyTheme(theme, remember = false) {
  const isDark = theme === "dark";
  document.documentElement.dataset.theme = isDark ? "dark" : "light";
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeToggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
  themeLabel.textContent = isDark ? "Light mode" : "Dark mode";
  themeIcon.textContent = isDark ? "☼" : "◐";
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", isDark ? "#1b211e" : "#f5f0e8");
  if (remember) {
    try {
      localStorage.setItem(THEME_KEY, isDark ? "dark" : "light");
    } catch {
      // Theme preference is a convenience; the interface still works if storage is blocked.
    }
  }
}

const savedTheme = getStoredTheme();
const preferredTheme = savedTheme || (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
applyTheme(preferredTheme);
themeToggle.addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
});

function assertResponse(response, filename) {
  if (!response.ok) throw new Error(`Could not load ${filename} (${response.status})`);
  return response;
}

function formatParameterCount(count) {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  return `${Math.round(count / 1000)}K`;
}

const TYPE_INTERVAL_MS = 15;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function typeOut(text) {
  outputText.textContent = "";
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion) {
    outputText.textContent = text;
    return;
  }
  for (const character of text) {
    outputText.textContent += character;
    await sleep(TYPE_INTERVAL_MS);
  }
}

async function loadModel() {
  const [metadataResponse, vocabResponse, weightsResponse] = await Promise.all([
    fetch("./model.json").then((response) => assertResponse(response, "model.json")),
    fetch("./vocab.json").then((response) => assertResponse(response, "vocab.json")),
    fetch("./weights.bin").then((response) => assertResponse(response, "weights.bin")),
  ]);
  const [metadata, vocab, weights] = await Promise.all([
    metadataResponse.json(),
    vocabResponse.json(),
    weightsResponse.arrayBuffer(),
  ]);
  tokenizer = new CharTokenizer(vocab);
  model = new BrowserGPT(metadata, weights);
  modelStats.textContent = `${formatParameterCount(metadata.parameter_count)} parameters · ${metadata.block_size} token context`;
}

function showWritingMode() {
  if (!currentPrompt) return;
  writingPrompt.textContent = currentPrompt;
  writingInput.value = "";
  writingMode.hidden = false;
  document.body.style.overflow = "hidden";
  requestAnimationFrame(() => writingInput.focus());
}

function hideWritingMode() {
  writingMode.hidden = true;
  document.body.style.overflow = "";
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatWriting(value) {
  const safe = escapeHtml(value.trim());
  if (!safe) return "<p class=\"empty\">No writing yet.</p>";
  return safe.split(/\n{2,}/).map((paragraph) => `<p>${paragraph.replace(/\n/g, "<br />")}</p>`).join("");
}

function saveWriting() {
  const date = new Date();
  const stamp = date.toISOString().slice(0, 10);
  const safePrompt = escapeHtml(currentPrompt);
  const documentHtml = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${safePrompt}</title>
  <style>
    @page { size: Letter; margin: 0.75in; }
    :root { color: #202723; background: #f5f0e8; }
    body { margin: 0; background: #f5f0e8; color: #202723; font-family: Georgia, serif; }
    article { max-width: 680px; margin: 72px auto; padding: 0 28px; }
    .eyebrow { margin: 0 0 26px; color: #c35e42; font: 11px/1.4 monospace; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 0; color: #c35e42; font-size: 36px; font-weight: 500; letter-spacing: -.035em; line-height: 1.08; }
    .rule { height: 1px; margin: 40px 0 34px; background: #202723; opacity: .22; }
    .work { font-size: 19px; line-height: 1.75; }
    .work p { margin: 0 0 1.2em; }
    .empty { color: #77736a; font-style: italic; }
    .credit { margin-top: 72px; color: #77736a; font: 10px/1.5 monospace; letter-spacing: .08em; text-transform: uppercase; }
  </style>
</head>
<body>
  <article>
    <p class="eyebrow">Tiny AI prompt-writer</p>
    <h1>${safePrompt}</h1>
    <div class="rule"></div>
    <div class="work">${formatWriting(writingInput.value)}</div>
    <p class="credit">Written ${stamp}</p>
  </article>
</body>
</html>`;
  const blob = new Blob([documentHtml], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `tiny-ai-prompt-${stamp}.html`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function handleGenerate() {
  if (!model || isBusy) return;
  isBusy = true;
  generateButton.disabled = true;
  writeButton.hidden = true;
  generateButton.classList.add("is-working");
  buttonLabel.textContent = "Writing";
  outputText.textContent = "";
  output.dataset.state = "typing";
  currentPrompt = "";
  setStatus("GENERATING", "working");

  try {
    // Match the training sampler: a special BOS token followed by the literal
    // <bos> marker seen in the training text. Stop at the literal <eos> marker.
    const literalBos = tokenizer.encode("<bos>");
    const literalEos = tokenizer.encode("<eos>");
    const ids = await model.generate({
      inputIds: [tokenizer.bosId, ...literalBos],
      bosId: tokenizer.bosId,
      eosId: tokenizer.eosId,
      stopSequenceIds: literalEos,
      maxNewTokens: 194,
      temperature: 0.8,
    });
    const text = tokenizer.decode(ids);
    const stopAt = text.indexOf("<eos>");
    currentPrompt = (stopAt >= 0 ? text.slice(0, stopAt) : text).trim();
    await typeOut(currentPrompt || "The model reached <eos> before writing a prompt.");
    output.dataset.state = "done";
    writeButton.hidden = !currentPrompt;
    setStatus("MODEL READY");
  } catch (error) {
    console.error(error);
    await typeOut("The local model could not generate this time. Check the browser console for details.");
    output.dataset.state = "error";
    setStatus("MODEL ERROR", "error");
  } finally {
    isBusy = false;
    generateButton.disabled = false;
    generateButton.classList.remove("is-working");
    buttonLabel.textContent = "Generate";
  }
}

generateButton.addEventListener("click", handleGenerate);
writeButton.addEventListener("click", showWritingMode);
exitWritingButton.addEventListener("click", hideWritingMode);
saveWritingButton.addEventListener("click", saveWriting);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !writingMode.hidden) hideWritingMode();
});

loadModel()
  .then(() => {
    setModelState("ready");
    generateButton.disabled = false;
    setStatus("MODEL READY");
  })
  .catch((error) => {
    console.error(error);
    setModelState("error");
    setStatus("MODEL UNAVAILABLE", "error");
    modelStats.textContent = "Model unavailable";
    outputPlaceholder.textContent = "Start a local static server to load the model.";
  });
