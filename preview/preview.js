const PATH_FRONT = "../anki/note-type/front.html";
const PATH_BACK  = "../anki/note-type/back.html";

const AUDIO_BASE = "../media/audio/";
const IMAGE_BASE = "../media/images/";

async function loadText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path} (${res.status})`);
  return await res.text();
}

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path} (${res.status})`);
  return await res.json();
}

function stripTTS(tpl) {
  // Remove Anki TTS tags for browser preview
  return tpl.replace(/{{\s*tts[^}]*}}/g, "<!-- tts omitted in preview -->");
}

function renderAnkiConditionals(tpl, data) {
  // {{#field}}...{{/field}} (show if field is truthy/non-empty)
  tpl = tpl.replace(/{{#([^}]+)}}([\s\S]*?){{\/\1}}/g, (_, key, inner) => {
    const v = (data[key] ?? "").toString().trim();
    return v ? inner : "";
  });

  // {{^field}}...{{/field}} (show if field is falsy/empty)
  tpl = tpl.replace(/{{\^([^}]+)}}([\s\S]*?){{\/\1}}/g, (_, key, inner) => {
    const v = (data[key] ?? "").toString().trim();
    return v ? "" : inner;
  });

  return tpl;
}

function renderAnkiFields(tpl, data) {
  return tpl.replace(/{{([^}]+)}}/g, (_, keyRaw) => {
    const key = keyRaw.trim();
    const v = data[key] ?? "";
    return v.toString();
  });
}

function soundTagToAudio(html) {
  // Replace occurrences of [sound:FILE.mp3] with <audio controls src="...">
  return html.replace(/\[sound:([^\]]+)\]/g, (_, filename) => {
    const safe = filename.trim();
    const src = `${AUDIO_BASE}${encodeURIComponent(safe)}`;
    return `
      <audio controls preload="none" style="max-width: 100%;">
        <source src="${src}">
        Your browser cannot play this audio.
      </audio>
    `;
  });
}

function rewriteImageSrcs(html) {
  // If your "image" field contains <img src="x.png">, rewrite to ../media/images/x.png
  // Skip if src is already absolute (http/https/data) or already starts with ../ or /
  return html.replace(/<img([^>]*?)\ssrc\s*=\s*["']([^"']+)["']([^>]*?)>/gi, (m, pre, src, post) => {
    const s = src.trim();
    const isAbsolute =
      s.startsWith("http://") ||
      s.startsWith("https://") ||
      s.startsWith("data:") ||
      s.startsWith("/") ||
      s.startsWith("../");

    const newSrc = isAbsolute ? s : `${IMAGE_BASE}${s}`;
    return `<img${pre} src="${newSrc}"${post}>`;
  });
}

function renderTemplate(tpl, data) {
  let out = tpl;
  out = stripTTS(out);
  out = renderAnkiConditionals(out, data);
  out = renderAnkiFields(out, data);
  out = soundTagToAudio(out);
  out = rewriteImageSrcs(out);
  return out;
}

function setError(msg) {
  document.getElementById("error").textContent = msg || "";
}

function setNightMode(enabled) {
  // Your CSS uses: .nightMode .card {...}
  // So toggle a .nightMode class on <body>
  document.body.classList.toggle("nightMode", enabled);
}

async function main() {
  const noteSelect = document.getElementById("noteSelect");
  const frontHost = document.getElementById("frontHost");
  const backHost = document.getElementById("backHost");
  const reloadBtn = document.getElementById("reloadBtn");
  const nightMode = document.getElementById("nightMode");

  let notes = await loadJson("./mock.json");
  if (!Array.isArray(notes) || notes.length === 0) {
    throw new Error("preview/mock.json must be a non-empty array of note objects.");
  }

  // Fill dropdown
  noteSelect.innerHTML = "";
  notes.forEach((n, idx) => {
    const label = `${n.id ?? `note-${idx+1}`} — ${n.hanzi ?? ""} ${n.pinyin ?? ""}`.trim();
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = label;
    noteSelect.appendChild(opt);
  });

  let frontTpl = await loadText(PATH_FRONT);
  let backTpl = await loadText(PATH_BACK);

  function renderSelected() {
    setError("");
    const idx = Number(noteSelect.value || "0");
    const note = notes[idx] ?? notes[0];

    // Make sure Tags exists (your template uses {{Tags}})
    const data = { ...note, Tags: (note.Tags ?? note.tags ?? "").toString() };

    frontHost.innerHTML = renderTemplate(frontTpl, data);
    backHost.innerHTML = renderTemplate(backTpl, data);
  }

  noteSelect.addEventListener("change", renderSelected);

  reloadBtn.addEventListener("click", async () => {
    try {
      setError("");
      frontTpl = await loadText(PATH_FRONT);
      backTpl = await loadText(PATH_BACK);
      renderSelected();
    } catch (e) {
      setError(e?.stack || String(e));
    }
  });

  nightMode.addEventListener("change", () => {
    setNightMode(nightMode.checked);
  });

  // Initial render
  renderSelected();
}

main().catch(err => {
  setError(err?.stack || String(err));
});
