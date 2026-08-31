/* Sparkstation Console — Voice Studio (Talk + Voices). Same-origin API; no build step. */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const LANGS = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian", "Portuguese", "Russian", "Auto"];
  const ENGINE_LABEL = { clone: "Cloned (VoiceClone)", stock: "Stock (CustomVoice)", design: "Designed (VoiceDesign)" };
  const ENGINE_ORDER = ["clone", "design", "stock"];

  let config = { auth_required: false, grafana_url: null };
  let voices = [];
  let status = null;
  let session = null;
  let statusTimer = null;

  // ── api ──────────────────────────────────────────────────────────────────
  function apiKey() { try { return localStorage.getItem("sparkstation.apiKey") || ""; } catch (e) { return ""; } }
  function headers(extra) {
    const h = Object.assign({}, extra || {});
    const k = apiKey(); if (k) h["X-API-Key"] = k;
    return h;
  }
  async function api(method, path, body, opts) {
    const init = { method, headers: headers() };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
    const r = await fetch(path, init);
    if (opts && opts.raw) return r;
    let data = null;
    const ct = r.headers.get("content-type") || "";
    if (ct.includes("application/json")) data = await r.json();
    if (!r.ok) {
      const msg = (data && (data.detail || data.error)) || r.statusText || ("HTTP " + r.status);
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function toast(msg, bad, warn) {
    const t = $("#toast"); t.textContent = msg; t.hidden = false;
    t.classList.toggle("bad", !!bad); t.classList.toggle("warn", !bad && !!warn);
    clearTimeout(t._h); t._h = setTimeout(() => { t.hidden = true; }, bad || warn ? 7000 : 3000);
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function fillLangs() { $$(".lang-select").forEach((sel) => { sel.innerHTML = LANGS.map((l) => `<option>${l}</option>`).join(""); }); }

  // ── navigation ───────────────────────────────────────────────────────────
  const BUILT_SECTIONS = { voice: "#section-voice", cluster: "#section-cluster", logs: "#section-logs" };
  function showSection(name) {
    $$(".nav-item[data-section]").forEach((a) => a.classList.toggle("active", a.dataset.section === name));
    for (const sel of Object.values(BUILT_SECTIONS)) $(sel).hidden = BUILT_SECTIONS[name] !== sel;
    $("#section-soon").hidden = !!BUILT_SECTIONS[name];
    if (!BUILT_SECTIONS[name]) $("#soon-title").textContent = ($(`.nav-item[data-section="${name}"]`) || {}).textContent || "Coming soon";
    if (name === "cluster") refreshCluster();
    if (name === "logs") refreshLogSources();
    clusterVisible = name === "cluster";
    logsVisible = name === "logs";
  }
  function showTab(name) {
    $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $("#tab-talk").hidden = name !== "talk";
    $("#tab-voices").hidden = name !== "voices";
    location.hash = "voice/" + name;
  }
  function route() {
    const h = location.hash.replace(/^#/, "");
    const [section, tab] = h.split("/");
    if (!section || section === "voice") { showSection("voice"); showTab(tab === "voices" ? "voices" : "talk"); }
    else showSection(section);
  }

  // ── status ───────────────────────────────────────────────────────────────
  async function refreshStatus() {
    try {
      status = await api("GET", "/voice/status");
      const bot = status.bot.ok;
      const line = $("#stack-status");
      line.textContent = bot ? `voice stack up on ${status.role}` : `voice bot not running on ${status.role}`;
      line.className = "status-line " + (bot ? "ok" : "bad");
      renderEngineStatus();
      // keep polling faster while an engine restarts
      const busy = Object.values(status.engines).some((e) => e.apply && e.apply.state === "restarting");
      clearTimeout(statusTimer);
      statusTimer = setTimeout(refreshStatus, busy ? 3000 : 20000);
    } catch (e) {
      $("#stack-status").textContent = "voice API: " + e.message;
      $("#stack-status").className = "status-line bad";
      clearTimeout(statusTimer);
      statusTimer = setTimeout(refreshStatus, 20000);
    }
  }
  function renderEngineStatus() {
    if (!status) return;
    const el = $("#engine-status");
    el.innerHTML = ENGINE_ORDER.map((name) => {
      const e = status.engines[name]; if (!e) return "";
      let cls = e.ok ? "ok" : "bad", note = e.ok ? "ready" : "down";
      if (e.apply && e.apply.state === "restarting") { cls = "busy"; note = "restarting (~45 s)…"; }
      else if (e.apply && e.apply.state === "error") { cls = "bad"; note = "restart failed: " + e.apply.error; }
      return `<span class="${cls}" title="port ${e.port}">${esc(e.label)} — ${esc(note)}</span>`;
    }).join("") + (status.default ? `<span title="what Sparky speaks in for new sessions">★ default: ${esc(status.default.voice)} (${esc(status.default.engine)})</span>` : `<span>no default set — bot falls back to its CASCADE_VOICE env</span>`);
  }

  // ── voices ───────────────────────────────────────────────────────────────
  async function loadVoices() {
    try {
      voices = await api("GET", "/voice/voices");
      renderVoices();
      renderTalkVoiceSelect();
    } catch (e) { toast("cannot load voices: " + e.message, true); }
  }
  function sampleText() { return $("#sample-text").value.trim(); }

  function renderVoices() {
    const list = $("#voices-list");
    list.innerHTML = "";
    for (const engine of ENGINE_ORDER) {
      const vs = voices.filter((v) => v.engine === engine);
      const group = document.createElement("div");
      group.className = "group";
      group.innerHTML = `<h3>${esc(ENGINE_LABEL[engine])} · ${vs.length}</h3>`;
      if (!vs.length) { group.innerHTML += `<p class="muted">none yet</p>`; list.appendChild(group); continue; }
      const table = document.createElement("table");
      table.className = "voices";
      table.innerHTML = `<thead><tr><th>Voice</th><th>Language</th><th>${engine === "design" ? "Description (identity)" : "Style instruct"}</th>${engine === "clone" ? "<th>Reference transcript</th>" : ""}<th></th></tr></thead>`;
      const tbody = document.createElement("tbody");
      for (const v of vs) tbody.appendChild(voiceRow(v));
      table.appendChild(tbody);
      group.appendChild(table);
      list.appendChild(group);
    }
  }

  function voiceRow(v) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${esc(v.id)}</strong>${v.is_default ? '<span class="default-badge">default</span>' : ""}${v.speaker && v.speaker !== v.id ? `<div class="muted">speaker ${esc(v.speaker)}</div>` : ""}</td>
      <td>${esc(v.language)}</td>
      <td class="instruct-cell"><div class="instruct-view">${v.instruct ? esc(v.instruct) : '<span class="muted">—</span>'}</div></td>
      ${v.engine === "clone" ? `<td class="instruct-cell">${esc(v.ref_text || "")}</td>` : ""}
      <td class="actions">
        <button data-act="play" title="synthesize the sample text with this voice">▶</button>
        <button data-act="edit" title="edit ${v.engine === "design" ? "description" : "style instruct"}">✎</button>
        ${v.is_default ? "" : `<button data-act="default" title="make this Sparky's voice">★</button>`}
        ${v.engine === "stock" || v.is_default ? "" : `<button data-act="delete" class="danger" title="delete">🗑</button>`}
      </td>`;
    tr.querySelector('[data-act="play"]').onclick = (ev) => playVoice(v, ev.currentTarget);
    tr.querySelector('[data-act="edit"]').onclick = () => editVoice(v, tr);
    const d = tr.querySelector('[data-act="default"]'); if (d) d.onclick = () => setDefault(v);
    const x = tr.querySelector('[data-act="delete"]'); if (x) x.onclick = () => deleteVoice(v);
    return tr;
  }

  let currentAudio = null;
  async function playVoice(v, btn) {
    try {
      btn.disabled = true; btn.textContent = "…";
      const body = { voice: v.id, engine: v.engine };
      if (sampleText()) body.text = sampleText();
      const r = await api("POST", "/voice/speak", body, { raw: true });
      if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch (e) {} throw new Error(d || r.statusText); }
      const blob = await r.blob();
      if (currentAudio) { currentAudio.pause(); URL.revokeObjectURL(currentAudio.src); }
      currentAudio = new Audio(URL.createObjectURL(blob));
      await currentAudio.play();
    } catch (e) { toast("play failed: " + e.message, true); }
    finally { btn.disabled = false; btn.textContent = "▶"; }
  }

  function editVoice(v, tr) {
    const cell = tr.querySelector(".instruct-cell");
    if (cell.querySelector("textarea")) return;
    const original = cell.innerHTML;
    cell.innerHTML = `<textarea rows="3">${esc(v.instruct)}</textarea>
      <div class="row"><button class="primary" data-save>Save</button><button class="link-btn" data-cancel>cancel</button>
      ${v.engine !== "design" ? '<span class="muted">clone/stock edits restart that engine (~45 s)</span>' : ""}</div>`;
    cell.querySelector("[data-cancel]").onclick = () => { cell.innerHTML = original; };
    cell.querySelector("[data-save]").onclick = async () => {
      try {
        const res = await api("PATCH", `/voice/voices/${v.engine}/${encodeURIComponent(v.id)}`, { instruct: cell.querySelector("textarea").value });
        toast(res.applying ? "saved — engine restarting" : "saved");
        await loadVoices(); refreshStatus();
      } catch (e) { toast("save failed: " + e.message, true); }
    };
  }

  async function setDefault(v) {
    try {
      await api("POST", `/voice/voices/${v.engine}/${encodeURIComponent(v.id)}/default`);
      toast(`${v.id} is now Sparky's default voice (new sessions)`);
      await loadVoices(); refreshStatus();
    } catch (e) { toast("set default failed: " + e.message, true); }
  }

  async function deleteVoice(v) {
    if (!confirm(`Delete ${v.engine} voice "${v.id}"?${v.engine === "clone" ? " Its reference clip is removed from the registry too." : ""}`)) return;
    try {
      const res = await api("DELETE", `/voice/voices/${v.engine}/${encodeURIComponent(v.id)}`);
      toast(res.applying ? "deleted — engine restarting" : "deleted");
      await loadVoices(); refreshStatus();
    } catch (e) { toast("delete failed: " + e.message, true); }
  }

  // ── design form ──────────────────────────────────────────────────────────
  function bindDesignForm() {
    const form = $("#design-form"), msg = $("#design-msg");
    $("#show-design").onclick = () => { form.hidden = false; $("#clone-form").hidden = true; $("#design-id").focus(); };
    $("#design-cancel").onclick = () => { form.hidden = true; };
    $("#design-preview").onclick = async () => {
      const instruct = $("#design-instruct").value.trim();
      if (instruct.length < 3) { msg.textContent = "write a description first"; msg.className = "status-line warn"; return; }
      msg.textContent = "synthesizing preview…"; msg.className = "status-line";
      try {
        const body = { engine: "design", instruct, language: $("#design-language").value };
        const t = $("#design-preview-text").value.trim() || sampleText(); if (t) body.text = t;
        const r = await api("POST", "/voice/speak", body, { raw: true });
        if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch (e) {} throw new Error(d || r.statusText); }
        const a = $("#design-audio"); a.src = URL.createObjectURL(await r.blob()); a.hidden = false; await a.play();
        msg.textContent = "preview ready (every take differs a little — designed voices are re-rolled from the description)";
      } catch (e) { msg.textContent = "preview failed: " + e.message; msg.className = "status-line bad"; }
    };
    $("#design-save").onclick = async () => {
      try {
        const res = await api("POST", "/voice/voices/design", { id: $("#design-id").value.trim(), instruct: $("#design-instruct").value.trim(), language: $("#design-language").value });
        toast(`designed voice ${res.voice} saved`); form.hidden = true;
        $("#design-id").value = ""; $("#design-instruct").value = "";
        await loadVoices(); refreshStatus();
      } catch (e) { msg.textContent = "save failed: " + e.message; msg.className = "status-line bad"; }
    };
  }

  // ── clone form (record in-browser or upload) ─────────────────────────────
  let recorder = null, recordedBlob = null, recordTimer = null;
  function bindCloneForm() {
    const form = $("#clone-form"), msg = $("#clone-msg"), state = $("#clone-record-state");
    $("#show-clone").onclick = () => { form.hidden = false; $("#design-form").hidden = true; $("#clone-id").focus(); };
    $("#clone-cancel").onclick = () => { form.hidden = true; stopRecording(); };
    $("#clone-file").onchange = () => { recordedBlob = null; const f = $("#clone-file").files[0]; if (f) { const a = $("#clone-audio"); a.src = URL.createObjectURL(f); a.hidden = false; } };

    function stopRecording() { if (recorder && recorder.state !== "inactive") recorder.stop(); clearInterval(recordTimer); }
    $("#clone-record").onclick = async () => {
      if (recorder && recorder.state === "recording") { stopRecording(); return; }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
        const chunks = [];
        recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
        recorder.onstop = () => {
          stream.getTracks().forEach((t) => t.stop());
          recordedBlob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
          $("#clone-file").value = "";
          const a = $("#clone-audio"); a.src = URL.createObjectURL(recordedBlob); a.hidden = false;
          $("#clone-record").textContent = "● Record again";
          state.textContent = `recorded ${elapsed.toFixed(1)} s`;
        };
        let elapsed = 0; const t0 = performance.now();
        recorder.start(250);
        $("#clone-record").textContent = "■ Stop";
        recordTimer = setInterval(() => {
          elapsed = (performance.now() - t0) / 1000;
          state.textContent = `recording ${elapsed.toFixed(1)} s` + (elapsed > 12 ? " — that's plenty, stop soon" : "");
          if (elapsed >= 19.5) stopRecording();
        }, 100);
      } catch (e) { msg.textContent = "microphone: " + e.message; msg.className = "status-line bad"; }
    };

    $("#clone-save").onclick = async () => {
      const file = recordedBlob || $("#clone-file").files[0];
      if (!file) { msg.textContent = "record or choose a clip first"; msg.className = "status-line warn"; return; }
      const fd = new FormData();
      fd.append("id", $("#clone-id").value.trim());
      fd.append("ref_text", $("#clone-text").value.trim());
      fd.append("language", $("#clone-language").value);
      fd.append("instruct", $("#clone-instruct").value.trim());
      const ext = recordedBlob ? ((recordedBlob.type || "").includes("ogg") ? ".ogg" : ".webm") : "";
      fd.append("file", file, recordedBlob ? "recording" + ext : file.name);
      msg.textContent = "uploading + registering (the clone engine restarts, ~45 s)…"; msg.className = "status-line";
      $("#clone-save").disabled = true;
      try {
        const res = await api("POST", "/voice/voices/clone", fd);
        toast(`✓ clone voice ${res.voice} registered (${res.duration_seconds}s clip)` + (res.warning ? " — note: " + res.warning : ""), false, !!res.warning);
        form.hidden = true; recordedBlob = null; $("#clone-id").value = ""; $("#clone-text").value = ""; $("#clone-audio").hidden = true;
        await loadVoices(); refreshStatus();
      } catch (e) { msg.textContent = "failed: " + e.message; msg.className = "status-line bad"; }
      finally { $("#clone-save").disabled = false; }
    };
  }

  // ── talk ─────────────────────────────────────────────────────────────────
  function renderTalkVoiceSelect() {
    const sel = $("#talk-voice");
    const prev = sel.value;
    sel.innerHTML = `<option value="">(bot default)</option>` + voices.map((v) => `<option value="${esc(v.engine)}:${esc(v.id)}">${esc(v.id)} — ${esc(v.engine)}${v.is_default ? " ★" : ""}</option>`).join("");
    if (prev && Array.from(sel.options).some((o) => o.value === prev)) sel.value = prev;
  }
  function addMsg(cls, text, partialKey) {
    const box = $("#transcript");
    if (partialKey) {
      let el = box.querySelector(`[data-partial="${partialKey}"]`);
      if (!el) { el = document.createElement("div"); el.dataset.partial = partialKey; box.appendChild(el); }
      el.className = "msg " + cls; el.textContent = text;
      box.scrollTop = box.scrollHeight; return el;
    }
    const el = document.createElement("div"); el.className = "msg " + cls; el.textContent = text;
    box.appendChild(el); box.scrollTop = box.scrollHeight; return el;
  }
  function finalizePartial(key) { const el = $("#transcript").querySelector(`[data-partial="${key}"]`); if (el) { el.removeAttribute("data-partial"); el.classList.remove("partial"); } }

  function bindTalk() {
    const connect = $("#talk-connect"), stop = $("#talk-stop"), state = $("#talk-state");
    const setState = (s) => { state.textContent = s; state.className = "status-line " + (/live/.test(s) ? "ok" : /busy|fail|unreach|closed \(1011/.test(s) ? "bad" : ""); };
    $("#talk-clear").onclick = () => { $("#transcript").innerHTML = ""; };
    connect.onclick = async () => {
      if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) { setState("no microphone API — open the console on localhost or over HTTPS"); return; }
      const [engine, voice] = ($("#talk-voice").value || ":").split(":");
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      connect.hidden = true; stop.hidden = false;
      let botSaid = "";
      session = new SparkTalk.TalkSession({
        url: `${proto}//${location.host}/voice/talk`,
        voice: voice || null, engine: engine || null,
        brain: $("#talk-brain").value,
        systemInstruction: $("#talk-sysprompt").value.trim(),
        onState: setState,
        onLevel: (peak) => { $("#mic-level").style.width = Math.min(100, Math.round(peak * 140)) + "%"; },
        onStats: (s) => {
          $("#talk-stats").innerHTML = [
            `turns ${s.turns}`, `bot audio ${(s.botAudioBytes / 48000).toFixed(1)} s`,
            s.ttfbMs != null ? `TTS ttfb ${s.ttfbMs} ms` : null,
          ].filter(Boolean).map((x) => `<span>${esc(x)}</span>`).join("");
        },
        onMessage: (kind, data) => {
          if (kind !== "message" || !data) return;
          const t = data.type, d = data.data || {};
          if (t === "user-transcription") {
            if (d.final) { addMsg("user", d.text, "user"); finalizePartial("user"); }
            else addMsg("user partial", d.text, "user");
          } else if (t === "bot-transcription" || t === "bot-tts-text") {
            if (t === "bot-tts-text" && d.text) { botSaid += (botSaid ? " " : "") + d.text; addMsg("bot partial", botSaid, "bot"); }
            if (t === "bot-transcription" && d.text) { addMsg("bot", d.text, "bot"); }
          } else if (t === "bot-stopped-speaking") { finalizePartial("bot"); botSaid = ""; }
          else if (t === "bot-started-speaking") { botSaid = ""; }
          else if (t === "configured") addMsg("sys", "session configured: " + JSON.stringify(d));
          else if (t === "bot-ready") addMsg("sys", "bot ready");
          else if (t === "error") addMsg("sys", "error: " + (d.message || JSON.stringify(d)));
          else if (t === "llm-function-call") addMsg("sys", `brain wanted tool ${d.function_name} — console has no tools`);
        },
        onClosed: () => { connect.hidden = false; stop.hidden = true; $("#mic-level").style.width = "0"; session = null; },
      });
      try { await session.start(); }
      catch (e) { setState("failed: " + e.message); session.stop(); }
    };
    stop.onclick = () => { if (session) session.stop(); };
  }

  // ── cluster & models ─────────────────────────────────────────────────────
  let clusterVisible = false, clusterTimer = null, profilesInfo = null;

  async function refreshCluster() {
    clearTimeout(clusterTimer);
    try {
      const [res, det] = await Promise.all([api("GET", "/resources"), api("GET", "/models/detailed")]);
      renderResources(res);
      renderModels(det.models || []);
      if (!profilesInfo) { profilesInfo = await api("GET", "/profiles"); renderStartControls(det.models || []); }
      else renderStartControls(det.models || []);
      $("#profile-chip").textContent = profilesInfo ? `profile: ${profilesInfo.active}` : "";
    } catch (e) { toast("cluster refresh failed: " + e.message, true); }
    if (clusterVisible) clusterTimer = setTimeout(refreshCluster, 10000);
  }

  function renderResources(r) {
    const usedPct = Math.round(100 * r.unified_memory_used_gb / r.unified_memory_gb);
    $("#resource-cards").innerHTML = `
      <div class="card"><div class="big">${r.unified_memory_used_gb.toFixed(1)} <span class="sub">/ ${r.unified_memory_gb.toFixed(0)} GB</span></div>
        <div class="sub">primary unified memory (limit ${r.unified_memory_limit_gb.toFixed(0)} GB)</div>
        <div class="membar${usedPct > 80 ? " hot" : ""}"><div style="width:${usedPct}%"></div></div></div>
      <div class="card"><div class="big">${r.gpu_temperature_c.toFixed(0)}°C</div><div class="sub">GPU temp · ${r.gpu_power_draw_w.toFixed(0)} W draw</div></div>
      <div class="card"><div class="big">${r.resident_models_count} <span class="sub">/ ${r.max_resident_models}</span></div><div class="sub">resident models (all hosts)</div></div>`;
  }

  function fmtIdle(sec) {
    if (sec == null) return "—";
    if (sec < 90) return Math.round(sec) + "s";
    if (sec < 5400) return Math.round(sec / 60) + "m";
    return (sec / 3600).toFixed(1) + "h";
  }

  function renderModels(models) {
    const rows = models.map((m) => {
      const running = m.status === "running", suspended = m.status === "suspended";
      const hb = m.health_status === "healthy" ? "💚" : m.health_status === "unhealthy" ? "💔" : "";
      const acts = [
        running || m.status === "starting" ? `<button data-act="stop" title="stop">■</button>` : "",
        running && m.model_type === "chat" ? `<button data-act="suspend" title="suspend (free memory, fast resume)">⏸</button>` : "",
        suspended ? `<button data-act="resume" title="resume">▶</button>` : "",
      ].join("");
      return `<tr data-id="${esc(m.id)}" data-alias="${esc(m.alias || m.model_name)}">
        <td><strong>${esc(m.alias || m.model_name)}</strong>${m.is_default ? '<span class="default-badge">default</span>' : ""}${m.is_vision ? ' 👁' : ""}<div class="muted">${esc(m.backend)} · ${esc(m.model_type)}</div></td>
        <td>${esc(m.host)}</td>
        <td><span class="st ${esc(m.status)}">${esc(m.status)}</span><span class="hb">${hb}</span></td>
        <td>${m.port || "—"}</td>
        <td>${m.memory_gb != null ? m.memory_gb + " GB" : "—"}</td>
        <td>${fmtIdle(m.idle_seconds)}</td>
        <td class="actions">${acts}</td></tr>`;
    }).join("");
    $("#models-table").innerHTML = models.length
      ? `<table class="models"><thead><tr><th>Model</th><th>Host</th><th>Status</th><th>Port</th><th>Memory</th><th>Idle</th><th></th></tr></thead><tbody>${rows}</tbody></table>`
      : '<p class="muted">no models in the registry</p>';
    $$("#models-table [data-act]").forEach((b) => { b.onclick = () => modelAction(b.closest("tr"), b.dataset.act); });
  }

  async function modelAction(tr, act) {
    const id = tr.dataset.id, alias = tr.dataset.alias;
    const warn = { stop: `Stop ${alias}? Clients using it will fail until it's started again.`,
                   suspend: `Suspend ${alias}? First request after resume pays the reload.`,
                   resume: null }[act];
    if (warn && !confirm(warn)) return;
    try {
      await api("POST", `/models/${encodeURIComponent(id)}/${act}`);
      toast(`${act} requested for ${alias}`);
      refreshCluster();
    } catch (e) { toast(`${act} ${alias} failed: ` + e.message, true); }
  }

  function renderStartControls(models) {
    if (!profilesInfo) return;
    const live = new Set(models.filter((m) => ["running", "starting"].includes(m.status)).map((m) => m.alias));
    const aliasSel = $("#start-alias"), profSel = $("#start-profile");
    const prevA = aliasSel.value, prevP = profSel.value;
    const profile = prevP || profilesInfo.active;
    const inProfile = new Set(profilesInfo.profiles[profile] || []);
    const opt = (a) => {
      const info = (profilesInfo.aliases || {})[a] || {};
      const where = info.host ? ` — ${info.host}${info.memory_gb ? `, ${info.memory_gb} GB` : ""}` : "";
      return `<option value="${esc(a)}"${live.has(a) ? " disabled" : ""}>${esc(a)}${esc(where)}${live.has(a) ? " (live)" : ""}</option>`;
    };
    const rest = profilesInfo.all_aliases.filter((a) => !inProfile.has(a));
    aliasSel.innerHTML =
      `<optgroup label="in profile ${esc(profile)}">${[...inProfile].sort().map(opt).join("")}</optgroup>` +
      (rest.length ? `<optgroup label="all specs (on-demand)">${rest.map(opt).join("")}</optgroup>` : "");
    profSel.innerHTML = Object.keys(profilesInfo.profiles).map((p) => `<option value="${esc(p)}"${p === profile ? " selected" : ""}>profile: ${esc(p)}</option>`).join("");
    if (prevA && Array.from(aliasSel.options).some((o) => o.value === prevA)) aliasSel.value = prevA;
    profSel.onchange = () => renderStartControls(models);
  }

  function bindCluster() {
    $("#cluster-refresh").onclick = () => { profilesInfo = null; refreshCluster(); };
    $("#start-btn").onclick = async () => {
      const alias = $("#start-alias").value, profile = $("#start-profile").value;
      if (!alias) return;
      if (!confirm(`Start ${alias} (profile ${profile})? Large models can take minutes and lots of memory.`)) return;
      const msg = $("#start-msg");
      msg.textContent = `starting ${alias}…`; msg.className = "status-line";
      try {
        await api("POST", `/models/${encodeURIComponent(alias)}/start-by-alias?profile=${encodeURIComponent(profile)}`);
        msg.textContent = `${alias} launching — watch its status above`; msg.className = "status-line ok";
        refreshCluster();
      } catch (e) { msg.textContent = "start failed: " + e.message; msg.className = "status-line bad"; }
    };
  }

  // ── logs ─────────────────────────────────────────────────────────────────
  let logsVisible = false, logTimer = null;

  async function refreshLogSources() {
    try {
      const d = await api("GET", "/logs");
      const sel = $("#log-source"); const prev = sel.value;
      sel.innerHTML = d.sources.map((src) => `<option value="${esc(src.id)}">${esc(src.label)}${src.status ? ` (${esc(src.status)})` : ""}</option>`).join("");
      if (prev && Array.from(sel.options).some((o) => o.value === prev)) sel.value = prev;
    } catch (e) { toast("cannot list logs: " + e.message, true); }
    refreshLog();
  }

  async function refreshLog() {
    clearTimeout(logTimer);
    const src = $("#log-source").value;
    if (src) {
      try {
        const r = await api("GET", `/logs/${encodeURIComponent(src)}?lines=${$("#log-lines").value}`, undefined, { raw: true });
        if (!r.ok) throw new Error((await r.text()).slice(0, 200));
        const view = $("#log-view");
        view.textContent = await r.text();
        view.scrollTop = view.scrollHeight;
      } catch (e) { $("#log-view").textContent = "error: " + e.message; }
    }
    if (logsVisible && $("#log-follow").checked) logTimer = setTimeout(refreshLog, 4000);
  }

  function bindLogs() {
    $("#log-refresh").onclick = refreshLog;
    $("#log-source").onchange = refreshLog;
    $("#log-lines").onchange = refreshLog;
    $("#log-follow").onchange = refreshLog;
  }

  // ── api key (only when the supervisor enforces one) ──────────────────────
  function bindApiKey() {
    const btn = $("#apikey-btn");
    btn.onclick = () => {
      const k = prompt("Supervisor X-API-Key (stored in this browser only)", apiKey());
      if (k !== null) { try { localStorage.setItem("sparkstation.apiKey", k.trim()); } catch (e) {} toast("API key saved"); }
    };
  }

  // ── boot ─────────────────────────────────────────────────────────────────
  async function boot() {
    fillLangs();
    $$(".nav-item[data-section]").forEach((a) => { a.onclick = (ev) => { ev.preventDefault(); location.hash = a.dataset.section; }; });
    $$(".tab").forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });
    window.addEventListener("hashchange", route);
    route();
    bindDesignForm(); bindCloneForm(); bindTalk(); bindApiKey(); bindCluster(); bindLogs();
    $("#voices-refresh").onclick = () => { loadVoices(); refreshStatus(); };
    try {
      config = await (await fetch("/console/config.json")).json();
      if (config.grafana_url) { const g = $("#grafana-link"); g.href = config.grafana_url; g.hidden = false; }
      $("#apikey-btn").hidden = !config.auth_required;
    } catch (e) { /* static-only fallback */ }
    refreshStatus();
    loadVoices();
  }
  boot();
})();
