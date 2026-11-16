"use strict";

// The page keeps no rules of its own: the server sends the position and the
// list of legal moves, the page draws them and sends back the chosen move.

const SVG = "http://www.w3.org/2000/svg";
// One square is U units wide. Sub-pixel units (1 per square) make browsers draw
// CSS-transformed pieces with a smeared backdrop, so the board uses round numbers.
const U = 100;
const board = document.getElementById("board");
const $ = (id) => document.getElementById(id);

let state = null;       // the last state received from the server
let pieces = [];        // what is drawn right now: {row, col, kind, el}
let selected = null;    // "row,col" of the selected piece
let busy = false;       // waiting for the server
let flipped = false;    // Black at the bottom

const REASONS = {
  no_moves: "przeciwnik nie ma ruchu",
  resignation: "poddanie partii",
  repetition: "trzykrotne powtórzenie pozycji",
  no_progress: "zbyt wiele ruchów damkami bez bicia",
  length: "limit długości partii",
};
const COLOUR_PL = { white: "białe", black: "czarne" };

// --- talking to the server ----------------------------------------------------------

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

const wantAnalysis = () => $("show-analysis").checked;

// --- drawing ------------------------------------------------------------------------

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.appendChild(node);
  return node;
}

// board coordinates (row 0 = White's side) to SVG coordinates
function xy(row, col) {
  const n = state.size;
  return flipped ? [(n - 1 - col) * U, row * U] : [col * U, (n - 1 - row) * U];
}
const centre = (row, col) => xy(row, col).map((v) => v + U / 2);
const key = (row, col) => `${row},${col}`;

function drawBoard() {
  const n = state.size;
  board.setAttribute("viewBox", `0 0 ${n * U} ${n * U}`);
  board.replaceChildren();
  const squares = el("g", {}, board);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const [x, y] = xy(row, col);
      const dark = (row + col) % 2 === 0;
      const rect = el("rect", { x, y, width: U, height: U, class: dark ? "sq-dark" : "sq-light" }, squares);
      if (dark) rect.addEventListener("click", () => onSquare(row, col));
    }
  }
  // file letters along the bottom, rank numbers on the left
  for (let i = 0; i < n; i++) {
    const [fx] = xy(0, i);
    el("text", { x: fx + 0.86 * U, y: (n - 0.06) * U, class: "coord" }, squares).textContent = "abcdefghij"[i];
    const [, ry] = xy(i, 0);
    el("text", { x: 0.04 * U, y: ry + 0.2 * U, class: "coord" }, squares).textContent = i + 1;
  }
  el("g", { id: "marks" }, board);
  el("g", { id: "pieces" }, board);
  el("g", { id: "overlay" }, board);
}

function pieceNode(piece) {
  const g = el("g", { class: `piece ${piece.kind}` });
  el("circle", { class: "body", r: 0.38 * U }, g);
  el("circle", { class: "rim", r: 0.27 * U }, g);
  if (piece.kind === "W" || piece.kind === "B") {
    el("path", { class: "crown", d: "M-17 10 L-20 -10 L-9 0 L0 -14 L9 0 L20 -10 L17 10 Z" }, g);
  }
  g.addEventListener("click", () => onSquare(piece.row, piece.col));
  place(piece, g);
  return g;
}

function place(piece, node = piece.el) {
  const [cx, cy] = centre(piece.row, piece.col);
  node.style.transform = `translate(${cx}px, ${cy}px)`;
}

function drawPieces(list) {
  const layer = $("pieces");
  layer.replaceChildren();
  pieces = list.map((p) => ({ ...p }));
  for (const piece of pieces) {
    piece.el = pieceNode(piece);
    layer.appendChild(piece.el);
  }
}

function drawMarks() {
  const layer = $("marks");
  layer.replaceChildren();
  const last = state.last_move;
  if (last) {
    for (const [row, col] of [last.path[0], last.path[last.path.length - 1]]) {
      const [x, y] = xy(row, col);
      el("rect", { x, y, width: U, height: U, class: "sq-last" }, layer);
    }
  }
  if (busy || state.result) return;
  const starts = new Set(state.legal.map((m) => key(...m.path[0])));
  if (!selected) {
    // show which pieces can move (captures are mandatory, so often only a few)
    for (const start of starts) {
      const [row, col] = start.split(",").map(Number);
      const [cx, cy] = centre(row, col);
      el("circle", { cx, cy, r: 0.44 * U, class: "movable" }, layer);
    }
    return;
  }
  const [srow, scol] = selected.split(",").map(Number);
  const [sx, sy] = xy(srow, scol);
  el("rect", { x: sx + 0.035 * U, y: sy + 0.035 * U, width: 0.93 * U, height: 0.93 * U, class: "sq-select" }, layer);
  for (const move of state.legal.filter((m) => key(...m.path[0]) === selected)) {
    const [row, col] = move.path[move.path.length - 1];
    const [cx, cy] = centre(row, col);
    el("circle", { cx, cy, r: 0.13 * U, class: "target" }, layer);
  }
}

function drawArrow(move) {
  const layer = $("overlay");
  if (!layer) return; // the board has not been drawn yet
  layer.replaceChildren();
  if (!move) return;
  const points = move.path.map(([row, col]) => centre(row, col).join(",")).join(" ");
  el("polyline", { points, class: "arrow" }, layer);
  const [ex, ey] = centre(...move.path[move.path.length - 1]);
  el("circle", { cx: ex, cy: ey, r: 0.13 * U, class: "arrow-dot" }, layer);
  for (const [row, col] of move.captures) {
    const [cx, cy] = centre(row, col);
    const d = 0.2 * U;
    el("line", { x1: cx - d, y1: cy - d, x2: cx + d, y2: cy + d, class: "arrow-x" }, layer);
    el("line", { x1: cx - d, y1: cy + d, x2: cx + d, y2: cy - d, class: "arrow-x" }, layer);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Move a piece along its path on the board that is currently drawn.
async function animate(move) {
  const [row0, col0] = move.path[0];
  const piece = pieces.find((p) => p.row === row0 && p.col === col0);
  if (!piece) return;
  $("pieces").appendChild(piece.el); // draw it above the others
  for (const [row, col] of move.path.slice(1)) {
    piece.row = row;
    piece.col = col;
    place(piece);
    await sleep(190);
  }
  for (const [row, col] of move.captures) {
    const taken = pieces.find((p) => p.row === row && p.col === col && p !== piece);
    if (taken) taken.el.classList.add("gone");
  }
  if (move.captures.length) await sleep(200);
  pieces = pieces.filter((p) => !move.captures.some(([r, c]) => p.row === r && p.col === c && p !== piece));
}

// --- the panels ---------------------------------------------------------------------

const percent = (x) => `${Math.round(100 * x)}%`;

function renderSeats() {
  const bottom = flipped ? "black" : "white";
  const top = flipped ? "white" : "black";
  for (const [id, colour] of [["seat-top", top], ["seat-bottom", bottom]]) {
    const player = state.players[colour];
    const seat = $(id);
    seat.replaceChildren();
    const dot = document.createElement("span");
    dot.className = `dot ${colour}`;
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = player.name;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = `${COLOUR_PL[colour]}${player.ai ? ` · ${state.simulations} symulacji na ruch` : ""}`;
    seat.append(dot, name, tag);
    seat.classList.toggle("active", !state.result && state.turn === colour);
  }
}

function renderStatus() {
  const status = $("status");
  status.replaceChildren();
  if (state.result) {
    const { winner, reason } = state.result;
    status.append(winner === "draw" ? "Remis" : `Wygrywają ${COLOUR_PL[winner]}: ${state.players[winner].name}`);
    const why = document.createElement("span");
    why.className = "why";
    why.textContent = REASONS[reason] || reason;
    status.append(why);
  } else if (busy) {
    status.append("AI liczy…");
  } else {
    const mover = state.players[state.turn];
    status.append(`Ruch: ${mover.name} (${COLOUR_PL[state.turn]})`);
    if (state.legal.length && state.legal[0].captures.length) {
      const why = document.createElement("span");
      why.className = "why";
      why.textContent = "Bicie jest obowiązkowe.";
      status.append(why);
    }
  }
  $("resign").disabled = Boolean(state.result) || busy || state.ply === 0;
}

function renderAnalysis() {
  $("analysis-body").hidden = !wantAnalysis();
  const tbody = $("moves-table").querySelector("tbody");
  tbody.replaceChildren();
  const analysis = state.analysis || (state.result ? state.ai_move : null);
  const shown = state.analysis;
  if (!analysis) {
    $("eval-text").textContent = state.result ? "Partia zakończona." : "—";
    return;
  }
  const white = (analysis.value_white + 1) / 2;
  $("evalbar-white").style.width = percent(white);
  const network = (analysis.network_value_white + 1) / 2;
  $("eval-text").textContent =
    `Białe ${percent(white)} · czarne ${percent(1 - white)} ` +
    `(sama sieć, bez przeszukiwania: białe ${percent(network)})`;
  if (!shown) return;
  const moves = [...shown.moves].sort((a, b) => b.visits - a.visits || b.prior - a.prior);
  for (const move of moves) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.className = "move";
    name.textContent = move.name;
    const prior = document.createElement("td");
    prior.textContent = percent(move.prior);
    const share = document.createElement("td");
    share.innerHTML = '<div class="share"><span></span><div class="track"><div class="fill"></div></div></div>';
    share.querySelector("span").textContent = percent(move.share);
    share.querySelector(".fill").style.width = percent(move.share);
    const value = document.createElement("td");
    value.textContent = move.value === null ? "—" : percent((move.value + 1) / 2);
    row.append(name, prior, share, value);
    row.addEventListener("mouseenter", () => drawArrow(move));
    row.addEventListener("mouseleave", () => drawArrow(null));
    tbody.appendChild(row);
  }
}

function renderAiCard() {
  const ai = state.ai_move;
  $("ai-card").hidden = !ai;
  if (!ai) return;
  const chosen = ai.moves.find((m) => m.id === ai.chosen);
  const others = ai.moves.filter((m) => m.id !== ai.chosen).sort((a, b) => b.visits - a.visits).slice(0, 2);
  let text = `${ai.name}: ${percent(chosen.share)} symulacji`;
  if (ai.moves.length === 1) text = `${ai.name}: jedyny legalny ruch`;
  else if (others.length) {
    text += `. Rozważało też: ${others.map((m) => `${m.name} (${percent(m.share)})`).join(", ")}`;
  }
  const mine = ai.by === "white" ? ai.value_white : -ai.value_white;
  text += `. Swoje szanse oceniało na ${percent((mine + 1) / 2)}.`;
  $("ai-text").textContent = text;
}

function renderHistory() {
  const parts = [];
  state.history.forEach((name, index) => {
    parts.push(index % 2 === 0 ? `${index / 2 + 1}. ${name}` : name);
  });
  $("history").textContent = parts.join("  ") || "—";
}

function render() {
  drawMarks();
  renderSeats();
  renderStatus();
  renderAnalysis();
  renderAiCard();
  renderHistory();
}

async function renderResults() {
  const data = await api("/api/results");
  const tbody = $("standings").querySelector("tbody");
  tbody.replaceChildren();
  const wdl = (r) => `${r.wins}-${r.draws}-${r.losses}`;
  for (const entry of data.standings) {
    const row = document.createElement("tr");
    for (const text of [entry.name, entry.games, wdl(entry.vs_ai), wdl(entry.vs_people)]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.appendChild(cell);
    }
    tbody.appendChild(row);
  }
  if (!data.standings.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "muted";
    cell.textContent = "Jeszcze nikt nie skończył partii.";
    row.appendChild(cell);
    tbody.appendChild(row);
  }
  const recent = $("recent");
  recent.replaceChildren();
  for (const game of data.recent) {
    const item = document.createElement("li");
    const result = game.winner === "draw" ? "remis" : `wygrały ${COLOUR_PL[game.winner]}`;
    item.textContent = `${game.played_at.replace("T", " ").slice(0, 16)} · ${game.white} – ${game.black} · ${result} · ${game.plies} ruchów`;
    recent.appendChild(item);
  }
}

// --- playing ------------------------------------------------------------------------

function onSquare(row, col) {
  if (!state || busy || state.result) return;
  const here = key(row, col);
  const fromSelected = selected
    ? state.legal.filter((m) => key(...m.path[0]) === selected && key(...m.path[m.path.length - 1]) === here)
    : [];
  if (fromSelected.length === 1) return play(fromSelected[0]);
  if (fromSelected.length > 1) return choose(fromSelected);
  const canStart = state.legal.some((m) => key(...m.path[0]) === here);
  selected = canStart && selected !== here ? here : null;
  hideChooser();
  drawMarks();
}

// Two different captures can start and end on the same squares: ask which one.
function choose(moves) {
  const box = $("chooser");
  box.replaceChildren("Które bicie?");
  for (const move of moves) {
    const button = document.createElement("button");
    button.textContent = move.name;
    button.addEventListener("mouseenter", () => drawArrow(move));
    button.addEventListener("mouseleave", () => drawArrow(null));
    button.addEventListener("click", () => play(move));
    box.appendChild(button);
  }
  box.hidden = false;
}
function hideChooser() { $("chooser").hidden = true; drawArrow(null); }

async function play(move) {
  hideChooser();
  selected = null;
  busy = true;
  drawMarks();
  renderStatus();
  const request = api("/api/move", { game: state.game, move: move.id, analysis: wantAnalysis() });
  await animate(move);
  try {
    const next = await request;
    if (next.ai_move) {
      // show the position after my move, then let the AI's piece travel
      drawPieces(afterMove(state.pieces, move));
      await sleep(120);
      await animate(next.ai_move);
    }
    await show(next);
  } catch (error) {
    busy = false;
    alert(error.message);
    await show(await api(`/api/state?game=${state.game}`));
  }
}

// The position after a move, worked out locally just for the animation.
function afterMove(list, move) {
  const n = state.size;
  const [r0, c0] = move.path[0];
  const [r1, c1] = move.path[move.path.length - 1];
  const taken = new Set(move.captures.map(([r, c]) => key(r, c)));
  return list
    .filter((p) => !taken.has(key(p.row, p.col)) || (p.row === r0 && p.col === c0))
    .map((p) => {
      if (p.row !== r0 || p.col !== c0) return p;
      let kind = p.kind;
      if (kind === "w" && r1 === n - 1) kind = "W";
      if (kind === "b" && r1 === 0) kind = "B";
      return { row: r1, col: c1, kind };
    });
}

async function show(next) {
  const firstDraw = !state || state.size !== next.size || state.game !== next.game;
  const wasOver = state && state.result;
  state = next;
  busy = false;
  if (firstDraw) drawBoard();
  drawPieces(state.pieces);
  render();
  if (state.result && !wasOver) renderResults();
}

async function newGame(settings) {
  busy = true;
  if (state) renderStatus();
  // against the AI the human sits at the bottom; two humans keep White at the bottom
  flipped = settings.opponent === "ai" && settings.colour === "black";
  localStorage.setItem("warcaby-settings", JSON.stringify(settings));
  const next = await api("/api/new", { ...settings, analysis: wantAnalysis() });
  state = null;
  selected = null;
  hideChooser();
  await show(next);
}

// --- controls -----------------------------------------------------------------------

const dialog = $("new-dialog");
const form = $("new-form");

function savedSettings() {
  try { return JSON.parse(localStorage.getItem("warcaby-settings")) || {}; } catch { return {}; }
}

function syncDialog() {
  const human = form.opponent.value === "human";
  $("opponent-name-row").hidden = !human;
  $("strength-row").hidden = human;
}

$("new-game").addEventListener("click", () => {
  const saved = savedSettings();
  form.name.value = saved.name || "";
  form.opponent_name.value = saved.opponent_name || "";
  form.opponent.value = saved.opponent || "ai";
  form.colour.value = saved.colour || "white";
  form.simulations.value = String(saved.simulations || 200);
  syncDialog();
  dialog.showModal();
});
form.addEventListener("change", syncDialog);
$("new-cancel").addEventListener("click", () => dialog.close());
form.addEventListener("submit", () => {
  newGame({
    name: form.name.value.trim(),
    opponent: form.opponent.value,
    opponent_name: form.opponent_name.value.trim() || "Znajomy",
    colour: form.colour.value,
    simulations: Number(form.simulations.value),
  }).catch((error) => alert(error.message));
});

$("resign").addEventListener("click", async () => {
  if (!state || state.result || !confirm("Poddać partię?")) return;
  await show(await api("/api/resign", { game: state.game }));
});

$("show-analysis").addEventListener("change", async () => {
  if (!state) return;
  if (wantAnalysis() && !state.analysis && !state.result) {
    busy = true;
    renderStatus();
    await show(await api("/api/analyse", { game: state.game }));
  } else {
    renderAnalysis();
  }
});

(async function start() {
  const info = await api("/api/info");
  $("model-label").textContent = info.model.trained
    ? `model: ${info.model.label}`
    : "Brak wytrenowanej sieci: AI gra samym przeszukiwaniem, bez sieci.";
  renderResults();
  const saved = savedSettings();
  await newGame({
    name: saved.name || "Gość",
    opponent: "ai",
    opponent_name: saved.opponent_name || "Znajomy",
    colour: saved.colour || "white",
    simulations: saved.simulations || 200,
  });
})();
