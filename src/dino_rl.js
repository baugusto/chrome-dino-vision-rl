/**
 * Dino Vision RL v5.5 — cole tudo em um Snippet novo em chrome://dino.
 * TensorFlow.js + Double DQN, saltos por velocidade e visao do canvas.
 * Comandos: DinoRL5.status(), .stop(), .start(), .exportModel(), .importModel().
 * Obstaculos e uma metrica aproximada; nao substitui o placar do Chrome.
 */
(async () => {
  "use strict";
  globalThis.DinoRL5?.stop?.();
  globalThis.DinoVisionRL?.stop?.();

  const C = Object.freeze({
    tickMs: 33, freezeFrames: 11, longFrameMs: 250, duckHoldMs: 320,
    gamma: 0.985, rate: 0.0002, batch: 32, warmup: 500,
    replay: 12000, trainAfterEpisode: 10, syncEvery: 10,
    epsilonMin: 0.03, epsilonStart: 0.12,
  });

  async function getTF() {
    if (globalThis.tf?.sequential) return globalThis.tf;
    for (const url of [
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
      "https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
    ]) {
      try {
        await new Promise((resolve, reject) => {
          const s = document.createElement("script");
          s.src = url; s.onload = resolve; s.onerror = reject;
          document.head.appendChild(s);
        });
        if (globalThis.tf?.sequential) return globalThis.tf;
      } catch (_) {}
    }
    throw new Error("Nao consegui carregar TensorFlow.js.");
  }

  const tf = await getTF();
  await tf.ready();
  const canvas = document.querySelector("canvas.runner-canvas") ||
    document.querySelector("canvas");
  if (!canvas || canvas.width < 200) throw new Error("Canvas do Dino nao encontrado.");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Leitura do canvas indisponivel.");

  const scale = canvas.width / (parseFloat(canvas.style.width) || canvas.width);
  const stateSize = 16;
  const makeModel = () => {
    const m = tf.sequential();
    m.add(tf.layers.dense({ inputShape: [stateSize], units: 64, activation: "relu" }));
    m.add(tf.layers.dense({ units: 64, activation: "relu" }));
    m.add(tf.layers.dense({ units: 3, activation: "linear" }));
    m.compile({ optimizer: tf.train.adam(C.rate), loss: tf.losses.huberLoss });
    return m;
  };
  let model = makeModel();
  const target = makeModel();
  const sync = () => tf.tidy(() => {
    target.setWeights(model.getWeights().map(w => w.clone()));
  });
  sync();

  const memory = [], important = [], recent = [];
  let running = true, busy = false, hidden = document.hidden;
  let raf = 0, restartTimer = 0, keyTimer = 0, duck = false;
  let duckUntil = -Infinity;
  let prevPixels = null, prevObs = null, lastTick = 0, frozen = 0;
  let inkDark = true;
  let activeMs = 0, episode = 0, passed = 0, passedTotal = 0;
  let bestPassed = 0, bestActive = 0, sawObstacle = false, sawMotion = false;
  let pending = null, lastAction = 0, lastLoss = null, batches = 0;
  let lastJumpAt = -Infinity, passCandidateAt = null;
  let jumpTrial = null, passTrial = null;
  let epsilon = C.epsilonStart, evaluatedEpisodes = 0;
  let mode = "reforco", restartPending = false;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
  // A velocidade recomeça baixa a cada partida, mas estas faixas não reiniciam.
  // Uma falha lenta só altera sua faixa, sem apagar o salto aprendido em alta velocidade.
  const speedBands = [
    [150, 0.28], [300, 0.30], [425, 0.33], [550, 0.36],
    [675, 0.39], [800, 0.42], [950, 0.45],
  ].map(([speed, base]) => ({speed, base, correction:0, attempts:0, successes:0}));
  function leadForSpeed(pxPerSecond) {
    const speed = pxPerSecond / scale;
    let i = 0;
    while (i < speedBands.length-2 && speed > speedBands[i+1].speed) i++;
    const a = speedBands[i], b = speedBands[i+1];
    const ratio = clamp((speed-a.speed)/(b.speed-a.speed), 0, 1);
    return clamp((a.base+a.correction)*(1-ratio) +
      (b.base+b.correction)*ratio, 0.20, 0.60);
  }
  function learnJump(trial, success, obs) {
    if (!trial) return;
    const speed = trial.speed;
    const index = speedBands.reduce((best, b, i) =>
      Math.abs(b.speed-speed) < Math.abs(speedBands[best].speed-speed) ? i : best, 0);
    const band = speedBands[index];
    band.attempts++;
    if (success) band.successes++;
    const near = trial.near;
    const age = (obs?.at || performance.now()) - trial.at;
    const dy = near?.dy ?? (obs && prevObs ?
      (obs.dino.top-prevObs.dino.top)/scale : 0);
    let direction = dy < -0.8 ? 1 : dy > 0.8 ? -1 : age < 210 ? 1 : -1;
    const lowClearance = near && near.clearance < Math.min(near.height*0.65+7, 34);
    const step = success ? (lowClearance ? 0.004*direction : 0) : 0.012*direction;
    band.correction = clamp(band.correction+step, -0.10, 0.10);
  }

  const hud = document.createElement("div");
  hud.id = "dino-rl5-hud";
  Object.assign(hud.style, {
    position: "fixed", top: "10px", right: "10px", zIndex: 99999,
    background: "#151923eb", color: "#b9ffca", padding: "10px",
    borderRadius: "6px", font: "12px/1.5 monospace", pointerEvents: "none",
    minWidth: "250px", whiteSpace: "pre-line",
  });
  document.documentElement.appendChild(hud);
  const overlay = document.createElement("canvas");
  overlay.width = canvas.width; overlay.height = canvas.height;
  Object.assign(overlay.style, { position: "fixed", zIndex: 99998,
    pointerEvents: "none" });
  document.documentElement.appendChild(overlay);
  const overlayCtx = overlay.getContext("2d");

  function key(type, name, code, number) {
    const e = new KeyboardEvent(type, { key: name, code, bubbles: true });
    for (const field of ["keyCode", "which"]) {
      try { Object.defineProperty(e, field, { get: () => number }); } catch (_) {}
    }
    document.dispatchEvent(e);
  }
  function releaseDuck() {
    if (duck) key("keyup", "ArrowDown", "ArrowDown", 40);
    duck = false; duckUntil = -Infinity;
  }
  function jump() {
    releaseDuck();
    key("keydown", " ", "Space", 32);
    lastJumpAt = performance.now();
    clearTimeout(keyTimer);
    keyTimer = setTimeout(() => key("keyup", " ", "Space", 32), 130);
  }
  function crashed() {
    return globalThis.Runner?.instance_?.crashed === true;
  }
  function restartGame() {
    // Enter no keyup reinicia sem aguardar a liberacao do Espaco.
    key("keyup", "Enter", "Enter", 13);
  }
  function act(action, obs) {
    const now = performance.now();
    if (action === 2 && obs.birdLevel === "medio" && obs.ttc < 0.42) {
      duckUntil = Math.max(duckUntil, now + C.duckHoldMs);
    }
    if (action === 1 && now - lastJumpAt < 200) action = 0;
    if (action === 0 && now < duckUntil &&
        (!obs.bird || obs.birdLevel === "medio")) action = 2;
    if (obs.bird && obs.birdLevel !== "medio" && action !== 1) releaseDuck();
    if (action === 1) {
      jump();
      if (obs.next && (!obs.bird || obs.birdLevel === "baixo")) {
        jumpTrial = { speed: obs.speed/scale, lead: obs.lead,
          kind: obs.bird ? "ave-baixa" : "cacto",
          actualTtc: obs.ttc, at: obs.at, near: null };
      }
    }
    if (action === 0) releaseDuck();
    else if (action === 2) {
      if (!duck) key("keydown", "ArrowDown", "ArrowDown", 40);
      duck = true;
    }
    lastAction = action;
    return action;
  }

  function columns(mask, w, x0, x1, y0, y1) {
    const result = []; let box = null;
    for (let x = x0; x <= x1; x++) {
      let count = 0, top = y1, bottom = y0;
      for (let y = y0; y <= y1; y++) if (mask[y * w + x]) {
        count++; top = Math.min(top, y); bottom = Math.max(bottom, y);
      }
      if (count >= 2) {
        if (!box) box = { left: x, right: x, top, bottom, pixels: 0 };
        box.right = x; box.top = Math.min(box.top, top);
        box.bottom = Math.max(box.bottom, bottom); box.pixels += count;
      } else if (box && x - box.right > 2) {
        result.push(box); box = null;
      }
    }
    if (box) result.push(box);
    return result.filter(b => b.right - b.left >= 3 && b.bottom - b.top >= 6 && b.pixels >= 12);
  }

  function observe(now) {
    const w = canvas.width, h = canvas.height;
    const raw = ctx.getImageData(0, 0, w, h).data;
    const bg = raw[0] * 0.299 + raw[1] * 0.587 + raw[2] * 0.114;
    // getImageData le pixels ANTES de filtros CSS como invert(100%).
    // Nuvens semi-transparentes ou claras nao devem virar sprites.
    if (raw[3] >= 180) {
      inkDark = bg >= 145;
    } else {
      let dark = 0, light = 0;
      for (let y = Math.floor(h*0.55); y < Math.floor(h*0.85); y += 2) {
        for (let x = Math.floor(25*scale); x < Math.min(w, Math.floor(90*scale)); x += 2) {
          const i = (y*w+x)*4;
          if (raw[i+3] < 200) continue;
          const lum = raw[i]*0.299 + raw[i+1]*0.587 + raw[i+2]*0.114;
          if (lum >= 30 && lum < 145) dark++;
          if (lum >= 145 && lum < 235) light++;
        }
      }
      if (dark+light > 12) inkDark = dark >= light;
    }
    const mask = new Uint8Array(w * h);
    const ground = Math.floor(h * 0.87);
    const dinoMinX = Math.floor(20 * scale);
    const dinoMaxX = Math.floor(105 * scale);
    let changes = 0, endInk = 0;
    for (let y = Math.floor(h * 0.10); y < ground - 2; y++) {
      for (let x = 20; x < w - 10; x++) {
        const p = y * w + x, i = p * 4;
        const lum = raw[i] * 0.299 + raw[i + 1] * 0.587 + raw[i + 2] * 0.114;
        const alpha = raw[i+3];
        mask[p] = alpha >= 180 && (inkDark
          ? lum >= 30 && lum < 145
          : lum >= 110 && lum < 235) ? 1 : 0;
        if (prevPixels && x > 125*scale && mask[p] !== prevPixels[p]) changes++;
        if (mask[p] && x > w*0.36 && x < w*0.64 &&
            y > h*0.20 && y < h*0.46) endInk++;
      }
    }
    prevPixels = mask;
    const dino = columns(mask, w, dinoMinX, Math.min(dinoMaxX, w - 1),
      Math.floor(h * 0.10), ground - 3)
      .filter(b => b.left < 55*scale && b.pixels > 16*scale*scale)
      .sort((a, b) => Math.abs(a.left-45*scale)-Math.abs(b.left-45*scale))[0] ||
      prevObs?.dino || { left: 45*scale, right: 87*scale,
        top: 84*scale, bottom: ground - 3 };
    const rawCandidates = columns(mask, w,
      Math.max(dinoMaxX, Math.floor(dino.right + 8*scale)), w - 4,
      Math.floor(h * 0.10), ground - 3)
      .map(b => {
        const width = (b.right-b.left+1)/scale;
        const height = (b.bottom-b.top+1)/scale;
        const fromGround = (ground-b.bottom)/scale;
        // O corpo da ave é uma faixa horizontal contínua. Cactos agrupados
        // podem ser largos, mas seus troncos ficam separados nessa linha.
        let widestRow = 0;
        for (let y = b.top; y <= b.bottom; y++) {
          let run = 0;
          for (let x = b.left; x <= b.right; x++) {
            run = mask[y*w+x] ? run+1 : 0;
            widestRow = Math.max(widestRow, run);
          }
        }
        const birdAboveGround = fromGround > 16 && fromGround <= 85 &&
          height >= 7 && height <= 40 && width >= 18 && width <= 64;
        const lowBirdShape = fromGround <= 16 && fromGround >= -3 &&
          height >= 7 && height <= 34 && width >= 28 && width <= 64 &&
          width / height >= 1.4 && widestRow >= 16*scale;
        const bird = birdAboveGround || lowBirdShape;
        const cactus = !bird && fromGround <= 16 && height >= 16 &&
          height <= 62 && width >= 6 && width <= 80;
        const density = b.pixels / ((b.right-b.left+1)*(b.bottom-b.top+1));
        return { ...b, kind: density < 0.18 ? null :
          bird ? "ave" : cactus ? "cacto" : null };
      })
      .filter(b => b.kind);
    const delta = prevObs ? Math.max(1, now - prevObs.at) : C.tickMs;
    // Cacto/ave atravessa a tela rapidamente; paisagem parada ou lenta nao entra.
    const obstacles = rawCandidates
      .map(b => {
        const previous = prevObs?.rawCandidates?.find(p =>
          p.kind === b.kind &&
          p.left-b.left >= 2*scale &&
          p.left-b.left <= (delta*scale + 8*scale) &&
          Math.abs(p.top-b.top) <= 14*scale &&
          Math.abs((p.right-p.left)-(b.right-b.left)) <= 18*scale);
        return previous ? { ...b, measuredSpeed: (previous.left-b.left)*1000/delta } : null;
      })
      .filter(Boolean)
      .sort((a, b) => a.left - b.left);
    const next = obstacles[0] || null;
    const distance = next ? Math.max(0, next.left - dino.right) : w;
    const measuredSpeed = next?.measuredSpeed || 0;
    const speed = measuredSpeed > 50*scale && measuredSpeed < 1000*scale
      ? (prevObs?.speed || measuredSpeed) * 0.55 + measuredSpeed * 0.45
      : (prevObs?.speed || 330*scale);
    const ttc = next ? distance / speed : 9;
    const lead = leadForSpeed(speed);
    const airborne = dino.bottom < ground - Math.max(8*scale, h * 0.06);
    const bird = next?.kind === "ave";
    // Posições do sprite no Chrome: 50, 75 e 100 px; cada nível pede outra ação.
    const birdBottom = next ? next.bottom / scale : 0;
    const groundCss = ground / scale;
    const birdLevel = !bird ? null : birdBottom <= groundCss-40 ? "alto" :
      birdBottom <= groundCss-17 ? "medio" : "baixo";
    const obs = { at: now, w, h, dino, next, distance, speed, ttc, lead,
      bird, birdLevel, airborne, changes, endInk, rawCandidates,
      second: obstacles[1] || null };
    obs.state = [
      clamp(ttc / 1.5, 0, 1), clamp(distance / w, 0, 1),
      clamp(speed / (1100*scale), 0, 1), next ? clamp((next.right-next.left) / (75*scale), 0, 1) : 0,
      next ? clamp((next.bottom-next.top) / (75*scale), 0, 1) : 0,
      bird ? 1 : 0, airborne ? 1 : 0,
      clamp(dino.top / h, 0, 1),
      clamp(0.5 + (prevObs ? (dino.top-prevObs.dino.top) / (15*scale) : 0), 0, 1),
      obs.second ? clamp((obs.second.left-dino.right) / w, 0, 1) : 1,
      duck ? 1 : 0, next ? 1 : 0,
      clamp(0.5 + (speed-(prevObs?.speed || speed))/(120*scale), 0, 1),
      clamp(lead/0.65, 0, 1),
      bird ? clamp((ground-next.bottom)/(85*scale), 0, 1) : 0,
      birdLevel === "alto" ? 0.33 : birdLevel === "medio" ? 0.67 :
        birdLevel === "baixo" ? 1 : 0,
    ];
    return obs;
  }

  function debug(obs) {
    const r = canvas.getBoundingClientRect();
    Object.assign(overlay.style, { left: `${r.left}px`, top: `${r.top}px`,
      width: `${r.width}px`, height: `${r.height}px` });
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    for (const [b, color] of [[obs.dino, "#2d8cff"], [obs.next, "#ff4b4b"]]) {
      if (!b) continue;
      overlayCtx.strokeStyle = color; overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(b.left, b.top, b.right-b.left+1, b.bottom-b.top+1);
    }
    overlayCtx.fillStyle = "#f34c4c";
    overlayCtx.font = "12px monospace";
    if (obs.next) overlayCtx.fillText(`${obs.ttc.toFixed(2)} s`, obs.next.left, Math.max(12, obs.next.top-5));
  }

  function show(obs = prevObs) {
    hud.textContent = [
      `Dino RL v5.5 | ${hidden ? "PAUSADO (aba oculta)" : mode}`,
      `Episodio ${episode} | ${Math.round(activeMs/1000)}s ativos`,
      `Obstaculos ${passed} | recorde ${bestPassed}`,
      `Ultimas 20: ${mean(recent).toFixed(1)} obstaculos`,
      `TTC ${obs?.next ? obs.ttc.toFixed(2)+"s" : "--"} | visao ${obs?.next ? (obs.bird ? "ave "+obs.birdLevel : "cacto") : "livre"}`,
      `Velocidade ${obs ? Math.round(obs.speed/scale) : "--"} px/s | salto ${(obs?.lead ?? leadForSpeed(330*scale)).toFixed(2)}s`,
      `Epsilon ${epsilon.toFixed(3)} | saltos avaliados ${speedBands.reduce((n,b) => n+b.attempts,0)}`,
      `Replay ${memory.length} | batches ${batches} | perda ${lastLoss === null ? "--" : lastLoss.toFixed(4)}`,
      "Rede + regra por altura e velocidade",
    ].join("\n");
  }

  function heuristic(obs) {
    if (!obs.next || obs.ttc > 0.85) return 0;
    if (obs.birdLevel === "alto") return 0;
    if (obs.birdLevel === "medio") return obs.ttc < 0.40 ? 2 : 0;
    if (!obs.airborne && obs.at-lastJumpAt >= 200 &&
        obs.ttc <= obs.lead && obs.ttc >= 0.08) return 1;
    return 0;
  }

  function permitted(obs, action) {
    if (action === 1 && (obs.airborne ||
        (obs.bird && obs.birdLevel !== "baixo") ||
        obs.at-lastJumpAt < 200)) return 0;
    if (action === 2 && obs.birdLevel !== "medio") return 0;
    return action;
  }

  function choose(obs) {
    const rule = heuristic(obs);
    if (mode !== "reforco" || memory.length < C.warmup || passedTotal < 3 ||
        !obs.next || obs.ttc > 0.80) return rule;
    evaluatedEpisodes++;
    const mix = Math.min(0.80, 0.20 + (episode - 30) / 120);
    if (Math.random() > mix) return rule;
    // Exploracao ocorre apenas na zona do obstaculo, nunca na pista vazia.
    epsilon = Math.max(C.epsilonMin, C.epsilonStart * Math.exp(-evaluatedEpisodes/6000));
    let proposed;
    if (Math.random() < epsilon) {
      proposed = obs.birdLevel === "alto" ? 0 :
        obs.birdLevel === "medio" ? (Math.random() < 0.5 ? 0 : 2) :
        (Math.random() < 0.5 ? 0 : 1);
    } else {
      proposed = tf.tidy(() => {
        const x = tf.tensor2d([obs.state], [1, stateSize]);
        const q = model.predict(x).dataSync();
        const valid = obs.birdLevel === "alto" ? [0] :
          obs.birdLevel === "medio" ? [0,2] :
          obs.airborne ? [0] : [0,1];
        return valid.reduce((a, b) => q[b] > q[a] ? b : a);
      });
    }
    proposed = permitted(obs, proposed);
    // Evita que uma Q-value inicial e aleatoria force morte certa.
    if (obs.birdLevel === "alto") return 0;
    if (obs.birdLevel === "medio") {
      if (obs.ttc < 0.30) return 2;
      if (obs.ttc > 0.42) return 0;
      return proposed;
    }
    // A rede pode afinar o salto, mas não antecipá-lo além da faixa aprendida.
    if (!obs.airborne && obs.ttc < obs.lead-0.035 && obs.ttc >= 0.08) return 1;
    if (proposed === 1 && obs.ttc > obs.lead+0.02) return 0;
    return proposed;
  }

  function remember(t) {
    memory.push(t);
    if (t.important) important.push(t);
    if (memory.length > C.replay) {
      const old = memory.shift(); old.valid = false;
    }
    if (important.length > C.replay) important.shift();
  }

  function transition(old, obs) {
    if (!old) return;
    const vanished = old.obs.next && old.obs.ttc < 0.30 &&
      (!obs.next || obs.distance > old.obs.distance + 100);
    if (vanished && passCandidateAt === null) {
      passCandidateAt = obs.at;
      if (!old.obs.bird || old.obs.birdLevel === "baixo") {
        passTrial = jumpTrial; jumpTrial = null;
      }
    }
    const confirmedPass = passCandidateAt !== null &&
      obs.at-passCandidateAt > 350 && obs.changes > 12 && frozen === 0;
    if (confirmedPass) {
      passed++; passedTotal++; passCandidateAt = null;
      learnJump(passTrial, true, obs); passTrial = null;
    }
    remember({ state: old.obs.state, action: old.action,
      reward: (confirmedPass ? 1 : 0.002) - (old.action === 1 ? 0.004 : 0),
      nextState: obs.state, done: false,
      important: old.obs.ttc < 0.70 || confirmedPass, valid: true });
  }

  async function trainOne() {
    if (!running || memory.length < C.warmup) return;
    const batch = Array.from({ length: C.batch }, () => {
      const pool = important.length && Math.random() < 0.5 ? important : memory;
      for (let i = 0; i < 10; i++) {
        const item = pool[Math.floor(Math.random() * pool.length)];
        if (item?.valid) return item;
      }
      return memory[Math.floor(Math.random() * memory.length)];
    });
    const xs = batch.map(t => t.state), ns = batch.map(t => t.nextState);
    let tx, ty;
    try {
      const q = tf.tidy(() => model.predict(tf.tensor2d(xs, [C.batch,stateSize])).arraySync());
      const nextQ = tf.tidy(() => model.predict(tf.tensor2d(ns, [C.batch,stateSize])).arraySync());
      const targetQ = tf.tidy(() => target.predict(tf.tensor2d(ns, [C.batch,stateSize])).arraySync());
      const ys = q.map((row, i) => {
        const t = batch[i], copy = row.slice();
        const best = nextQ[i].indexOf(Math.max(...nextQ[i]));
        copy[t.action] = clamp(t.reward + (t.done ? 0 : C.gamma * targetQ[i][best]), -3, 3);
        return copy;
      });
      tx = tf.tensor2d(xs, [C.batch,stateSize]);
      ty = tf.tensor2d(ys, [C.batch,3]);
      const loss = await model.trainOnBatch(tx, ty);
      lastLoss = Number(Array.isArray(loss) ? loss[0] : loss);
      batches++;
    } catch (error) {
      console.error("[DinoRL5] Treino falhou:", error);
      running = false;
    } finally { tx?.dispose(); ty?.dispose(); }
  }

  function finish(terminalObs = null) {
    if (restartPending) return;
    restartPending = true;
    const collided = terminalObs?.next || prevObs?.next;
    const collidedKind = collided?.kind === "ave" ? "ave-baixa" : "cacto";
    if (jumpTrial && collided && jumpTrial.kind === collidedKind) {
      learnJump(jumpTrial, false, terminalObs || prevObs);
    }
    jumpTrial = null; passTrial = null;
    passCandidateAt = null;
    releaseDuck();
    // A ultima transicao observada apos o congelamento ainda pertence ao jogo.
    // Remove as capturas congeladas e pune a ultima decisao valida.
    const discard = Math.min(frozen, Math.max(0, memory.length - 1));
    for (let i = 0; i < discard; i++) {
      const t = memory.pop(); t.valid = false;
    }
    if (terminalObs && pending) {
      remember({ state: pending.obs.state, action: pending.action,
        reward: -2, nextState: terminalObs.state, done: true,
        important: true, valid: true });
    }
    const terminal = memory[memory.length - 1];
    if (terminal && !terminal.done) {
      terminal.done = true; terminal.reward = -2;
      terminal.important = true; important.push(terminal);
    }
    const seconds = activeMs / 1000;
    episode++; bestPassed = Math.max(bestPassed, passed);
    bestActive = Math.max(bestActive, seconds);
    recent.push(passed); if (recent.length > 20) recent.shift();
    console.info(`[DinoRL5] ep=${episode} ativos=${seconds.toFixed(1)}s `+
      `obstaculos=${passed} media20=${mean(recent).toFixed(1)} `+
      `melhor=${bestPassed} batches=${batches} modo=${mode}`);
    show();
    void (async () => {
      busy = true;
      for (let i = 0; i < C.trainAfterEpisode && running; i++) {
        await trainOne();
        await tf.nextFrame();
      }
      if (episode % C.syncEvery === 0) sync();
      busy = false;
      if (!running) return;
      restartTimer = setTimeout(() => {
        if (!running || hidden) return;
        prevPixels = null; prevObs = null; pending = null;
        frozen = 0; sawMotion = false; sawObstacle = false;
        passCandidateAt = null; jumpTrial = null; passTrial = null;
        activeMs = 0; passed = 0; lastTick = 0;
        restartPending = false; restartGame(); show();
      }, 250);
    })();
  }

  function frame(now) {
    if (!running) return;
    raf = requestAnimationFrame(frame);
    if (hidden || busy || restartPending) return;
    if (lastTick && now-lastTick < C.tickMs) return;
    const gap = lastTick ? now-lastTick : C.tickMs;
    lastTick = now;
    if (gap > C.longFrameMs) {
      // Aba suspensa ou compilacao lenta: nao cria transicao nem inventa tempo.
      prevPixels = null; prevObs = null; pending = null; frozen = 0;
      passCandidateAt = null; jumpTrial = null; passTrial = null;
      return;
    }
    activeMs += Math.min(gap, C.tickMs*2);
    let obs;
    try { obs = observe(now); } catch (error) {
      console.error("[DinoRL5] Leitura do canvas falhou:", error);
      globalThis.DinoRL5.stop(); return;
    }
    debug(obs);
    if (jumpTrial && obs.next &&
        (!obs.bird || obs.birdLevel === "baixo") && obs.ttc < 0.13) {
      jumpTrial.near = {
        clearance: Math.max(0, (Math.floor(obs.h*0.87)-obs.dino.bottom)/scale),
        dy: prevObs ? (obs.dino.top-prevObs.dino.top)/scale : 0,
        height: (obs.next.bottom-obs.next.top+1)/scale,
      };
    }
    if (obs.next) sawObstacle = true;
    if (obs.changes > 12*scale) sawMotion = true;
    // Texto GAME OVER no centro + imagem parada. Pista vazia nao encerra partida.
    frozen = sawMotion && obs.endInk > 170*scale*scale &&
      obs.changes < 3*scale ? frozen+1 : 0;
    if (crashed()) { finish(obs); return; }
    if (pending) transition(pending, obs);
    prevObs = obs;
    if (sawObstacle && activeMs > 2000 && frozen >= C.freezeFrames) {
      finish(); return;
    }
    const action = act(choose(obs), obs);
    pending = { obs, action };
    show(obs);
  }

  function visibility() {
    hidden = document.hidden;
    prevPixels = null; prevObs = null; pending = null;
    frozen = 0; lastTick = 0; passCandidateAt = null;
    jumpTrial = null; passTrial = null;
    if (!hidden && restartPending && !busy) {
      clearTimeout(restartTimer);
      restartTimer = setTimeout(() => {
        if (!running || document.hidden) return;
        activeMs = 0; passed = 0; sawMotion = false; sawObstacle = false;
        passCandidateAt = null; jumpTrial = null; passTrial = null;
        restartPending = false; restartGame();
      }, 250);
    }
    show();
  }
  document.addEventListener("visibilitychange", visibility);

  globalThis.DinoRL5 = {
    stop() {
      running = false; cancelAnimationFrame(raf);
      clearTimeout(restartTimer); clearTimeout(keyTimer);
      releaseDuck(); document.removeEventListener("visibilitychange", visibility);
      overlay.remove(); hud.remove();
    },
    start() {
      if (running) return;
      running = true; hidden = document.hidden;
      restartPending = false; prevPixels = null; prevObs = null; pending = null;
      lastTick = 0; frozen = 0; activeMs = 0; passed = 0;
      sawObstacle = false; sawMotion = false; passCandidateAt = null;
      jumpTrial = null; passTrial = null;
      if (!hud.isConnected) document.documentElement.appendChild(hud);
      if (!overlay.isConnected) document.documentElement.appendChild(overlay);
      document.addEventListener("visibilitychange", visibility);
      jump(); raf = requestAnimationFrame(frame);
    },
    status() {
      const s = { running, hidden, mode, episode, passed, passedTotal,
        meanPassedLast20: +mean(recent).toFixed(2), bestPassed,
        activeSeconds: +(activeMs/1000).toFixed(2), bestActiveSeconds: +bestActive.toFixed(2),
        detectedObstacle: Boolean(prevObs?.next), ttc: prevObs?.ttc.toFixed(2),
        birdLevel: prevObs?.birdLevel ?? null,
        dino: prevObs?.dino, obstacle: prevObs?.next,
        speedPxPerSecond: prevObs ? Math.round(prevObs.speed/scale) : null,
        leadSeconds: prevObs?.lead ?? leadForSpeed(330*scale),
        speedBands: speedBands.map(b => ({...b, correction: +b.correction.toFixed(3)})),
        replay: memory.length, batches,
        epsilon: +epsilon.toFixed(3), backend: tf.getBackend(),
        pixelMode: inkDark ? "sprite escuro" : "sprite claro",
        ducking: duck, duckRemainingMs: Math.max(0, Math.round(duckUntil-performance.now())) };
      console.table(s); return s;
    },
    async exportModel() {
      try {
        await model.save("downloads://dino-rl-v5");
        console.info("[DinoRL5] Arquivos do modelo exportados.");
      } catch (error) {
        console.error("[DinoRL5] O Chrome bloqueou a exportacao:", error);
      }
    },
    async importModel() {
      const picker = document.createElement("input");
      picker.type = "file"; picker.multiple = true; picker.accept = ".json,.bin";
      picker.onchange = async () => {
        const files = [...picker.files].sort((a,b) =>
          Number(b.name.endsWith(".json")) - Number(a.name.endsWith(".json")));
        if (files.length !== 2) return console.error("Selecione juntos o JSON e o .bin.");
        try {
          const loaded = await tf.loadLayersModel(tf.io.browserFiles(files));
          if (loaded.inputs[0].shape.at(-1) !== stateSize) {
            loaded.dispose(); throw new Error("Modelo com estado incompatível.");
          }
          const old = model; model = loaded;
          model.compile({ optimizer: tf.train.adam(C.rate), loss: tf.losses.huberLoss });
          old.dispose(); sync();
          console.info("[DinoRL5] Modelo importado.");
        } catch (error) { console.error("[DinoRL5] Falha na importacao:", error); }
      };
      picker.click();
    },
    config: C,
  };

  console.info("[DinoRL5] Iniciado. Azul=Dino, vermelho=obstaculo; use DinoRL5.status().");
  jump(); raf = requestAnimationFrame(frame);
})();
