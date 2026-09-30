const vm = require('vm');
const fs = require('fs');
const assert = require('assert');
const path = require('path');

const width = 600, height = 150;
let now = 0, frame = null, timerId = 0;
let scene = 'landscape';
let theme = 'rawLight';
let cactusSpeed = 0, birdSpeed = 0, slowCorrection = 0;
let highJumpKeys = 0, highDuckKeys = 0, lowJumpKeys = 0;
const keydowns = n => keys.filter(([t,code]) => t === 'keydown' && code === n).length;
const timers = new Map(), keys = [];
const game = { crashed: false };
const canvas = {
  width, height, style: {}, isConnected: true,
  getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
  getContext: () => ({
    getImageData: () => {
      const data = new Uint8ClampedArray(width * height * 4);
      if (theme === 'rawDark') {
        for (let i = 0; i < data.length; i += 4) {
          data[i] = data[i + 1] = data[i + 2] = 29;
          data[i + 3] = 255;
        }
      }
      const rect = (x0, x1, y0, y1, shade = 80) => {
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
          const i = (y * width + x) * 4;
          data[i] = data[i + 1] = data[i + 2] = shade;
          data[i + 3] = 255;
        }
      };
      const sprite = theme === 'rawLight' ? 80 : 182;
      const scenery = theme === 'rawLight' ? 213 : 43;
      rect(45, 85, 84, 129, sprite);
      rect(25, 70, 20, 65, scenery); // lua sobre a area horizontal do Dino
      const cloudX = Math.max(110, 240 - Math.floor(now / 33) * 5);
      rect(cloudX, cloudX + 40, 70, 79, scenery); // nuvem ate na velocidade do cacto
      const x = Math.max(100, 550 - Math.floor(now / 33) * 5);
      if (scene === 'cactus') rect(x, x + 20, 90, 128, sprite);
      if (scene === 'cactusGroup') {
        for (const offset of [0,12,24]) rect(x+offset, x+offset+10, 105, 128, sprite);
      }
      if (scene.startsWith('bird')) {
        const start = scene === 'birdMid' ? 35 : scene === 'birdHigh' ? 130 : 161;
        const bx = Math.max(100, 250 - (Math.floor(now / 33) - start) * 8);
        const top = scene === 'birdHigh' ? 62 : scene === 'birdMid' ? 88 : 108;
        rect(bx, bx + 40, top, top + 20, sprite);
      }
      return { data };
    },
    clearRect() {}, strokeRect() {}, fillText() {},
  }),
};
const root = { appendChild(el) { el.isConnected = true; }, style: {} };
const document = {
  hidden: false, head: root, documentElement: root,
  querySelector: sel => sel.includes('canvas') ? canvas : null,
  createElement: type => type === 'canvas' ? {
    width, height, style: {}, isConnected: true,
    getContext: () => ({ clearRect() {}, strokeRect() {}, fillText() {} }),
    remove() { this.isConnected = false; },
  } : { style: {}, isConnected: true, remove() { this.isConnected = false; } },
  addEventListener() {}, removeEventListener() {},
  dispatchEvent(e) { keys.push([e.type, e.keyCode]); },
};
const tf = {
  sequential: () => ({
    layers: [], add(layer) { this.layers.push(layer); }, compile() {},
    getWeights: () => [{ clone: () => ({}) }], setWeights() {},
    predict(input) { return { arraySync: () => input.data.map(() => [0, 0, 0]),
      dataSync: () => [0, 0, 0] }; },
    trainOnBatch: async () => 0.1,
  }),
  layers: { dense: x => x }, train: { adam: () => ({}) },
  losses: { huberLoss: () => 0 }, tidy: fn => fn(),
  tensor2d: data => ({ data, dispose() {} }),
  ready: async () => {}, nextFrame: async () => {}, getBackend: () => 'cpu',
};
const context = { document, tf, Runner: { instance_: game },
  KeyboardEvent: class { constructor(type, attrs) { this.type = type; Object.assign(this, attrs); } },
  performance: { now: () => now }, console: { info() {}, error(e) { throw e; }, table() {} },
  requestAnimationFrame(fn) { frame = fn; return 1; }, cancelAnimationFrame() {},
  setTimeout(fn, delay) { const id = ++timerId; timers.set(id, [now + delay, fn]); return id; },
  clearTimeout(id) { timers.delete(id); },
};
context.globalThis = context;
async function main() {
  const script = path.join(__dirname, '..', 'src', 'dino_rl.js');
  vm.runInNewContext(fs.readFileSync(script, 'utf8'), context);
  for (let i = 0; i < 205; i++) {
    await Promise.resolve();
    now += 33;
    for (const [id, [due, fn]] of [...timers]) if (due <= now) {
      timers.delete(id); fn();
    }
    if (i === 89) game.crashed = true;
    if (i === 94) game.crashed = false;
    if (i === 15) scene = 'cactusGroup';
    if (i === 29) scene = 'cactus';
    if (i === 35) { scene = 'birdMid'; theme = 'rawDark'; }
    if (i === 55) scene = 'cactus';
    if (i === 130) {
      scene = 'birdHigh'; highJumpKeys = keydowns(32); highDuckKeys = keydowns(40);
    }
    if (i === 161) { scene = 'birdLow'; lowJumpKeys = keydowns(32); }
    frame?.(now);
    if (i === 12) {
      const s = context.DinoRL5.status();
      assert(!s.detectedObstacle, 'paisagem detectada como obstaculo');
      assert(s.dino.top >= 80, 'lua misturada ao Dino');
    }
    if (i === 22) {
      const s = context.DinoRL5.status();
      assert.equal(s.obstacle.kind, 'cacto', 'grupo de cactos confundido com ave baixa');
      cactusSpeed = s.speedPxPerSecond;
    }
    if (i === 42) {
      const s = context.DinoRL5.status();
      assert.equal(s.obstacle.kind, 'ave');
      assert.equal(s.birdLevel, 'medio');
      birdSpeed = s.speedPxPerSecond;
    }
    if (i === 56) {
      const s = context.DinoRL5.status();
      assert(s.ducking, 'abaixamento terminou assim que a ave sumiu');
      assert(s.duckRemainingMs >= 100, 'tempo residual curto');
    }
    if (i === 67) assert(!context.DinoRL5.status().ducking,
      'abaixamento nao terminou apos a passagem da ave');
    if (i === 92) slowCorrection = context.DinoRL5.status().speedBands[0].correction;
    if (i === 143) {
      const s = context.DinoRL5.status();
      assert.equal(s.obstacle.kind, 'ave', 'ave alta nao detectada');
      assert.equal(s.birdLevel, 'alto');
      assert(!s.ducking, 'ave alta provocou abaixamento');
      assert.equal(s.speedBands[0].correction, slowCorrection,
        'reinicio apagou o ajuste de salto lento');
    }
    if (i === 150) {
      assert.equal(keydowns(32), highJumpKeys, 'ave alta provocou salto');
      assert.equal(keydowns(40), highDuckKeys, 'ave alta provocou abaixamento');
    }
    if (i === 173) {
      const s = context.DinoRL5.status();
      assert.equal(s.obstacle.kind, 'ave', 'ave baixa confundida com cacto');
      assert.equal(s.birdLevel, 'baixo');
      assert(!s.ducking, 'ave baixa provocou abaixamento');
    }
    if (i === 181) assert(keydowns(32) > lowJumpKeys,
      'ave baixa nao provocou salto');
  }
  const status = context.DinoRL5.status();
  assert(status.episode >= 1, 'colisao nao encerrou o episodio');
  assert(keys.some(([t, n]) => t === 'keyup' && n === 13), 'Enter nao reiniciou');
  assert(status.replay > 0, 'sem dados de replay');
  assert(birdSpeed > cactusSpeed, 'velocidade do objeto nao se adaptou');
  assert(status.speedBands.some(b => b.attempts > 0 && b.correction !== 0),
    'colisao apos salto nao ajustou a faixa de velocidade');
  assert(keys.some(([t,n]) => t === 'keydown' && n === 32), 'sem saltos');
  console.log({ episode: status.episode, replay: status.replay,
    batches: status.batches, enter: keys.filter(([t,n]) => t === 'keyup' && n === 13).length });
}
main().catch(error => { console.error(error); process.exitCode = 1; });
