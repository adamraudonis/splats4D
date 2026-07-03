// splat4d viewer — three.js WebGPURenderer (hard requirement, no WebGL fallback).
// Features: streaming playback with seek, live re-encode via the dev API
// (sliders -> Rust encoder), and a split-screen comparison against the
// original uncompressed .splat frames (same renderer, shared camera + sort).

import * as THREE from 'three/webgpu';
import WebGPU from 'three/addons/capabilities/WebGPU.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createSplatMesh, type SplatMesh } from './splatmesh';
import { OrigLoader, type OrigFrame } from './origloader';

interface Meta {
  n: number;
  t: number;
  fps: number;
  gop: number;
  aabb: number[];
  bounds: { pos_m: number; scale_rel: number; rgb: number; alpha: number; rot: number };
  gops: { f0: number; f1: number; t0: number; t1: number; offset: number; len: number }[];
  fileSize: number;
  denoised: boolean;
  headerMs: number;
}

const $ = (id: string) => document.getElementById(id)!;
const overlay = $('overlay');
const overlayMsg = $('overlay-msg');
const overlaySub = $('overlay-sub');

function fail(msg: string, sub = ''): never {
  overlay.classList.remove('hidden');
  overlay.querySelector('.spin')?.remove();
  overlayMsg.textContent = msg;
  overlaySub.textContent = sub;
  throw new Error(msg);
}

async function init() {
  if (!WebGPU.isAvailable()) {
    fail('WebGPU is required', 'This viewer intentionally has no WebGL fallback. Use Chrome 113+, Edge, Safari 26+, or Firefox 141+.');
  }
  const renderer = new THREE.WebGPURenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x0b0d10);
  // splat colors are sRGB and blended as-is (reference-viewer convention)
  renderer.toneMapping = THREE.NoToneMapping;
  await renderer.init();
  const backend = (renderer as unknown as { backend: { isWebGPUBackend?: boolean } }).backend;
  if (backend.isWebGPUBackend !== true) {
    fail('WebGPU initialization fell back to WebGL', 'Refusing to run — this app is WebGPU-only.');
  }
  renderer.domElement.classList.add('webgpu');
  $('canvas-wrap').appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 200);
  let controls: OrbitControls | null = null;

  // ---- persistent state across re-encodes ----
  const tPage = performance.now();
  let splats: SplatMesh | null = null;
  let origMesh: SplatMesh | null = null;
  const origLoader = new OrigLoader('/frames', 0); // n patched on first meta
  let meta: Meta | null = null;
  let playing = false;
  let timeSec = 0;
  let compareOn = false;
  let dividerFrac = 0.5;

  // ---- per-session (per loaded file) state ----
  let session = 0;
  let sessionReady = false; // static state received; safe to request frames/sorts
  let worker: Worker | null = null;
  let buffered: boolean[] = [];
  let lastShownFrame = -1;
  let frameInFlight = false;
  let waitingGop = -1;
  let sortInFlight = false;
  let needSort = true;
  const lastSortRow = new Float32Array(3);
  const viewProj = new THREE.Matrix4();
  let fpsCount = 0;
  let fpsTime = performance.now();

  const timeline = $('timeline') as HTMLCanvasElement;
  const tctx = timeline.getContext('2d')!;

  // ================= playback plumbing =================
  function requestFrame(frame: number) {
    if (!meta || !worker || !sessionReady || frameInFlight) return;
    if (frame === lastShownFrame && waitingGop < 0) return;
    frameInFlight = true;
    worker.postMessage({ type: 'frame', frame });
  }

  function maybeSort(force = false) {
    if (!splats || sortInFlight || !meta || !worker || !sessionReady) return;
    viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const e = viewProj.elements;
    if (!force) {
      const dot = lastSortRow[0] * e[2] + lastSortRow[1] * e[6] + lastSortRow[2] * e[10];
      const a2 = e[2] ** 2 + e[6] ** 2 + e[10] ** 2;
      const b2 = lastSortRow[0] ** 2 + lastSortRow[1] ** 2 + lastSortRow[2] ** 2;
      if (b2 > 0 && Math.abs(dot / Math.sqrt(a2 * b2) - 1) < 0.01) return;
    }
    sortInFlight = true;
    lastSortRow[0] = e[2];
    lastSortRow[1] = e[6];
    lastSortRow[2] = e[10];
    worker.postMessage({ type: 'sort', viewProj: new Float32Array(e) });
  }

  function uploadOrig(f: OrigFrame) {
    if (!origMesh) return;
    origMesh.posArr.set(f.pos);
    origMesh.scaleArr.set(f.scale);
    origMesh.quatArr.set(f.quat);
    origMesh.rgbaArr.set(f.rgba);
    origMesh.markPos();
    origMesh.markScale();
    origMesh.markQuat();
    origMesh.markRgba();
  }

  /** (re)create meshes when the splat count changes (sequence switch) */
  function ensureMeshes(n: number) {
    if (splats && splats.posArr.length === n * 4) return;
    splats?.dispose(scene);
    origMesh?.dispose(scene);
    origMesh = null;
    splats = createSplatMesh(n);
    scene.add(splats.mesh);
    origLoader.n = n;
    if (compareOn) {
      origMesh = createSplatMesh(n, { sharedSort: { sortArr: splats.sortArr, sortNode: splats.sortNode } });
      scene.add(origMesh.mesh);
    }
    onResize();
    applyClips();
  }

  function refreshOrig(frame: number) {
    if (!compareOn) return;
    void origLoader.load(frame).then((f) => {
      if (f && compareOn && frame === lastShownFrame) uploadOrig(f);
    });
  }

  let cameraReady = false;
  function firstStaticSetup(m: Meta) {
    ensureMeshes(m.n);
    if (cameraReady) return;
    cameraReady = true;

    const [ax, , az, bx, , bz] = m.aabb;
    const originInside = ax < 0 && bx > 0 && az < 0 && bz > 0;
    const tx = originInside ? 0 : (ax + bx) / 2;
    const ty = originInside ? -0.9 : (m.aabb[1] + m.aabb[4]) / 2;
    const tz = originInside ? 0.3 : (az + bz) / 2;
    camera.up.set(0, -1, 0);
    camera.position.set(tx, ty - 0.3, tz - 2.9);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(tx, ty, tz);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.update();

    (window as unknown as Record<string, unknown>).__cam = (
      px: number, py: number, pz: number, qx: number, qy: number, qz: number
    ) => {
      camera.position.set(px, py, pz);
      controls!.target.set(qx, qy, qz);
      controls!.update();
    };
    (window as unknown as Record<string, unknown>).__play = (p: boolean) => setPlaying(p);
    (window as unknown as Record<string, unknown>).__compare = (on: boolean) => setCompare(on);

    onResize();
    overlay.classList.add('hidden');
    $('m-ttfv').textContent = `${(performance.now() - tPage).toFixed(0)} ms`;
  }

  function loadFile(url: string) {
    session++;
    const mySession = session;
    worker?.terminate();
    sessionReady = false;
    buffered = [];
    lastShownFrame = -1;
    frameInFlight = false;
    waitingGop = -1;
    sortInFlight = false;
    needSort = true;

    const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    worker = w;
    w.onmessage = (ev: MessageEvent) => {
      if (mySession !== session) return; // stale worker
      const m = ev.data;
      if (m.type === 'meta') {
        meta = m as Meta;
        for (let i = 0; i < meta.gops.length; i++) buffered.push(false);
        $('m-total').textContent = (meta.fileSize / 1e6).toFixed(1);
        $('m-splats').textContent = meta.n.toLocaleString();
        $('m-dyn').textContent = `gop ${meta.gop}`;
        $('m-bounds').textContent = `±${(meta.bounds.pos_m * 1000).toFixed(1)}mm pos · ±${meta.bounds.rgb} color · ±${meta.bounds.rot}/128 rot${meta.denoised ? ' · denoised' : ''}`;
        $('m-ratio').textContent = `${(meta.fileSize / 1e6).toFixed(1)} MB ← ${((meta.n * meta.t * 32) / 1e6).toFixed(0)} MB raw (${((meta.n * meta.t * 32) / meta.fileSize).toFixed(1)}×)`;
        overlaySub.textContent = `${meta.n.toLocaleString()} splats · ${meta.t} frames @ ${meta.fps} fps`;
      } else if (m.type === 'static') {
        if (!meta) return;
        $('m-static').textContent = `${m.staticMs.toFixed(0)} ms`;
        firstStaticSetup(meta);
        splats!.posArr.set(new Float32Array(m.pos));
        splats!.scaleArr.set(new Float32Array(m.scale));
        splats!.quatArr.set(new Uint32Array(m.quat));
        splats!.rgbaArr.set(new Uint32Array(m.rgba));
        splats!.markPos();
        splats!.markScale();
        splats!.markQuat();
        splats!.markRgba();
        sessionReady = true;
        if (pendingCamera) {
          applyCameraHint(pendingCamera);
          pendingCamera = null;
        }
        maybeSort(true);
        lastShownFrame = -1;
        requestFrame(Math.min(meta.t - 1, Math.floor(timeSec * meta.fps)));
      } else if (m.type === 'frame') {
        if (!splats) return;
        lastShownFrame = m.frame;
        if (!m.approximate) waitingGop = -1;
        splats.posArr.set(new Float32Array(m.pos));
        splats.quatArr.set(new Uint32Array(m.quat));
        splats.rgbaArr.set(new Uint32Array(m.rgba));
        splats.markPos();
        splats.markQuat();
        splats.markRgba();
        w.postMessage({ type: 'return', kind: 'pos', buffer: m.pos }, [m.pos]);
        w.postMessage({ type: 'return', kind: 'quat', buffer: m.quat }, [m.quat]);
        w.postMessage({ type: 'return', kind: 'rgba', buffer: m.rgba }, [m.rgba]);
        frameInFlight = false;
        $('m-frame').textContent = String(m.frame);
        $('m-decode').textContent = m.decodeMs.toFixed(1);
        needSort = true;
        refreshOrig(m.frame);
      } else if (m.type === 'miss') {
        frameInFlight = false;
        waitingGop = m.gop;
      } else if (m.type === 'buffered') {
        buffered[m.gop] = true;
        $('m-loaded').textContent = (m.bytesLoaded / 1e6).toFixed(1);
      } else if (m.type === 'sorted') {
        if (!splats) return;
        splats.sortArr.set(new Uint32Array(m.indices));
        splats.markSort();
        w.postMessage({ type: 'return', kind: 'sort', buffer: m.indices }, [m.indices]);
        sortInFlight = false;
        $('m-sort').textContent = m.sortMs.toFixed(1);
      } else if (m.type === 'sorted-skip') {
        sortInFlight = false;
        needSort = true;
      } else if (m.type === 'error') {
        fail('Stream error', m.message);
      }
    };
    // resolve relative paths against the page, not the worker script location
    w.postMessage({ type: 'load', url: new URL(url, location.href).href });
  }

  // ================= compare mode =================
  const divider = $('divider');
  const compareBtn = $('compare-btn') as HTMLButtonElement;

  function applyClips() {
    const ndc = dividerFrac * 2 - 1;
    if (compareOn && splats && origMesh) {
      origMesh.setClip(ndc, -1);
      splats.setClip(ndc, 1);
      divider.style.display = 'block';
      divider.style.left = `${dividerFrac * 100}%`;
    } else {
      splats?.setClip(0, 0);
      origMesh?.setClip(0, 0);
      divider.style.display = 'none';
    }
  }

  function setCompare(on: boolean) {
    if (!splats || !meta) return;
    if (on && !origLoader.hasPerm()) {
      $('enc-status').textContent = 'compare needs the dev API';
      return;
    }
    compareOn = on;
    compareBtn.classList.toggle('on', on);
    if (on) {
      if (!origMesh) {
        origMesh = createSplatMesh(meta.n, { sharedSort: { sortArr: splats.sortArr, sortNode: splats.sortNode } });
        scene.add(origMesh.mesh);
        onResize();
      }
      origMesh.mesh.visible = true;
      refreshOrig(Math.max(0, lastShownFrame));
    } else if (origMesh) {
      origMesh.mesh.visible = false;
    }
    applyClips();
  }
  compareBtn.onclick = () => setCompare(!compareOn);

  // divider drag
  {
    const grip = divider.querySelector('.grip') as HTMLElement;
    let dragging = false;
    grip.addEventListener('pointerdown', (e) => {
      dragging = true;
      try {
        grip.setPointerCapture(e.pointerId);
      } catch { /* synthetic */ }
    });
    addEventListener('pointermove', (e) => {
      if (!dragging) return;
      dividerFrac = Math.min(0.98, Math.max(0.02, e.clientX / innerWidth));
      applyClips();
    });
    addEventListener('pointerup', () => (dragging = false));
  }

  // ================= encode panel =================
  const seqSelect = $('s-seq') as HTMLSelectElement;
  let currentSeq = 'juggle_2s';
  interface CamHint {
    position: number[];
    target: number[];
    up?: number[];
    fov?: number; // vertical degrees — forward-facing captures want a narrow one
  }
  const seqCameras = new Map<string, CamHint | null>();
  let pendingCamera: CamHint | null = null;

  function applyCameraHint(hint: CamHint | null) {
    if (!hint || !controls) return;
    camera.up.set(hint.up?.[0] ?? 0, hint.up?.[1] ?? -1, hint.up?.[2] ?? 0);
    camera.position.set(hint.position[0], hint.position[1], hint.position[2]);
    controls.target.set(hint.target[0], hint.target[1], hint.target[2]);
    camera.fov = hint.fov ?? 60;
    camera.updateProjectionMatrix();
    controls.update();
    onResize(); // focal uniforms depend on fov
  }

  async function loadSequenceList() {
    try {
      const r = await fetch('/api/sequences');
      if (!r.ok) return;
      const { sequences } = (await r.json()) as {
        sequences: { id: string; frames: number; fps: number; splats: number; camera: CamHint | null }[];
      };
      if (!sequences.length) return;
      seqSelect.innerHTML = '';
      for (const s of sequences) {
        const opt = document.createElement('option');
        opt.value = s.id;
        const secs = (s.frames / s.fps).toFixed(1);
        opt.textContent = `${s.id.replace(/_2s$/, '')} · ${secs}s · ${(s.splats / 1000).toFixed(0)}k`;
        seqSelect.appendChild(opt);
        seqCameras.set(s.id, s.camera);
      }
      const preferred = sequences.find((s) => s.id === 'juggle_2s') ?? sequences[0];
      currentSeq = preferred.id;
      seqSelect.value = currentSeq;
      pendingCamera = seqCameras.get(currentSeq) ?? null;
    } catch {
      /* API absent (static hosting) */
    }
  }

  seqSelect.onchange = () => {
    currentSeq = seqSelect.value;
    origLoader.reset(`/frames/${currentSeq}`);
    timeSec = 0;
    playing = false;
    playBtn.textContent = '▶';
    pendingCamera = seqCameras.get(currentSeq) ?? null;
    void runEncode();
  };

  const sliders = {
    pos: $('s-pos') as HTMLInputElement,
    col: $('s-col') as HTMLInputElement,
    rot: $('s-rot') as HTMLInputElement,
    scl: $('s-scl') as HTMLInputElement,
    gop: $('s-gop') as HTMLInputElement,
    dn: $('s-dn') as HTMLInputElement,
    z: $('s-z') as HTMLSelectElement,
  };
  const showVals = () => {
    $('o-pos').textContent = `±${sliders.pos.value} mm`;
    $('o-col').textContent = `±${sliders.col.value}/255`;
    $('o-rot').textContent = `±${sliders.rot.value}/128`;
    $('o-scl').textContent = `±${sliders.scl.value}%`;
    $('o-gop').textContent = `${sliders.gop.value} fr`;
  };
  for (const s of [sliders.pos, sliders.col, sliders.rot, sliders.scl, sliders.gop]) {
    s.addEventListener('input', showVals);
  }
  showVals();

  function encodeParams(): string {
    return new URLSearchParams({
      seq: currentSeq,
      pos_mm: sliders.pos.value,
      color_levels: sliders.col.value,
      rot_steps: sliders.rot.value,
      scale_pct: sliders.scl.value,
      gop: sliders.gop.value,
      denoise: sliders.dn.checked ? '1' : '0',
      zstd: sliders.z.value,
    }).toString();
  }

  interface EncodeResponse {
    url: string;
    perm: string;
    cached: boolean;
    wallMs: number;
    report: {
      output: { bytes: number; ratio: number };
      times_s: { total: number };
      static_fracs: Record<string, number>;
      verify: null | { pos_mm: number; rgb_levels: number; rot_units: number; scale_pct: number; ok: boolean };
      denoise: null | { mean_dev: number; p99_dev: number };
    };
  }

  function showEncodeStats(r: EncodeResponse) {
    const rep = r.report;
    const v = rep.verify;
    const sf = rep.static_fracs;
    $('enc-stats').innerHTML =
      `<div><b>${(rep.output.bytes / 1e6).toFixed(1)} MB</b> · <b>${rep.output.ratio.toFixed(1)}×</b> smaller` +
      ` · encoded in ${rep.times_s.total.toFixed(1)} s${r.cached ? ' (cached)' : ''}</div>` +
      `<div class="s">static: pos ${(sf.pos * 100).toFixed(0)}% rot ${(sf.rot * 100).toFixed(0)}%` +
      ` color ${(sf.rgb * 100).toFixed(0)}%</div>` +
      (v
        ? `<div class="s">verified ✓ max err: ${v.pos_mm.toFixed(2)}mm · ${v.rgb_levels}/255 · ${v.rot_units}/128</div>`
        : '') +
      (rep.denoise ? `<div class="s">denoise dev: mean ${rep.denoise.mean_dev.toFixed(1)}, p99 ${rep.denoise.p99_dev.toFixed(0)}</div>` : '');
  }

  // copy current settings as ready-to-use CLI flags
  const copyBtn = $('copy-btn') as HTMLButtonElement;
  copyBtn.onclick = () => {
    const flags =
      `--pos-mm ${sliders.pos.value} --color-levels ${sliders.col.value}` +
      ` --rot-steps ${sliders.rot.value} --scale-pct ${sliders.scl.value}` +
      ` --gop ${sliders.gop.value}${sliders.dn.checked ? ' --denoise-colors' : ''}` +
      ` --zstd-level ${sliders.z.value}`;
    const showFlags = () => {
      let el = document.getElementById('flags-out');
      if (!el) {
        el = document.createElement('div');
        el.id = 'flags-out';
        el.style.cssText =
          'margin-top:6px;padding:6px 8px;background:#0d1117;border:1px solid #2c3540;' +
          'border-radius:4px;user-select:all;word-break:break-all;color:#9ecbff;';
        $('enc-stats').before(el);
      }
      el.textContent = flags;
    };
    navigator.clipboard
      .writeText(flags)
      .then(() => {
        $('enc-status').textContent = 'copied ✓';
        showFlags();
        setTimeout(() => {
          if ($('enc-status').textContent === 'copied ✓') $('enc-status').textContent = '';
        }, 2000);
      })
      .catch(() => {
        // clipboard blocked (permissions/automation) — show selectable text instead
        $('enc-status').textContent = 'select & copy below';
        showFlags();
      });
  };

  const encodeBtn = $('encode-btn') as HTMLButtonElement;
  async function runEncode(): Promise<boolean> {
    encodeBtn.disabled = true;
    $('enc-status').textContent = 'encoding…';
    try {
      const r = await fetch(`/api/encode?${encodeParams()}`);
      if (!r.ok) throw new Error(`api ${r.status}`);
      const data = (await r.json()) as EncodeResponse;
      if ((data as unknown as { error?: string }).error) throw new Error((data as unknown as { error: string }).error);
      showEncodeStats(data);
      await origLoader.setPermUrl(data.perm);
      loadFile(data.url);
      $('enc-status').textContent = `${(data.wallMs / 1000).toFixed(1)} s`;
      return true;
    } catch (e) {
      $('enc-status').textContent = 'failed';
      console.error(e);
      return false;
    } finally {
      encodeBtn.disabled = false;
    }
  }
  encodeBtn.onclick = () => void runEncode();

  // ================= timeline / transport =================
  function drawTimeline() {
    if (!meta) return;
    const w = timeline.width;
    const h = timeline.height;
    tctx.clearRect(0, 0, w, h);
    tctx.fillStyle = '#1a212a';
    tctx.beginPath();
    tctx.roundRect(0, h / 2 - 7, w, 14, 7);
    tctx.fill();
    const dur = meta.t / meta.fps;
    tctx.fillStyle = '#3a4654';
    meta.gops.forEach((g, i) => {
      if (!buffered[i]) return;
      const x0 = (g.t0 / dur) * w;
      const x1 = ((Math.min(g.f1 + 1, meta!.t) / meta!.fps) / dur) * w;
      tctx.fillRect(x0, h / 2 - 7, x1 - x0, 14);
    });
    tctx.fillStyle = '#4da3ff';
    tctx.beginPath();
    tctx.roundRect(0, h / 2 - 7, Math.max(14, (timeSec / dur) * w), 14, 7);
    tctx.fill();
    const px = (timeSec / dur) * w;
    tctx.fillStyle = '#fff';
    tctx.beginPath();
    tctx.arc(px, h / 2, 9, 0, Math.PI * 2);
    tctx.fill();
    if (waitingGop >= 0) {
      tctx.strokeStyle = '#4da3ff';
      tctx.lineWidth = 3;
      tctx.beginPath();
      tctx.arc(px, h / 2, 13, performance.now() / 200, performance.now() / 200 + 4);
      tctx.stroke();
    }
  }

  function seekTo(clientX: number) {
    if (!meta) return;
    const rect = timeline.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    timeSec = frac * (meta.t / meta.fps);
    requestFrame(Math.min(meta.t - 1, Math.floor(timeSec * meta.fps)));
  }
  let scrubbing = false;
  timeline.addEventListener('pointerdown', (e) => {
    scrubbing = true;
    try {
      timeline.setPointerCapture(e.pointerId);
    } catch { /* synthetic */ }
    seekTo(e.clientX);
  });
  timeline.addEventListener('pointermove', (e) => scrubbing && seekTo(e.clientX));
  timeline.addEventListener('pointerup', () => (scrubbing = false));

  const playBtn = $('play') as HTMLButtonElement;
  function setPlaying(p: boolean) {
    playing = p;
    playBtn.textContent = p ? '⏸' : '▶';
  }
  playBtn.onclick = () => setPlaying(!playing);
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      setPlaying(!playing);
    } else if (e.code === 'KeyC') {
      setCompare(!compareOn);
    }
  });

  function onResize() {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    const pr = Math.min(devicePixelRatio, 2);
    const w = innerWidth * pr;
    const h = innerHeight * pr;
    const fy = camera.projectionMatrix.elements[5] * (h / 2);
    const fx = camera.projectionMatrix.elements[0] * (w / 2);
    splats?.setViewport(w, h, fx, fy);
    origMesh?.setViewport(w, h, fx, fy);
  }
  addEventListener('resize', onResize);

  // ================= boot =================
  const fileOverride = new URLSearchParams(location.search).get('file');
  if (fileOverride) {
    $('panel').style.display = 'none';
    compareBtn.style.display = 'none';
    loadFile(fileOverride);
  } else {
    await loadSequenceList();
    origLoader.reset(`/frames/${currentSeq}`);
    const ok = await runEncode(); // default params; cached after first run
    if (!ok) {
      // no dev API (static hosting) — fall back to a pre-encoded file,
      // optionally described by a demo.json ({file, camera}) next to the page
      $('panel').style.display = 'none';
      compareBtn.style.display = 'none';
      let demoFile = 'juggle.splat4d';
      try {
        const d = await fetch('demo.json');
        if (d.ok) {
          const j = (await d.json()) as { file?: string; camera?: CamHint };
          demoFile = j.file ?? demoFile;
          pendingCamera = j.camera ?? null;
        }
      } catch {
        /* keep defaults */
      }
      loadFile(demoFile);
    }
  }

  // ================= main loop =================
  let lastT = performance.now();
  function loop() {
    requestAnimationFrame(loop);
    const now = performance.now();
    const dt = (now - lastT) / 1000;
    lastT = now;
    controls?.update();

    if (meta && splats) {
      const dur = meta.t / meta.fps;
      if (playing && waitingGop < 0) {
        timeSec += dt;
        if (timeSec >= dur) timeSec -= dur;
      }
      const frame = Math.min(meta.t - 1, Math.floor(timeSec * meta.fps));
      if (frame !== lastShownFrame) requestFrame(frame);
      if (needSort) {
        maybeSort(true);
        needSort = false;
      } else {
        maybeSort(false);
      }
      $('clock').textContent = `${timeSec.toFixed(2)} / ${dur.toFixed(2)}`;
      drawTimeline();
    }

    renderer.render(scene, camera);
    fpsCount++;
    if (now - fpsTime > 1000) {
      $('m-fps').textContent = String(fpsCount);
      fpsCount = 0;
      fpsTime = now;
    }
  }
  loop();
}

init().catch((e) => console.error(e));
