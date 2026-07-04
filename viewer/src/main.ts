// splat4d viewer on the parity-proven WebGPU pipeline (the faithful port of
// antimatter15/splat — see viewer/public/webgpu + compare.html harness).
// Streaming playback with seek, live re-encode via the dev API, and a
// split-screen compare against the original uncompressed .splat frames.

import { SplatRenderer, SplatSet } from './renderer';
import {
  getProjectionMatrix,
  multiply4,
  viewMatrixFromHint,
  SplatControls,
  type CamHint,
} from './camera';

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
  if (!navigator.gpu) {
    fail('WebGPU is required', 'This viewer has no WebGL fallback. Use Chrome 113+, Edge, Safari 26+, or Firefox 141+.');
  }
  const canvas = document.createElement('canvas');
  canvas.classList.add('webgpu');
  $('canvas-wrap').appendChild(canvas);
  let renderer: SplatRenderer;
  try {
    renderer = await SplatRenderer.create(canvas);
  } catch (e) {
    fail('WebGPU initialization failed', String(e));
  }

  // ---- persistent state ----
  const tPage = performance.now();
  let meta: Meta | null = null;
  let splats: SplatSet | null = null; // compressed set
  let origSet: SplatSet | null = null; // compare set
  let playing = false;
  let timeSec = 0;
  let compareOn = false;
  let dividerFrac = 0.5;
  let fov = 60;
  const bootParams = new URLSearchParams(location.search);
  const bare = bootParams.get('bare') === '1'; // harness mode: no UI, canvas sized like the port
  const pixelRatio = bare ? devicePixelRatio : Math.min(devicePixelRatio, 2);
  if (bare) for (const id of ['hud', 'panel', 'bar']) $(id).style.display = 'none';

  const controls = new SplatControls(canvas, viewMatrixFromHint({ position: [0, -1.2, -2.9], target: [0, -0.9, 0.3] }));

  // ---- per-session state ----
  let session = 0;
  let sessionReady = false;
  let worker: Worker | null = null;
  let buffered: boolean[] = [];
  let lastShownFrame = -1;
  let frameInFlight = false;
  let waitingGop = -1;
  let latestIndices: Uint32Array | null = null;
  let fpsCount = 0;
  let fpsTime = performance.now();

  const timeline = $('timeline') as HTMLCanvasElement;
  const tctx = timeline.getContext('2d')!;

  // ---- original-frame cache for compare (raw bytes, pre-permutation) ----
  let currentSeq = 'juggle_2s';
  const origCache = new Map<number, ArrayBuffer>();
  let origLru: number[] = [];
  let origPermLoaded = false;
  let origShownFrame = -1;

  async function fetchOrigFrame(frame: number): Promise<ArrayBuffer | null> {
    const hit = origCache.get(frame);
    if (hit) return hit;
    try {
      const r = await fetch(`/frames/${currentSeq}/frame_${String(frame).padStart(4, '0')}.splat`);
      if (!r.ok) return null;
      const buf = await r.arrayBuffer();
      origCache.set(frame, buf);
      origLru.push(frame);
      while (origLru.length > 6) origCache.delete(origLru.shift()!);
      return buf;
    } catch {
      return null;
    }
  }

  function refreshOrig(frame: number) {
    if (!compareOn || !worker || !origPermLoaded) return;
    void fetchOrigFrame(frame).then((buf) => {
      if (buf && compareOn && frame === lastShownFrame && worker) {
        // copy: the worker call is repeatable from cache
        const copy = buf.slice(0);
        worker.postMessage({ type: 'origframe', frame, buffer: copy }, [copy]);
      }
    });
  }

  // ---- playback plumbing ----
  function requestFrame(frame: number) {
    if (!meta || !worker || !sessionReady || frameInFlight) return;
    if (frame === lastShownFrame && waitingGop < 0) return;
    frameInFlight = true;
    worker.postMessage({ type: 'frame', frame });
  }

  function ensureSets(n: number) {
    if (splats && splats.count >= 0 && splatsN === n) return;
    splats?.dispose();
    origSet?.dispose();
    origSet = null;
    splats = renderer.createSet();
    splatsN = n;
    if (compareOn) makeOrigSet();
    applyClips();
  }
  let splatsN = -1;

  function makeOrigSet() {
    if (!origSet) origSet = renderer.createSet();
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
    latestIndices = null;
    origShownFrame = -1;

    const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    worker = w;
    w.onmessage = (ev: MessageEvent) => {
      if (mySession !== session) return;
      const m = ev.data;
      if (m.type === 'meta') {
        meta = m as Meta;
        for (let i = 0; i < meta.gops.length; i++) buffered.push(false);
        $('m-total').textContent = (meta.fileSize / 1e6).toFixed(1);
        $('m-splats').textContent = meta.n.toLocaleString();
        $('m-dyn').textContent = `gop ${meta.gop}`;
        $('m-bounds').textContent =
          `±${(meta.bounds.pos_m * 1000).toFixed(1)}mm pos` +
          ` · ${meta.bounds.rgb === 0 ? 'exact' : `±${meta.bounds.rgb}`} color` +
          ` · ${meta.bounds.rot === 0 ? 'exact' : `±${meta.bounds.rot}/128`} rot${meta.denoised ? ' · denoised' : ''}`;
        $('m-ratio').textContent = `${(meta.fileSize / 1e6).toFixed(1)} MB ← ${((meta.n * meta.t * 32) / 1e6).toFixed(0)} MB raw (${((meta.n * meta.t * 32) / meta.fileSize).toFixed(1)}×)`;
        overlaySub.textContent = `${meta.n.toLocaleString()} splats · ${meta.t} frames @ ${meta.fps} fps`;
        if (origPermUrl) {
          void fetch(origPermUrl)
            .then((r) => r.arrayBuffer())
            .then((p) => {
              if (mySession !== session || !worker) return;
              worker.postMessage({ type: 'perm', perm: p }, [p]);
              origPermLoaded = true;
            })
            .catch(() => (origPermLoaded = false));
        }
      } else if (m.type === 'static') {
        if (!meta) return;
        $('m-static').textContent = `${m.staticMs.toFixed(0)} ms`;
        ensureSets(meta.n);
        splats!.uploadTexture(m.texdata, m.texwidth, m.texheight);
        sessionReady = true;
        if (pendingCamera) {
          applyCameraHint(pendingCamera);
          pendingCamera = null;
          wantDefaultCam = false;
        } else if (wantDefaultCam) {
          applyCameraHint(defaultHintFromMeta(meta));
          wantDefaultCam = false;
        }
        overlay.classList.add('hidden');
        if ($('m-ttfv').textContent === '…') $('m-ttfv').textContent = `${(performance.now() - tPage).toFixed(0)} ms`;
        lastShownFrame = -1;
        requestFrame(Math.min(meta.t - 1, Math.floor(timeSec * meta.fps)));
      } else if (m.type === 'frame') {
        if (!splats) return;
        lastShownFrame = m.frame;
        if (!m.approximate) waitingGop = -1;
        const band = new Uint32Array(m.band, 0, 2048 * m.rows * 4);
        splats.uploadTexRows(band, m.rowStart, m.rows);
        w.postMessage({ type: 'return', kind: 'band', buffer: m.band }, [m.band]);
        frameInFlight = false;
        $('m-frame').textContent = String(m.frame);
        $('m-decode').textContent = m.decodeMs.toFixed(1);
        if (compareOn && origShownFrame !== m.frame) refreshOrig(m.frame);
      } else if (m.type === 'miss') {
        frameInFlight = false;
        waitingGop = m.gop;
      } else if (m.type === 'buffered') {
        buffered[m.gop] = true;
        $('m-loaded').textContent = (m.bytesLoaded / 1e6).toFixed(1);
      } else if (m.type === 'sorted') {
        if (!splats) return;
        const idx = new Uint32Array(m.indices, 0, m.count);
        splats.setIndices(idx, m.count);
        latestIndices = idx.slice(0);
        if (origSet && compareOn) origSet.setIndices(latestIndices, m.count);
        w.postMessage({ type: 'return', kind: 'sort', buffer: m.indices }, [m.indices]);
        $('m-sort').textContent = m.sortMs.toFixed(1);
      } else if (m.type === 'origtex') {
        if (origSet && compareOn) {
          const otex = new Uint32Array(m.texdata, 0, m.texwidth * m.texheight * 4);
          origSet.uploadTexture(otex, m.texwidth, m.texheight);
          if (latestIndices) origSet.setIndices(latestIndices, latestIndices.length);
          origShownFrame = m.frame;
        }
        w.postMessage({ type: 'return', kind: 'origtex', buffer: m.texdata }, [m.texdata]);
      } else if (m.type === 'dump') {
        (window as unknown as Record<string, unknown>).__dumpResult = m.stats;
      } else if (m.type === 'error') {
        fail('Stream error', m.message);
      }
    };
    w.postMessage({ type: 'load', url: new URL(url, location.href).href });
  }

  // ---- camera ----
  let pendingCamera: CamHint | null = null;
  let wantDefaultCam = true; // reset camera to a scene-default pose when a sequence has no hint
  const seqCameras = new Map<string, CamHint | null>();
  let origPermUrl: string | null = null;

  function applyCameraHint(hint: CamHint | null) {
    if (!hint) return;
    controls.viewMatrix = viewMatrixFromHint(hint);
    fov = hint.fov ?? 60;
  }

  function defaultHintFromMeta(m: Meta): CamHint {
    const [ax, , az, bx, , bz] = m.aabb;
    const originInside = ax < 0 && bx > 0 && az < 0 && bz > 0;
    const tx = originInside ? 0 : (ax + bx) / 2;
    const ty = originInside ? -0.9 : (m.aabb[1] + m.aabb[4]) / 2;
    const tz = originInside ? 0.3 : (az + bz) / 2;
    return { position: [tx, ty - 0.3, tz - 2.9], target: [tx, ty, tz], fov: 60 };
  }

  // ---- compare mode ----
  const divider = $('divider');
  const compareBtn = $('compare-btn') as HTMLButtonElement;

  function applyClips() {
    const ndc = dividerFrac * 2 - 1;
    if (compareOn && splats && origSet) {
      origSet.setClip(ndc, -1);
      splats.setClip(ndc, 1);
      divider.style.display = 'block';
      divider.style.left = `${dividerFrac * 100}%`;
    } else {
      splats?.setClip(0, 0);
      origSet?.setClip(0, 0);
      divider.style.display = 'none';
    }
  }

  function setCompare(on: boolean) {
    if (!splats || !meta) return;
    if (on && !origPermLoaded) {
      $('enc-status').textContent = 'compare needs the dev API';
      return;
    }
    compareOn = on;
    compareBtn.classList.toggle('on', on);
    if (on) {
      makeOrigSet();
      origSet!.visible = true;
      origShownFrame = -1;
      refreshOrig(Math.max(0, lastShownFrame));
    } else if (origSet) {
      origSet.visible = false;
    }
    applyClips();
  }
  compareBtn.onclick = () => setCompare(!compareOn);

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

  // ---- sequences + encode panel ----
  const seqSelect = $('s-seq') as HTMLSelectElement;

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
      const preferred = sequences.find((s) => s.id === 'flame_2s') ?? sequences[0];
      currentSeq = preferred.id;
      seqSelect.value = currentSeq;
      pendingCamera = seqCameras.get(currentSeq) ?? null;
    } catch {
      /* API absent (static hosting) */
    }
  }

  seqSelect.onchange = () => {
    currentSeq = seqSelect.value;
    origCache.clear();
    origLru = [];
    origPermLoaded = false;
    timeSec = 0;
    playing = false;
    playBtn.textContent = '▶';
    pendingCamera = seqCameras.get(currentSeq) ?? null;
    wantDefaultCam = true;
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
    $('o-col').textContent = sliders.col.value === '0' ? 'exact' : `±${sliders.col.value}/255`;
    $('o-rot').textContent = sliders.rot.value === '0' ? 'exact' : `±${sliders.rot.value}/128`;
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
      origPermUrl = data.perm;
      origPermLoaded = false;
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

  // ---- timeline / transport ----
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
    canvas.width = Math.round(innerWidth * pixelRatio);
    canvas.height = Math.round(innerHeight * pixelRatio);
  }
  addEventListener('resize', onResize);
  onResize();

  // ---- debug hooks ----
  (window as unknown as Record<string, unknown>).__cam = (
    px: number, py: number, pz: number, qx: number, qy: number, qz: number
  ) => applyCameraHint({ position: [px, py, pz], target: [qx, qy, qz], fov });
  (window as unknown as Record<string, unknown>).__play = (p: boolean) => setPlaying(p);
  (window as unknown as Record<string, unknown>).__compare = (on: boolean) => setCompare(on);
  (window as unknown as Record<string, unknown>).__vm = () => controls.viewMatrix.slice(0);
  (window as unknown as Record<string, unknown>).__setvm = (m: number[]) => (controls.viewMatrix = m);
  let focalOverride: [number, number] | null = null;
  (window as unknown as Record<string, unknown>).__setfocal = (fx: number, fy: number) => (focalOverride = [fx, fy]);
  (window as unknown as Record<string, unknown>).__frameShown = () => lastShownFrame;
  (window as unknown as Record<string, unknown>).__dump = () => worker?.postMessage({ type: 'dump' });
  (window as unknown as Record<string, unknown>).__seek = (f: number) => {
    if (!meta) return;
    setPlaying(false);
    timeSec = (f + 0.5) / meta.fps;
    requestFrame(f);
  };

  // ---- boot ----
  const fileOverride = bootParams.get('file');
  if (fileOverride) {
    $('panel').style.display = 'none';
    compareBtn.style.display = 'none';
    loadFile(fileOverride);
  } else {
    await loadSequenceList();
    const ok = await runEncode();
    if (!ok) {
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
      } catch { /* keep defaults */ }
      loadFile(demoFile);
    }
  }

  // ---- main loop ----
  let lastT = performance.now();
  function loop() {
    requestAnimationFrame(loop);
    const now = performance.now();
    const dt = (now - lastT) / 1000;
    lastT = now;
    controls.update();

    if (meta && splats && sessionReady) {
      const dur = meta.t / meta.fps;
      if (playing && waitingGop < 0) {
        // clamp: after a hidden-tab stall dt can be many seconds
        timeSec = (timeSec + Math.min(dt, 0.1)) % dur;
      }
      const frame = Math.min(meta.t - 1, Math.floor(timeSec * meta.fps));
      if (frame !== lastShownFrame) requestFrame(frame);

      const fy = focalOverride ? focalOverride[1] : (0.5 * innerHeight) / Math.tan((fov * Math.PI) / 360);
      const fx = focalOverride ? focalOverride[0] : fy;
      const proj = getProjectionMatrix(fx, fy, innerWidth, innerHeight);
      const viewProj = multiply4(proj, controls.viewMatrix);
      worker?.postMessage({ type: 'view', viewProj: new Float32Array(viewProj) });

      splats.setCamera(proj, controls.viewMatrix, fx, fy, innerWidth, innerHeight);
      if (origSet && compareOn) origSet.setCamera(proj, controls.viewMatrix, fx, fy, innerWidth, innerHeight);

      $('clock').textContent = `${timeSec.toFixed(2)} / ${dur.toFixed(2)}`;
      drawTimeline();
      renderer.render(compareOn && origSet ? [origSet, splats] : [splats]);
    } else {
      renderer.render([]);
    }
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
