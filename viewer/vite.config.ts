import { defineConfig, type Plugin } from 'vite';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';

const execFileP = promisify(execFile);

const REPO = path.resolve(__dirname, '..');
const CONVERTER = path.join(REPO, 'converter/target/release/splat4d');
const FRAMES_ROOT = path.join(REPO, 'data/frames');
const WEB_OUT = path.join(REPO, 'data/out/web');

function listSequences() {
  try {
    return fs
      .readdirSync(FRAMES_ROOT)
      .filter((d) => /^[a-z0-9_]+$/.test(d) && fs.existsSync(path.join(FRAMES_ROOT, d, 'frames.json')))
      .map((d) => {
        const m = JSON.parse(fs.readFileSync(path.join(FRAMES_ROOT, d, 'frames.json'), 'utf8'));
        return {
          id: d,
          frames: m.count ?? m.frames.length,
          fps: m.fps,
          splats: m.num_splats ?? 0,
          camera: m.camera ?? null,
        };
      })
      .sort((a, b) => a.id.localeCompare(b.id));
  } catch {
    return [];
  }
}

function seqDir(seq: string): string | null {
  if (!/^[a-z0-9_]+$/.test(seq)) return null;
  const dir = path.join(FRAMES_ROOT, seq);
  return fs.existsSync(path.join(dir, 'frames.json')) ? dir : null;
}

function num(v: string | null, def: number, lo: number, hi: number): number {
  const n = v === null ? def : parseFloat(v);
  if (!isFinite(n)) return def;
  return Math.min(hi, Math.max(lo, n));
}

function serveFile(req: IncomingMessage, res: ServerResponse, file: string) {
  let st: fs.Stats;
  try {
    st = fs.statSync(file);
  } catch {
    res.statusCode = 404;
    res.end('not found');
    return;
  }
  const range = req.headers.range;
  let start = 0;
  let end = st.size - 1;
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', 'application/octet-stream');
  // dev server: files are regenerated in place at identical URLs — never let
  // the browser serve stale splat data from its heuristic cache
  res.setHeader('Cache-Control', 'no-store');
  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range);
    if (m) {
      if (m[1]) start = parseInt(m[1], 10);
      if (m[2]) end = Math.min(parseInt(m[2], 10), st.size - 1);
      res.statusCode = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${st.size}`);
    }
  }
  res.setHeader('Content-Length', end - start + 1);
  fs.createReadStream(file, { start, end }).pipe(res);
}

/** Dev API: /api/encode re-runs the Rust converter with slider params (cached);
 *  /api/file/* serves encoder outputs; /frames/* serves original .splat frames. */
function splat4dDevApi(): Plugin {
  let queue: Promise<unknown> = Promise.resolve();
  return {
    name: 'splat4d-dev-api',
    configureServer(server) {
      fs.mkdirSync(WEB_OUT, { recursive: true });
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://x');

        if (url.pathname === '/api/sequences') {
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ sequences: listSequences() }));
          return;
        }

        if (url.pathname === '/api/encode') {
          const seq = url.searchParams.get('seq') ?? 'juggle_2s';
          const framesDir = seqDir(seq);
          if (!framesDir) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: `unknown sequence ${seq}` }));
            return;
          }
          const p = {
            pos_mm: num(url.searchParams.get('pos_mm'), 2, 0.1, 50),
            color_levels: Math.round(num(url.searchParams.get('color_levels'), 4, 0, 32)),
            rot_steps: Math.round(num(url.searchParams.get('rot_steps'), 0, 0, 8)),
            scale_pct: num(url.searchParams.get('scale_pct'), 2, 0.1, 25),
            gop: Math.round(num(url.searchParams.get('gop'), 30, 5, 300)),
            denoise: url.searchParams.get('denoise') === '1',
            zstd: Math.round(num(url.searchParams.get('zstd'), 3, 0, 19)),
          };
          const key = `${seq}_p${p.pos_mm}_c${p.color_levels}_r${p.rot_steps}_s${p.scale_pct}_g${p.gop}_d${p.denoise ? 1 : 0}_z${p.zstd}`
            .replace(/\./g, '-');
          const out = path.join(WEB_OUT, `${key}.splat4d`);
          const reportPath = path.join(WEB_OUT, `${key}.json`);
          const permPath = path.join(WEB_OUT, `${key}.perm`);

          // serialize encodes; cache by param key
          queue = queue.then(async () => {
            const t0 = Date.now();
            let cached = true;
            if (!fs.existsSync(out) || !fs.existsSync(reportPath) || !fs.existsSync(permPath)) {
              cached = false;
              await execFileP(
                CONVERTER,
                [
                  'encode',
                  '-i', framesDir,
                  '-o', out,
                  '--pos-mm', String(p.pos_mm),
                  '--color-levels', String(p.color_levels),
                  '--rot-steps', String(p.rot_steps),
                  '--scale-pct', String(p.scale_pct),
                  '--gop', String(p.gop),
                  '--zstd-level', String(p.zstd),
                  ...(p.denoise ? ['--denoise-colors'] : []),
                  '--report', reportPath,
                  '--perm-out', permPath,
                ],
                { timeout: 180000, maxBuffer: 16 << 20 }
              );
            }
            const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
            res.setHeader('Content-Type', 'application/json');
            res.end(
              JSON.stringify({
                url: `/api/file/${key}.splat4d`,
                perm: `/api/file/${key}.perm`,
                cached,
                wallMs: Date.now() - t0,
                params: p,
                report,
              })
            );
          }).catch((err) => {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          });
          return;
        }

        if (url.pathname.startsWith('/api/file/')) {
          const name = path.basename(url.pathname);
          serveFile(req, res, path.join(WEB_OUT, name));
          return;
        }

        if (url.pathname.startsWith('/frames/')) {
          const parts = url.pathname.split('/').filter(Boolean); // ['frames', seq, file]
          const dir = parts.length === 3 ? seqDir(parts[1]) : null;
          const name = parts.length === 3 ? parts[2] : '';
          if (!dir || !/^frame_\d+\.splat$|^frames\.json$/.test(name)) {
            res.statusCode = 403;
            res.end();
            return;
          }
          serveFile(req, res, path.join(dir, name));
          return;
        }

        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [splat4dDevApi()],
});
