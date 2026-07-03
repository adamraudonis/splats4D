#!/usr/bin/env node
// Bandwidth-throttled static file server with HTTP Range support + CORS.
// For honest streaming/seek latency measurements of .splat4d files.
//
// Usage: node tools/serve_throttled.mjs <dir> [port] [mbps]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const dir = process.argv[2] ?? '.';
const port = parseInt(process.argv[3] ?? '8901', 10);
const mbps = parseFloat(process.argv[4] ?? '50');
const bytesPerSec = (mbps * 1e6) / 8;
const CHUNK = 64 * 1024;

const server = http.createServer((req, res) => {
  const file = path.join(dir, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(path.resolve(dir)) && !file.startsWith(dir)) {
    res.writeHead(403).end();
    return;
  }
  let st;
  try {
    st = fs.statSync(file);
  } catch {
    res.writeHead(404).end('not found');
    return;
  }
  const range = req.headers.range;
  let start = 0;
  let end = st.size - 1;
  const headers = {
    'Accept-Ranges': 'bytes',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Expose-Headers': 'Content-Range, Content-Length',
    'Content-Type': 'application/octet-stream',
  };
  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range);
    if (m) {
      if (m[1]) start = parseInt(m[1], 10);
      if (m[2]) end = parseInt(m[2], 10);
      end = Math.min(end, st.size - 1);
      headers['Content-Range'] = `bytes ${start}-${end}/${st.size}`;
      headers['Content-Length'] = end - start + 1;
      res.writeHead(206, headers);
    }
  } else {
    headers['Content-Length'] = st.size;
    res.writeHead(200, headers);
  }

  const fd = fs.openSync(file, 'r');
  let pos = start;
  const t0 = Date.now();
  let sent = 0;
  const pump = () => {
    if (pos > end) {
      fs.closeSync(fd);
      res.end();
      return;
    }
    // token bucket: how many bytes are we allowed to have sent by now?
    const allowed = ((Date.now() - t0) / 1000) * bytesPerSec + CHUNK;
    if (sent >= allowed) {
      setTimeout(pump, 10);
      return;
    }
    const len = Math.min(CHUNK, end - pos + 1);
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, pos);
    pos += len;
    sent += len;
    if (res.write(buf)) pump();
    else res.once('drain', pump);
  };
  pump();
});

server.listen(port, () => {
  console.log(`throttled server: ${dir} on :${port} at ${mbps} Mbps`);
});
