// Gaussian splat rendering on three.js WebGPURenderer via TSL.
// One instanced draw of a 4-vertex quad; per-instance attributes live in
// storage buffers; the vertex stage does the EWA 3D->2D covariance projection
// (math from antimatter15/splat main.js, MIT) and emits clip-space corners.
//
// cov2d = A Σ Aᵀ with A = J·W (Jacobian × view rotation) is computed via
// dot products (a0·Σa0 etc.) — no matrix element indexing needed.

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  float,
  vec2,
  vec3,
  vec4,
  mat3,
  uniform,
  instancedArray,
  instanceIndex,
  varying,
  positionGeometry,
  cameraProjectionMatrix,
  modelViewMatrix,
  exp,
  sqrt,
  min,
  normalize,
  uint,
  dot,
  select,
} from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface SplatMesh {
  mesh: THREE.Mesh;
  posArr: Float32Array;
  scaleArr: Float32Array;
  quatArr: Uint32Array;
  rgbaArr: Uint32Array;
  sortArr: Uint32Array;
  sortNode: unknown; // instancedArray node, shareable with a second mesh
  markPos(): void;
  markScale(): void;
  markQuat(): void;
  markRgba(): void;
  markSort(): void;
  setViewport(w: number, h: number, focalX: number, focalY: number): void;
  /** compare-mode clip: NDC x divider; side -1 keeps x<divider, +1 keeps x>divider, 0 disables */
  setClip(divider: number, side: number): void;
  dispose(scene: THREE.Scene): void;
}

export interface SplatMeshOptions {
  /** share another mesh's sort order (index spaces must match) */
  sharedSort?: { sortArr: Uint32Array; sortNode: unknown };
}

export function createSplatMesh(n: number, opts: SplatMeshOptions = {}): SplatMesh {
  const posArr = new Float32Array(n * 4);
  const scaleArr = new Float32Array(n * 4);
  const quatArr = new Uint32Array(n);
  const rgbaArr = new Uint32Array(n);
  const sortArr = opts.sharedSort ? opts.sharedSort.sortArr : new Uint32Array(n);
  if (!opts.sharedSort) for (let i = 0; i < n; i++) sortArr[i] = i;

  const posBuf = instancedArray(posArr, 'vec4');
  const scaleBuf = instancedArray(scaleArr, 'vec4');
  const quatBuf = instancedArray(quatArr, 'uint');
  const rgbaBuf = instancedArray(rgbaArr, 'uint');
  const sortBuf = (opts.sharedSort ? opts.sharedSort.sortNode : instancedArray(sortArr, 'uint')) as ReturnType<
    typeof instancedArray
  >;

  const viewport = uniform(new THREE.Vector2(1, 1));
  const focal = uniform(new THREE.Vector2(1, 1));
  // compare clip: x = NDC divider, y = side (-1 keep left, +1 keep right, 0 off)
  const clipCfg = uniform(new THREE.Vector2(0, 0));

  const vColor = varying(vec4(0), 'vColor');
  const vPos = varying(vec2(0), 'vPos');
  const vNdcX = varying(float(0), 'vNdcX');

  const u8 = (v: any, shift: number) => float(v.shiftRight(uint(shift)).bitAnd(uint(0xff)));
  const m3 = mat3 as any; // typings lack the (vec3, vec3, vec3) column overload
  const v3 = vec3 as any; // typings mis-infer dot() results as vec3

  const vertexNode = Fn(() => {
    const splatId = sortBuf.element(instanceIndex);
    const center = (posBuf.element(splatId) as any).xyz;
    const scale = (scaleBuf.element(splatId) as any).xyz;
    const qp = quatBuf.element(splatId);
    const cp = rgbaBuf.element(splatId);

    const out = vec4(0, 0, 2, 1).toVar(); // default: clipped away
    const cam = (modelViewMatrix.mul(vec4(center, 1.0)) as any).toVar();
    const clip = (cameraProjectionMatrix.mul(cam) as any).toVar();

    // cull behind camera / far outside frustum
    const ndcZ = clip.z.div(clip.w);
    const lim = clip.w.mul(1.3);
    const inside = clip.w
      .greaterThan(0.0)
      .and(clip.x.abs().lessThan(lim))
      .and(clip.y.abs().lessThan(lim));

    If(inside, () => {
      // quaternion (w,x,y,z) from packed u8s
      const q = normalize(
        vec4(
          u8(qp, 0).sub(128.0),
          u8(qp, 8).sub(128.0),
          u8(qp, 16).sub(128.0),
          u8(qp, 24).sub(128.0)
        )
      ).toVar();
      const w = (q as any).x,
        x = (q as any).y,
        y = (q as any).z,
        z = (q as any).w;

      // R(q) columns
      const R = m3(
        v3(
          float(1.0).sub(float(2.0).mul(y.mul(y).add(z.mul(z)))),
          float(2.0).mul(x.mul(y).add(w.mul(z))),
          float(2.0).mul(x.mul(z).sub(w.mul(y)))
        ),
        v3(
          float(2.0).mul(x.mul(y).sub(w.mul(z))),
          float(1.0).sub(float(2.0).mul(x.mul(x).add(z.mul(z)))),
          float(2.0).mul(y.mul(z).add(w.mul(x)))
        ),
        v3(
          float(2.0).mul(x.mul(z).add(w.mul(y))),
          float(2.0).mul(y.mul(z).sub(w.mul(x))),
          float(1.0).sub(float(2.0).mul(x.mul(x).add(y.mul(y))))
        )
      );
      const S = m3(
        vec3((scale as any).x, 0, 0),
        v3(0, (scale as any).y, 0),
        v3(0, 0, (scale as any).z)
      );
      const M = (R as any).mul(S);
      const Vrk = M.mul(M.transpose()).toVar(); // Σ = R S² Rᵀ

      // view rotation columns (world -> view)
      const vc0 = (modelViewMatrix.mul(vec4(1, 0, 0, 0)) as any).xyz;
      const vc1 = (modelViewMatrix.mul(vec4(0, 1, 0, 0)) as any).xyz;
      const vc2 = (modelViewMatrix.mul(vec4(0, 0, 1, 0)) as any).xyz;

      // Jacobian rows (with screen-y flip, antimatter15 convention)
      const invZ = float(1.0).div(cam.z);
      const r0 = vec3(
        focal.x.mul(invZ),
        0.0,
        cam.x.negate().mul(focal.x).mul(invZ).mul(invZ)
      );
      const r1 = vec3(
        0.0,
        focal.y.negate().mul(invZ),
        cam.y.mul(focal.y).mul(invZ).mul(invZ)
      );

      // rows of A = J·W:  a_i = Wᵀ r_i  →  (a_i)_k = dot(view column k, r_i)
      const a0 = v3(dot(vc0, r0), dot(vc1, r0), dot(vc2, r0)).toVar();
      const a1 = v3(dot(vc0, r1), dot(vc1, r1), dot(vc2, r1)).toVar();

      // cov2d entries + small low-pass dilation (reference viewers use ~0.075
      // in true-covariance units; larger values visibly soften small splats)
      const c00 = dot(a0, (Vrk as any).mul(a0)).add(0.075).toVar();
      const c11 = dot(a1, (Vrk as any).mul(a1)).add(0.075).toVar();
      const c01 = dot(a0, (Vrk as any).mul(a1)).toVar();

      const mid = c00.add(c11).mul(0.5).toVar();
      const det = c00.mul(c11).sub(c01.mul(c01));
      const radius = sqrt(mid.mul(mid).sub(det).max(0.0001)).toVar();
      const lambda1 = mid.add(radius).toVar();
      const lambda2 = mid.sub(radius).toVar();

      If(lambda2.greaterThan(0.0), () => {
        const diag = normalize(vec2(c01, lambda1.sub(c00))).toVar();
        const major = min(sqrt(lambda1.mul(2.0)), 1024.0).mul(diag).toVar();
        const minor = min(sqrt(lambda2.mul(2.0)), 1024.0)
          .mul(vec2((diag as any).y, (diag as any).x.negate()))
          .toVar();

        const corner = positionGeometry.xy; // (±2, ±2)
        const centerNdc = clip.xy.div(clip.w);
        const ndc = centerNdc
          .add(corner.x.mul(major).mul(2.0).div(viewport))
          .add(corner.y.mul(minor).mul(2.0).div(viewport));

        vColor.assign(
          vec4(u8(cp, 0).div(255.0), u8(cp, 8).div(255.0), u8(cp, 16).div(255.0), u8(cp, 24).div(255.0))
        );
        vPos.assign(corner);
        vNdcX.assign((ndc as any).x);
        out.assign(vec4((ndc as any).x, (ndc as any).y, ndcZ, 1.0));
      });
    });
    return out;
  })();

  const colorNode = Fn(() => {
    // compare-mode split: keep only the configured side of the divider
    const keep = clipCfg.y
      .equal(0.0)
      .or(vNdcX.sub(clipCfg.x).mul(clipCfg.y).greaterThan(0.0));
    // quad param ±2 maps to ±2.83σ, so the gaussian is exp(-d²) with a hard
    // cutoff at d²=4 — matching the reference antimatter15/3DGS rasterizers.
    // (exp(-d²/2) here would render every splat √2× too wide.)
    const d2 = dot(vPos, vPos);
    const a = exp(d2.negate())
      .mul((vColor as any).w)
      .mul(select(keep.and(d2.lessThan(4.0)), float(1.0), float(0.0)));
    return vec4((vColor as any).xyz.mul(a), a);
  })();

  const material = new THREE.MeshBasicNodeMaterial();
  material.vertexNode = vertexNode as any;
  material.fragmentNode = colorNode as any;
  material.transparent = true;
  material.depthWrite = false;
  material.depthTest = false;
  material.blending = THREE.CustomBlending;
  material.blendEquation = THREE.AddEquation;
  material.blendSrc = THREE.OneFactor;
  material.blendDst = THREE.OneMinusSrcAlphaFactor;
  material.blendSrcAlpha = THREE.OneFactor;
  material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  material.side = THREE.DoubleSide;

  const geometry = new THREE.InstancedBufferGeometry();
  const corners = new Float32Array([-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0]);
  geometry.setAttribute('position', new THREE.BufferAttribute(corners, 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  geometry.instanceCount = n;

  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;

  const attrOf = (buf: any) => buf.value as THREE.BufferAttribute;

  return {
    mesh,
    posArr,
    scaleArr,
    quatArr,
    rgbaArr,
    sortArr,
    sortNode: sortBuf,
    markPos: () => (attrOf(posBuf).needsUpdate = true),
    markScale: () => (attrOf(scaleBuf).needsUpdate = true),
    markQuat: () => (attrOf(quatBuf).needsUpdate = true),
    markRgba: () => (attrOf(rgbaBuf).needsUpdate = true),
    markSort: () => (attrOf(sortBuf).needsUpdate = true),
    setViewport: (w, h, fx, fy) => {
      (viewport.value as THREE.Vector2).set(w, h);
      (focal.value as THREE.Vector2).set(fx, fy);
    },
    setClip: (divider, side) => {
      (clipCfg.value as THREE.Vector2).set(divider, side);
    },
    dispose: (scene) => {
      scene.remove(mesh);
      geometry.dispose();
      material.dispose();
    },
  };
}
