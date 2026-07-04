// Raw-WebGPU splat renderer — the parity-proven port of antimatter15/splat
// (see viewer/public/webgpu, verified pixel-identical to the WebGL reference).
// Identical WGSL, rgba32uint texture layout, under-blending, draw order.
// One additive extension, inert unless enabled: a compare-mode NDC clip
// (clip.y = 0 disables it and the shader output is bit-identical to the port).

export const TEXWIDTH = 2048;

const SHADER = /* wgsl */ `
struct Uniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    focal : vec2<f32>,
    viewport : vec2<f32>,
    clip : vec4<f32>, // x = ndc divider, y = side (-1 keep left, +1 keep right, 0 off)
};
@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var u_texture : texture_2d<u32>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) vColor : vec4<f32>,
    @location(1) vPosition : vec2<f32>,
    @location(2) vNdcX : f32,
};

@vertex
fn vs(@location(0) position : vec2<f32>, @location(1) index : i32) -> VSOut {
    var out_ : VSOut;
    out_.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out_.vColor = vec4<f32>(0.0);
    out_.vPosition = position;
    out_.vNdcX = 0.0;

    let idx = u32(index);
    let cen = textureLoad(u_texture, vec2<i32>(i32((idx & 0x3ffu) << 1u), i32(idx >> 10u)), 0);
    let cam = u.view * vec4<f32>(bitcast<vec3<f32>>(cen.xyz), 1.0);
    let pos2d = u.projection * cam;

    let clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        return out_;
    }

    let cov = textureLoad(u_texture, vec2<i32>(i32(((idx & 0x3ffu) << 1u) | 1u), i32(idx >> 10u)), 0);
    let u1 = unpack2x16float(cov.x); let u2 = unpack2x16float(cov.y); let u3 = unpack2x16float(cov.z);
    let Vrk = mat3x3<f32>(
        vec3<f32>(u1.x, u1.y, u2.x),
        vec3<f32>(u1.y, u2.y, u3.x),
        vec3<f32>(u2.x, u3.x, u3.y));

    let J = mat3x3<f32>(
        vec3<f32>(u.focal.x / cam.z, 0.0, -(u.focal.x * cam.x) / (cam.z * cam.z)),
        vec3<f32>(0.0, -u.focal.y / cam.z, (u.focal.y * cam.y) / (cam.z * cam.z)),
        vec3<f32>(0.0, 0.0, 0.0));

    let viewR = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    let T = transpose(viewR) * J;
    let cov2d = transpose(T) * Vrk * T;

    let mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    let radius = length(vec2<f32>((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    let lambda1 = mid + radius; let lambda2 = mid - radius;

    if (lambda2 < 0.0) { return out_; }
    let diagonalVector = normalize(vec2<f32>(cov2d[0][1], lambda1 - cov2d[0][0]));
    let majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    let minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    out_.vColor = clamp(pos2d.z / pos2d.w + 1.0, 0.0, 1.0) * vec4<f32>(
        f32(cov.w & 0xffu),
        f32((cov.w >> 8u) & 0xffu),
        f32((cov.w >> 16u) & 0xffu),
        f32((cov.w >> 24u) & 0xffu)) / 255.0;

    let vCenter = pos2d.xy / pos2d.w;
    out_.pos = vec4<f32>(
        vCenter
        + position.x * majorAxis / u.viewport
        + position.y * minorAxis / u.viewport, 0.0, 1.0);
    out_.vNdcX = out_.pos.x;
    return out_;
}

@fragment
fn fs(@location(0) vColor : vec4<f32>, @location(1) vPosition : vec2<f32>, @location(2) vNdcX : f32) -> @location(0) vec4<f32> {
    if (u.clip.y != 0.0 && (vNdcX - u.clip.x) * u.clip.y < 0.0) { discard; }
    let A = -dot(vPosition, vPosition);
    if (A < -4.0) { discard; }
    let B = exp(A) * vColor.a;
    return vec4<f32>(B * vColor.rgb, B);
}
`;

/** One renderable splat collection: its data texture, sorted index buffer,
 *  and per-set uniforms (camera shared by copy, clip per set). */
export class SplatSet {
  count = 0;
  texture: GPUTexture | null = null;
  texWidth = 0;
  texHeight = 0;
  indexBuffer: GPUBuffer | null = null;
  private indexCapacity = 0;
  bindGroup: GPUBindGroup | null = null;
  uniformBuffer: GPUBuffer;
  uniformData = new Float32Array(40); // proj 16 + view 16 + focal 2 + viewport 2 + clip 4
  visible = true;
  private r: SplatRenderer;

  constructor(r: SplatRenderer) {
    this.r = r;
    this.uniformBuffer = r.device.createBuffer({
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  uploadTexture(texdata: Uint32Array, texwidth: number, texheight: number) {
    if (!this.texture || this.texWidth !== texwidth || this.texHeight !== texheight) {
      this.texture?.destroy();
      this.texture = this.r.device.createTexture({
        size: [texwidth, texheight],
        format: 'rgba32uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.texWidth = texwidth;
      this.texHeight = texheight;
      this.bindGroup = this.r.device.createBindGroup({
        layout: this.r.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: this.texture.createView() },
        ],
      });
    }
    this.r.device.queue.writeTexture(
      { texture: this.texture },
      texdata.buffer as ArrayBuffer,
      { offset: texdata.byteOffset, bytesPerRow: texwidth * 16, rowsPerImage: texheight },
      [texwidth, texheight]
    );
  }

  /** update rows [yStart, yStart+rows) from a band buffer */
  uploadTexRows(band: Uint32Array, yStart: number, rows: number) {
    if (!this.texture) return;
    this.r.device.queue.writeTexture(
      { texture: this.texture, origin: [0, yStart] },
      band.buffer as ArrayBuffer,
      { offset: band.byteOffset, bytesPerRow: this.texWidth * 16, rowsPerImage: rows },
      [this.texWidth, rows]
    );
  }

  setIndices(indices: Uint32Array, count: number) {
    if (!this.indexBuffer || this.indexCapacity < indices.byteLength) {
      this.indexBuffer?.destroy();
      this.indexCapacity = indices.byteLength;
      this.indexBuffer = this.r.device.createBuffer({
        size: this.indexCapacity,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.r.device.queue.writeBuffer(this.indexBuffer, 0, indices);
    this.count = count;
  }

  setCamera(projection: ArrayLike<number>, view: ArrayLike<number>, fx: number, fy: number, vw: number, vh: number) {
    this.uniformData.set(projection as number[], 0);
    this.uniformData.set(view as number[], 16);
    this.uniformData[32] = fx;
    this.uniformData[33] = fy;
    this.uniformData[34] = vw;
    this.uniformData[35] = vh;
    this.r.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
  }

  setClip(divider: number, side: number) {
    this.uniformData[36] = divider;
    this.uniformData[37] = side;
  }

  dispose() {
    this.texture?.destroy();
    this.indexBuffer?.destroy();
    this.uniformBuffer.destroy();
    this.texture = null;
    this.indexBuffer = null;
    this.bindGroup = null;
  }
}

export class SplatRenderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  pipeline!: GPURenderPipeline;
  private vertexBuffer!: GPUBuffer;
  private quadIndexBuffer!: GPUBuffer;

  static async create(canvas: HTMLCanvasElement): Promise<SplatRenderer> {
    const r = new SplatRenderer();
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    r.device = await adapter.requestDevice();
    r.context = canvas.getContext('webgpu')!;
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    r.context.configure({ device: r.device, format: canvasFormat, alphaMode: 'premultiplied' });

    const module = r.device.createShaderModule({ code: SHADER });
    r.pipeline = r.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          { arrayStride: 8, stepMode: 'vertex', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
          { arrayStride: 4, stepMode: 'instance', attributes: [{ shaderLocation: 1, offset: 0, format: 'sint32' }] },
        ],
      },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [
          {
            format: canvasFormat,
            blend: {
              color: { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    const verts = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    r.vertexBuffer = r.device.createBuffer({
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    r.device.queue.writeBuffer(r.vertexBuffer, 0, verts);
    const qi = new Uint16Array([0, 1, 2, 0, 2, 3]);
    r.quadIndexBuffer = r.device.createBuffer({
      size: qi.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    r.device.queue.writeBuffer(r.quadIndexBuffer, 0, qi);
    return r;
  }

  createSet(): SplatSet {
    return new SplatSet(this);
  }

  render(sets: SplatSet[]) {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    for (const s of sets) {
      if (!s.visible || !s.bindGroup || !s.indexBuffer || s.count === 0) continue;
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, s.bindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setVertexBuffer(1, s.indexBuffer);
      pass.setIndexBuffer(this.quadIndexBuffer, 'uint16');
      pass.drawIndexed(6, s.count);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
