// WebGL2 fallback renderer — the antimatter15/splat GL layer (shaders and
// state verbatim) behind the same interface as the WebGPU SplatRenderer, so
// the worker/streaming pipeline is renderer-agnostic. Consumes the identical
// rgba32ui texture layout, sorted index buffer, and camera uniforms.
// Same additive clip extension as the WGSL port (u_clip.y = 0 -> inert).

export const TEXWIDTH = 2048;

const VS = `#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;
uniform vec4 u_clip;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;
out float vNdcX;

void main () {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec4(0.0);
    vPosition = position;
    vNdcX = 0.0;

    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport
        + position.y * minorAxis / viewport, 0.0, 1.0);
    vNdcX = gl_Position.x;
}`;

const FS = `#version 300 es
precision highp float;

uniform vec4 u_clip;

in vec4 vColor;
in vec2 vPosition;
in float vNdcX;

out vec4 fragColor;

void main () {
    if (u_clip.y != 0.0 && (vNdcX - u_clip.x) * u_clip.y < 0.0) discard;
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}`;

export class GlSplatSet {
  count = 0;
  visible = true;
  texWidth = 0;
  texHeight = 0;
  private texture: WebGLTexture | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private proj = new Float32Array(16);
  private view = new Float32Array(16);
  private focal = new Float32Array(2);
  private viewport = new Float32Array(2);
  private clip = new Float32Array(4);
  private r: GlSplatRenderer;

  constructor(r: GlSplatRenderer) {
    this.r = r;
  }

  private ensureTexture(texwidth: number, texheight: number) {
    const gl = this.r.gl;
    if (!this.texture || this.texWidth !== texwidth || this.texHeight !== texheight) {
      if (this.texture) gl.deleteTexture(this.texture);
      this.texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, texwidth, texheight);
      this.texWidth = texwidth;
      this.texHeight = texheight;
    } else {
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
    }
  }

  uploadTexture(texdata: Uint32Array, texwidth: number, texheight: number) {
    const gl = this.r.gl;
    this.ensureTexture(texwidth, texheight);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texwidth, texheight, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
  }

  /** update rows [yStart, yStart+rows) from a band buffer */
  uploadTexRows(band: Uint32Array, yStart: number, rows: number) {
    if (!this.texture) return;
    const gl = this.r.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, yStart, this.texWidth, rows, gl.RGBA_INTEGER, gl.UNSIGNED_INT, band);
  }

  setIndices(indices: Uint32Array, count: number) {
    const gl = this.r.gl;
    if (!this.vao) {
      this.vao = gl.createVertexArray();
      this.indexBuffer = gl.createBuffer();
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.r.vertexBuffer);
      gl.enableVertexAttribArray(this.r.aPosition);
      gl.vertexAttribPointer(this.r.aPosition, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
      gl.enableVertexAttribArray(this.r.aIndex);
      gl.vertexAttribIPointer(this.r.aIndex, 1, gl.INT, 0, 0);
      gl.vertexAttribDivisor(this.r.aIndex, 1);
      gl.bindVertexArray(null);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    this.count = count;
  }

  setCamera(projection: ArrayLike<number>, view: ArrayLike<number>, fx: number, fy: number, vw: number, vh: number) {
    this.proj.set(projection as number[]);
    this.view.set(view as number[]);
    this.focal[0] = fx;
    this.focal[1] = fy;
    this.viewport[0] = vw;
    this.viewport[1] = vh;
  }

  setClip(divider: number, side: number) {
    this.clip[0] = divider;
    this.clip[1] = side;
  }

  /** bind + set uniforms + draw (single shared program) */
  draw() {
    if (!this.vao || !this.texture || this.count === 0) return;
    const gl = this.r.gl;
    gl.uniformMatrix4fv(this.r.uProjection, false, this.proj);
    gl.uniformMatrix4fv(this.r.uView, false, this.view);
    gl.uniform2fv(this.r.uFocal, this.focal);
    gl.uniform2fv(this.r.uViewport, this.viewport);
    gl.uniform4fv(this.r.uClip, this.clip);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.count);
    gl.bindVertexArray(null);
  }

  dispose() {
    const gl = this.r.gl;
    if (this.texture) gl.deleteTexture(this.texture);
    if (this.indexBuffer) gl.deleteBuffer(this.indexBuffer);
    if (this.vao) gl.deleteVertexArray(this.vao);
    this.texture = null;
    this.indexBuffer = null;
    this.vao = null;
  }
}

export class GlSplatRenderer {
  readonly backend = 'webgl2';
  gl!: WebGL2RenderingContext;
  vertexBuffer!: WebGLBuffer;
  aPosition!: number;
  aIndex!: number;
  uProjection!: WebGLUniformLocation;
  uView!: WebGLUniformLocation;
  uFocal!: WebGLUniformLocation;
  uViewport!: WebGLUniformLocation;
  uClip!: WebGLUniformLocation;
  private canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  static async create(canvas: HTMLCanvasElement): Promise<GlSplatRenderer> {
    const r = new GlSplatRenderer(canvas);
    const gl = canvas.getContext('webgl2', { antialias: false });
    if (!gl) throw new Error('WebGL2 not supported');
    r.gl = gl;

    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error('shader compile: ' + gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error('program link: ' + gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
    gl.clearColor(0, 0, 0, 0);

    r.uProjection = gl.getUniformLocation(program, 'projection')!;
    r.uView = gl.getUniformLocation(program, 'view')!;
    r.uFocal = gl.getUniformLocation(program, 'focal')!;
    r.uViewport = gl.getUniformLocation(program, 'viewport')!;
    r.uClip = gl.getUniformLocation(program, 'u_clip')!;
    gl.uniform1i(gl.getUniformLocation(program, 'u_texture'), 0);

    r.vertexBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, r.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    r.aPosition = gl.getAttribLocation(program, 'position');
    r.aIndex = gl.getAttribLocation(program, 'index');
    return r;
  }

  createSet(): GlSplatSet {
    return new GlSplatSet(this);
  }

  render(sets: GlSplatSet[]) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    for (const s of sets) {
      if (s.visible) s.draw();
    }
  }
}
