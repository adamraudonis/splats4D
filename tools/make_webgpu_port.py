#!/usr/bin/env python3
"""Generate the faithful WebGPU port of antimatter15/splat.

Takes the reference main.js and splices in a WebGPU rendering layer, keeping
every non-GL line byte-identical (cameras, matrices, worker with CPU-side
covariance packing + 16-bit counting sort, all controls, streaming loader).

Replaced regions (anchored on exact reference source text):
  1. GLSL shader sources        -> WGSL transliteration
  2. WebGL context/setup block  -> WebGPU init (device, pipeline, buffers,
                                   rgba32uint texture, under-blending)
  3. frame-loop draw block      -> WebGPU render pass
  4. selectFile projection call -> port equivalent
"""
from pathlib import Path

SRC = Path("viewer/public/webgl/main.js")
DST = Path("viewer/public/webgpu/main.js")

src = SRC.read_text()

# ---------------------------------------------------------------- 1. shaders
shader_start = src.index("const vertexShaderSource = `")
shader_end = src.index("let defaultViewMatrix")
wgsl = r'''
// WGSL transliteration of the reference GLSL shaders (1:1).
const shaderSource = /* wgsl */ `
struct Uniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    focal : vec2<f32>,
    viewport : vec2<f32>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var u_texture : texture_2d<u32>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) vColor : vec4<f32>,
    @location(1) vPosition : vec2<f32>,
};

@vertex
fn vs(@location(0) position : vec2<f32>, @location(1) index : i32) -> VSOut {
    var out_ : VSOut;
    out_.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out_.vColor = vec4<f32>(0.0);
    out_.vPosition = position;

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
    return out_;
}

@fragment
fn fs(@location(0) vColor : vec4<f32>, @location(1) vPosition : vec2<f32>) -> @location(0) vec4<f32> {
    let A = -dot(vPosition, vPosition);
    if (A < -4.0) { discard; }
    let B = exp(A) * vColor.a;
    return vec4<f32>(B * vColor.rgb, B);
}
`;

'''
src = src[:shader_start] + wgsl + src[shader_end:]

# ------------------------------------------------- 2. context + setup block
gl_start = src.index("    const canvas = document.getElementById(\"canvas\");")
gl_end = src.index("    let activeKeys = [];")
webgpu_setup = r'''    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    // ---------------- WebGPU init (replaces the WebGL2 block) ----------------
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter");
    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: canvasFormat,
        alphaMode: "premultiplied",
    });

    const shaderModule = device.createShaderModule({ code: shaderSource });

    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vs",
            buffers: [
                {
                    arrayStride: 8,
                    stepMode: "vertex",
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                },
                {
                    arrayStride: 4,
                    stepMode: "instance",
                    attributes: [{ shaderLocation: 1, offset: 0, format: "sint32" }],
                },
            ],
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs",
            targets: [
                {
                    format: canvasFormat,
                    // gl.blendFuncSeparate(ONE_MINUS_DST_ALPHA, ONE, ONE_MINUS_DST_ALPHA, ONE)
                    blend: {
                        color: { srcFactor: "one-minus-dst-alpha", dstFactor: "one", operation: "add" },
                        alpha: { srcFactor: "one-minus-dst-alpha", dstFactor: "one", operation: "add" },
                    },
                },
            ],
        },
        primitive: { topology: "triangle-list" },
    });

    // quad: TRIANGLE_FAN [-2,-2, 2,-2, 2,2, -2,2] as two triangles
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = device.createBuffer({
        size: triangleVertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, triangleVertices);
    const quadIndex = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const quadIndexBuffer = device.createBuffer({
        size: quadIndex.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(quadIndexBuffer, 0, quadIndex);

    // uniforms: projection (64) + view (64) + focal (8) + viewport (8)
    const uniformData = new Float32Array(36);
    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let texture = null;
    let bindGroup = null;
    let indexBuffer = null;
    let indexBufferSize = 0;

    const makeBindGroup = () => {
        if (!texture) return;
        bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: texture.createView() },
            ],
        });
    };

    const writeUniforms = (view) => {
        uniformData.set(projectionMatrix, 0);
        uniformData.set(view, 16);
        uniformData[32] = camera.fx;
        uniformData[33] = camera.fy;
        uniformData[34] = innerWidth;
        uniformData[35] = innerHeight;
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    };

    const resize = () => {
        projectionMatrix = getProjectionMatrix(
            camera.fx,
            camera.fy,
            innerWidth,
            innerHeight,
        );
        canvas.width = Math.round(innerWidth / downsample);
        canvas.height = Math.round(innerHeight / downsample);
    };

    window.addEventListener("resize", resize);
    resize();

    worker.onmessage = (e) => {
        if (e.data.buffer) {
            splatData = new Uint8Array(e.data.buffer);
            if (e.data.save) {
                const blob = new Blob([splatData.buffer], {
                    type: "application/octet-stream",
                });
                const link = document.createElement("a");
                link.download = "model.splat";
                link.href = URL.createObjectURL(blob);
                document.body.appendChild(link);
                link.click();
            }
        } else if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            texture = device.createTexture({
                size: [texwidth, texheight],
                format: "rgba32uint",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture },
                texdata.buffer,
                { bytesPerRow: texwidth * 16, rowsPerImage: texheight },
                [texwidth, texheight],
            );
            makeBindGroup();
        } else if (e.data.depthIndex) {
            const { depthIndex, viewProj } = e.data;
            if (!indexBuffer || indexBufferSize < depthIndex.byteLength) {
                if (indexBuffer) indexBuffer.destroy();
                indexBufferSize = depthIndex.byteLength;
                indexBuffer = device.createBuffer({
                    size: indexBufferSize,
                    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                });
            }
            device.queue.writeBuffer(indexBuffer, 0, depthIndex);
            vertexCount = e.data.vertexCount;
        }
    };

'''
src = src[:gl_start] + webgpu_setup + src[gl_end:]

# ---------------------------------------------------- 3. frame-loop draw block
draw_old = """        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }"""
draw_new = """        if (vertexCount > 0 && bindGroup && indexBuffer) {
            document.getElementById("spinner").style.display = "none";
            writeUniforms(actualViewMatrix);
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(),
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.setVertexBuffer(0, vertexBuffer);
            pass.setVertexBuffer(1, indexBuffer);
            pass.setIndexBuffer(quadIndexBuffer, "uint16");
            pass.drawIndexed(6, vertexCount);
            pass.end();
            device.queue.submit([encoder.finish()]);
        } else {
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(),
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            pass.end();
            device.queue.submit([encoder.finish()]);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }"""
assert draw_old in src, "draw block anchor not found"
src = src.replace(draw_old, draw_new)

# ------------------------------------------- 4. selectFile projection update
sel_old = """                projectionMatrix = getProjectionMatrix(
                    camera.fx / downsample,
                    camera.fy / downsample,
                    canvas.width,
                    canvas.height,
                );
                gl.uniformMatrix4fv(u_projection, false, projectionMatrix);"""
sel_new = """                projectionMatrix = getProjectionMatrix(
                    camera.fx / downsample,
                    camera.fy / downsample,
                    canvas.width,
                    canvas.height,
                );"""
assert sel_old in src, "selectFile anchor not found"
src = src.replace(sel_old, sel_new)

# sanity: no gl. references left
leftover = [l for l in src.splitlines() if "gl." in l and "//" not in l.split("gl.")[0]]
for l in leftover:
    print("LEFTOVER GL:", l.strip())

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines), leftover gl refs: {len(leftover)}")
