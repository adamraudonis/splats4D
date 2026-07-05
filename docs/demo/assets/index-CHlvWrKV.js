(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin===`use-credentials`?t.credentials=`include`:e.crossOrigin===`anonymous`?t.credentials=`omit`:t.credentials=`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var e=`
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
`,t=class{count=0;texture=null;texWidth=0;texHeight=0;indexBuffer=null;indexCapacity=0;bindGroup=null;uniformBuffer;uniformData=new Float32Array(40);visible=!0;r;constructor(e){this.r=e,this.uniformBuffer=e.device.createBuffer({size:this.uniformData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}uploadTexture(e,t,n){(!this.texture||this.texWidth!==t||this.texHeight!==n)&&(this.texture?.destroy(),this.texture=this.r.device.createTexture({size:[t,n],format:`rgba32uint`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),this.texWidth=t,this.texHeight=n,this.bindGroup=this.r.device.createBindGroup({layout:this.r.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:this.texture.createView()}]})),this.r.device.queue.writeTexture({texture:this.texture},e.buffer,{offset:e.byteOffset,bytesPerRow:t*16,rowsPerImage:n},[t,n])}uploadTexRows(e,t,n){this.texture&&this.r.device.queue.writeTexture({texture:this.texture,origin:[0,t]},e.buffer,{offset:e.byteOffset,bytesPerRow:this.texWidth*16,rowsPerImage:n},[this.texWidth,n])}setIndices(e,t){(!this.indexBuffer||this.indexCapacity<e.byteLength)&&(this.indexBuffer?.destroy(),this.indexCapacity=e.byteLength,this.indexBuffer=this.r.device.createBuffer({size:this.indexCapacity,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST})),this.r.device.queue.writeBuffer(this.indexBuffer,0,e),this.count=t}setCamera(e,t,n,r,i,a){this.uniformData.set(e,0),this.uniformData.set(t,16),this.uniformData[32]=n,this.uniformData[33]=r,this.uniformData[34]=i,this.uniformData[35]=a,this.r.device.queue.writeBuffer(this.uniformBuffer,0,this.uniformData)}setClip(e,t){this.uniformData[36]=e,this.uniformData[37]=t}dispose(){this.texture?.destroy(),this.indexBuffer?.destroy(),this.uniformBuffer.destroy(),this.texture=null,this.indexBuffer=null,this.bindGroup=null}},n=class n{backend=`webgpu`;device;context;pipeline;vertexBuffer;quadIndexBuffer;static async create(t){let r=new n;if(!navigator.gpu)throw Error(`WebGPU not supported`);let i=await navigator.gpu.requestAdapter();if(!i)throw Error(`No WebGPU adapter`);r.device=await i.requestDevice(),r.context=t.getContext(`webgpu`);let a=navigator.gpu.getPreferredCanvasFormat();r.context.configure({device:r.device,format:a,alphaMode:`premultiplied`});let o=r.device.createShaderModule({code:e});r.pipeline=r.device.createRenderPipeline({layout:`auto`,vertex:{module:o,entryPoint:`vs`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:4,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`sint32`}]}]},fragment:{module:o,entryPoint:`fs`,targets:[{format:a,blend:{color:{srcFactor:`one-minus-dst-alpha`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one-minus-dst-alpha`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}});let s=new Float32Array([-2,-2,2,-2,2,2,-2,2]);r.vertexBuffer=r.device.createBuffer({size:s.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),r.device.queue.writeBuffer(r.vertexBuffer,0,s);let c=new Uint16Array([0,1,2,0,2,3]);return r.quadIndexBuffer=r.device.createBuffer({size:c.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),r.device.queue.writeBuffer(r.quadIndexBuffer,0,c),r}createSet(){return new t(this)}render(e){let t=this.device.createCommandEncoder(),n=t.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});for(let t of e)!t.visible||!t.bindGroup||!t.indexBuffer||t.count===0||(n.setPipeline(this.pipeline),n.setBindGroup(0,t.bindGroup),n.setVertexBuffer(0,this.vertexBuffer),n.setVertexBuffer(1,t.indexBuffer),n.setIndexBuffer(this.quadIndexBuffer,`uint16`),n.drawIndexed(6,t.count));n.end(),this.device.queue.submit([t.finish()])}},r=`#version 300 es
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
}`,i=`#version 300 es
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
}`,a=class{count=0;visible=!0;texWidth=0;texHeight=0;texture=null;indexBuffer=null;vao=null;proj=new Float32Array(16);view=new Float32Array(16);focal=new Float32Array(2);viewport=new Float32Array(2);clip=new Float32Array(4);r;constructor(e){this.r=e}ensureTexture(e,t){let n=this.r.gl;!this.texture||this.texWidth!==e||this.texHeight!==t?(this.texture&&n.deleteTexture(this.texture),this.texture=n.createTexture(),n.bindTexture(n.TEXTURE_2D,this.texture),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MAG_FILTER,n.NEAREST),n.texStorage2D(n.TEXTURE_2D,1,n.RGBA32UI,e,t),this.texWidth=e,this.texHeight=t):n.bindTexture(n.TEXTURE_2D,this.texture)}uploadTexture(e,t,n){let r=this.r.gl;this.ensureTexture(t,n),r.texSubImage2D(r.TEXTURE_2D,0,0,0,t,n,r.RGBA_INTEGER,r.UNSIGNED_INT,e)}uploadTexRows(e,t,n){if(!this.texture)return;let r=this.r.gl;r.bindTexture(r.TEXTURE_2D,this.texture),r.texSubImage2D(r.TEXTURE_2D,0,0,t,this.texWidth,n,r.RGBA_INTEGER,r.UNSIGNED_INT,e)}setIndices(e,t){let n=this.r.gl;this.vao||(this.vao=n.createVertexArray(),this.indexBuffer=n.createBuffer(),n.bindVertexArray(this.vao),n.bindBuffer(n.ARRAY_BUFFER,this.r.vertexBuffer),n.enableVertexAttribArray(this.r.aPosition),n.vertexAttribPointer(this.r.aPosition,2,n.FLOAT,!1,0,0),n.bindBuffer(n.ARRAY_BUFFER,this.indexBuffer),n.enableVertexAttribArray(this.r.aIndex),n.vertexAttribIPointer(this.r.aIndex,1,n.INT,0,0),n.vertexAttribDivisor(this.r.aIndex,1),n.bindVertexArray(null)),n.bindBuffer(n.ARRAY_BUFFER,this.indexBuffer),n.bufferData(n.ARRAY_BUFFER,e,n.DYNAMIC_DRAW),this.count=t}setCamera(e,t,n,r,i,a){this.proj.set(e),this.view.set(t),this.focal[0]=n,this.focal[1]=r,this.viewport[0]=i,this.viewport[1]=a}setClip(e,t){this.clip[0]=e,this.clip[1]=t}draw(){if(!this.vao||!this.texture||this.count===0)return;let e=this.r.gl;e.uniformMatrix4fv(this.r.uProjection,!1,this.proj),e.uniformMatrix4fv(this.r.uView,!1,this.view),e.uniform2fv(this.r.uFocal,this.focal),e.uniform2fv(this.r.uViewport,this.viewport),e.uniform4fv(this.r.uClip,this.clip),e.activeTexture(e.TEXTURE0),e.bindTexture(e.TEXTURE_2D,this.texture),e.bindVertexArray(this.vao),e.drawArraysInstanced(e.TRIANGLE_FAN,0,4,this.count),e.bindVertexArray(null)}dispose(){let e=this.r.gl;this.texture&&e.deleteTexture(this.texture),this.indexBuffer&&e.deleteBuffer(this.indexBuffer),this.vao&&e.deleteVertexArray(this.vao),this.texture=null,this.indexBuffer=null,this.vao=null}},o=class e{backend=`webgl2`;gl;vertexBuffer;aPosition;aIndex;uProjection;uView;uFocal;uViewport;uClip;canvas;constructor(e){this.canvas=e}static async create(t){let n=new e(t),a=t.getContext(`webgl2`,{antialias:!1});if(!a)throw Error(`WebGL2 not supported`);n.gl=a;let o=(e,t)=>{let n=a.createShader(e);if(a.shaderSource(n,t),a.compileShader(n),!a.getShaderParameter(n,a.COMPILE_STATUS))throw Error(`shader compile: `+a.getShaderInfoLog(n));return n},s=a.createProgram();if(a.attachShader(s,o(a.VERTEX_SHADER,r)),a.attachShader(s,o(a.FRAGMENT_SHADER,i)),a.linkProgram(s),!a.getProgramParameter(s,a.LINK_STATUS))throw Error(`program link: `+a.getProgramInfoLog(s));return a.useProgram(s),a.disable(a.DEPTH_TEST),a.enable(a.BLEND),a.blendFuncSeparate(a.ONE_MINUS_DST_ALPHA,a.ONE,a.ONE_MINUS_DST_ALPHA,a.ONE),a.blendEquationSeparate(a.FUNC_ADD,a.FUNC_ADD),a.clearColor(0,0,0,0),n.uProjection=a.getUniformLocation(s,`projection`),n.uView=a.getUniformLocation(s,`view`),n.uFocal=a.getUniformLocation(s,`focal`),n.uViewport=a.getUniformLocation(s,`viewport`),n.uClip=a.getUniformLocation(s,`u_clip`),a.uniform1i(a.getUniformLocation(s,`u_texture`),0),n.vertexBuffer=a.createBuffer(),a.bindBuffer(a.ARRAY_BUFFER,n.vertexBuffer),a.bufferData(a.ARRAY_BUFFER,new Float32Array([-2,-2,2,-2,2,2,-2,2]),a.STATIC_DRAW),n.aPosition=a.getAttribLocation(s,`position`),n.aIndex=a.getAttribLocation(s,`index`),n}createSet(){return new a(this)}render(e){let t=this.gl;t.viewport(0,0,this.canvas.width,this.canvas.height),t.clear(t.COLOR_BUFFER_BIT);for(let t of e)t.visible&&t.draw()}};function s(e,t,n,r){let i=.2;return[[2*e/n,0,0,0],[0,-(2*t)/r,0,0],[0,0,200/(200-i),1],[0,0,-40/(200-i),0]].flat()}function c(e,t){return[t[0]*e[0]+t[1]*e[4]+t[2]*e[8]+t[3]*e[12],t[0]*e[1]+t[1]*e[5]+t[2]*e[9]+t[3]*e[13],t[0]*e[2]+t[1]*e[6]+t[2]*e[10]+t[3]*e[14],t[0]*e[3]+t[1]*e[7]+t[2]*e[11]+t[3]*e[15],t[4]*e[0]+t[5]*e[4]+t[6]*e[8]+t[7]*e[12],t[4]*e[1]+t[5]*e[5]+t[6]*e[9]+t[7]*e[13],t[4]*e[2]+t[5]*e[6]+t[6]*e[10]+t[7]*e[14],t[4]*e[3]+t[5]*e[7]+t[6]*e[11]+t[7]*e[15],t[8]*e[0]+t[9]*e[4]+t[10]*e[8]+t[11]*e[12],t[8]*e[1]+t[9]*e[5]+t[10]*e[9]+t[11]*e[13],t[8]*e[2]+t[9]*e[6]+t[10]*e[10]+t[11]*e[14],t[8]*e[3]+t[9]*e[7]+t[10]*e[11]+t[11]*e[15],t[12]*e[0]+t[13]*e[4]+t[14]*e[8]+t[15]*e[12],t[12]*e[1]+t[13]*e[5]+t[14]*e[9]+t[15]*e[13],t[12]*e[2]+t[13]*e[6]+t[14]*e[10]+t[15]*e[14],t[12]*e[3]+t[13]*e[7]+t[14]*e[11]+t[15]*e[15]]}function l(e){let t=e[0]*e[5]-e[1]*e[4],n=e[0]*e[6]-e[2]*e[4],r=e[0]*e[7]-e[3]*e[4],i=e[1]*e[6]-e[2]*e[5],a=e[1]*e[7]-e[3]*e[5],o=e[2]*e[7]-e[3]*e[6],s=e[8]*e[13]-e[9]*e[12],c=e[8]*e[14]-e[10]*e[12],l=e[8]*e[15]-e[11]*e[12],u=e[9]*e[14]-e[10]*e[13],d=e[9]*e[15]-e[11]*e[13],f=e[10]*e[15]-e[11]*e[14],p=t*f-n*d+r*u+i*l-a*c+o*s;return p?[(e[5]*f-e[6]*d+e[7]*u)/p,(e[2]*d-e[1]*f-e[3]*u)/p,(e[13]*o-e[14]*a+e[15]*i)/p,(e[10]*a-e[9]*o-e[11]*i)/p,(e[6]*l-e[4]*f-e[7]*c)/p,(e[0]*f-e[2]*l+e[3]*c)/p,(e[14]*r-e[12]*o-e[15]*n)/p,(e[8]*o-e[10]*r+e[11]*n)/p,(e[4]*d-e[5]*l+e[7]*s)/p,(e[1]*l-e[0]*d-e[3]*s)/p,(e[12]*a-e[13]*r+e[15]*t)/p,(e[9]*r-e[8]*a-e[11]*t)/p,(e[5]*c-e[4]*u-e[6]*s)/p,(e[0]*u-e[1]*c+e[2]*s)/p,(e[13]*n-e[12]*i-e[14]*t)/p,(e[8]*i-e[9]*n+e[10]*t)/p]:null}function u(e,t,n,r,i){let a=Math.hypot(n,r,i);n/=a,r/=a,i/=a;let o=Math.sin(t),s=Math.cos(t),c=1-s,l=n*n*c+s,u=r*n*c+i*o,d=i*n*c-r*o,f=n*r*c-i*o,p=r*r*c+s,m=i*r*c+n*o,h=n*i*c+r*o,g=r*i*c-n*o,_=i*i*c+s;return[e[0]*l+e[4]*u+e[8]*d,e[1]*l+e[5]*u+e[9]*d,e[2]*l+e[6]*u+e[10]*d,e[3]*l+e[7]*u+e[11]*d,e[0]*f+e[4]*p+e[8]*m,e[1]*f+e[5]*p+e[9]*m,e[2]*f+e[6]*p+e[10]*m,e[3]*f+e[7]*p+e[11]*m,e[0]*h+e[4]*g+e[8]*_,e[1]*h+e[5]*g+e[9]*_,e[2]*h+e[6]*g+e[10]*_,e[3]*h+e[7]*g+e[11]*_,...e.slice(12,16)]}function d(e,t,n,r){return[...e.slice(0,12),e[0]*t+e[4]*n+e[8]*r+e[12],e[1]*t+e[5]*n+e[9]*r+e[13],e[2]*t+e[6]*n+e[10]*r+e[14],e[3]*t+e[7]*n+e[11]*r+e[15]]}function f(e){let t=e.position,n=e.target,r=(e,t)=>e.map((e,n)=>e-t[n]),i=e=>{let t=Math.hypot(e[0],e[1],e[2]);return e.map(e=>e/t)},a=(e,t)=>[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]],o=(e,t)=>e[0]*t[0]+e[1]*t[1]+e[2]*t[2],s=i(r(n,t)),c=i(a([0,1,0],s)),l=a(s,c);return[c[0],l[0],s[0],0,c[1],l[1],s[1],0,c[2],l[2],s[2],0,-o(t,c),-o(t,l),-o(t,s),1]}var p=class{viewMatrix;activeKeys=[];down=!1;startX=0;startY=0;constructor(e,t){this.viewMatrix=t,window.addEventListener(`keydown`,e=>{document.activeElement!==document.body&&document.activeElement!==null&&[`INPUT`,`SELECT`,`TEXTAREA`,`BUTTON`].includes(document.activeElement.tagName)||this.activeKeys.includes(e.code)||this.activeKeys.push(e.code)}),window.addEventListener(`keyup`,e=>{this.activeKeys=this.activeKeys.filter(t=>t!==e.code)}),window.addEventListener(`blur`,()=>this.activeKeys=[]),e.addEventListener(`wheel`,e=>{e.preventDefault();let t=e.deltaMode===1?10:e.deltaMode===2?innerHeight:1,n=l(this.viewMatrix);e.shiftKey?n=d(n,e.deltaX*t/innerWidth,e.deltaY*t/innerHeight,0):e.ctrlKey||e.metaKey?n=d(n,0,0,-10*(e.deltaY*t)/innerHeight):(n=d(n,0,0,4),n=u(n,-(e.deltaX*t)/innerWidth,0,1,0),n=u(n,e.deltaY*t/innerHeight,1,0,0),n=d(n,0,0,-4)),this.viewMatrix=l(n)},{passive:!1}),e.addEventListener(`mousedown`,e=>{e.preventDefault(),this.startX=e.clientX,this.startY=e.clientY,this.down=e.ctrlKey||e.metaKey?2:1}),e.addEventListener(`contextmenu`,e=>{e.preventDefault(),this.startX=e.clientX,this.startY=e.clientY,this.down=2}),e.addEventListener(`mousemove`,e=>{if(e.preventDefault(),this.down===1){let t=l(this.viewMatrix),n=5*(e.clientX-this.startX)/innerWidth,r=5*(e.clientY-this.startY)/innerHeight;t=d(t,0,0,4),t=u(t,n,0,1,0),t=u(t,-r,1,0,0),t=d(t,0,0,-4),this.viewMatrix=l(t),this.startX=e.clientX,this.startY=e.clientY}else if(this.down===2){let t=l(this.viewMatrix);t=d(t,-10*(e.clientX-this.startX)/innerWidth,0,10*(e.clientY-this.startY)/innerHeight),this.viewMatrix=l(t),this.startX=e.clientX,this.startY=e.clientY}}),e.addEventListener(`mouseup`,e=>{e.preventDefault(),this.down=!1});let n=0,r=0;e.addEventListener(`touchstart`,e=>{e.preventDefault(),e.touches.length===1?(this.startX=e.touches[0].clientX,this.startY=e.touches[0].clientY,this.down=1):e.touches.length===2&&(this.startX=e.touches[0].clientX,n=e.touches[1].clientX,this.startY=e.touches[0].clientY,r=e.touches[1].clientY,this.down=1)},{passive:!1}),e.addEventListener(`touchmove`,e=>{if(e.preventDefault(),e.touches.length===1&&this.down){let t=l(this.viewMatrix),n=4*(e.touches[0].clientX-this.startX)/innerWidth,r=4*(e.touches[0].clientY-this.startY)/innerHeight;t=d(t,0,0,4),t=u(t,n,0,1,0),t=u(t,-r,1,0,0),t=d(t,0,0,-4),this.viewMatrix=l(t),this.startX=e.touches[0].clientX,this.startY=e.touches[0].clientY}else if(e.touches.length===2){let t=Math.atan2(this.startY-r,this.startX-n)-Math.atan2(e.touches[0].clientY-e.touches[1].clientY,e.touches[0].clientX-e.touches[1].clientX),i=Math.hypot(this.startX-n,this.startY-r)/Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY),a=(e.touches[0].clientX+e.touches[1].clientX-(this.startX+n))/2,o=(e.touches[0].clientY+e.touches[1].clientY-(this.startY+r))/2,s=l(this.viewMatrix);s=u(s,t,0,0,1),s=d(s,-a/innerWidth,-o/innerHeight,0),s=d(s,0,0,3*(1-i)),this.viewMatrix=l(s),this.startX=e.touches[0].clientX,this.startY=e.touches[0].clientY,n=e.touches[1].clientX,r=e.touches[1].clientY}},{passive:!1}),e.addEventListener(`touchend`,e=>{e.preventDefault(),this.down=!1,this.startX=0,this.startY=0},{passive:!1})}update(){let e=l(this.viewMatrix),t=!1;this.activeKeys.includes(`ArrowUp`)&&(e=d(e,0,0,.1),t=!0),this.activeKeys.includes(`ArrowDown`)&&(e=d(e,0,0,-.1),t=!0),this.activeKeys.includes(`ArrowLeft`)&&(e=d(e,-.03,0,0),t=!0),this.activeKeys.includes(`ArrowRight`)&&(e=d(e,.03,0,0),t=!0),[`KeyJ`,`KeyK`,`KeyL`,`KeyI`].some(e=>this.activeKeys.includes(e))&&(e=d(e,0,0,4),e=u(e,this.activeKeys.includes(`KeyJ`)?-.05:this.activeKeys.includes(`KeyL`)?.05:0,0,1,0),e=u(e,this.activeKeys.includes(`KeyI`)?.05:this.activeKeys.includes(`KeyK`)?-.05:0,1,0,0),e=d(e,0,0,-4),t=!0),this.activeKeys.includes(`KeyA`)&&(e=d(e,-.03,0,0),t=!0),this.activeKeys.includes(`KeyD`)&&(e=d(e,.03,0,0),t=!0),this.activeKeys.includes(`KeyW`)&&(e=u(e,.005,1,0,0),t=!0),this.activeKeys.includes(`KeyS`)&&(e=u(e,-.005,1,0,0),t=!0),t&&(this.viewMatrix=l(e))}},m=e=>document.getElementById(e),h=m(`overlay`),g=m(`overlay-msg`),_=m(`overlay-sub`);function ee(e,t=``){throw h.classList.remove(`hidden`),h.querySelector(`.spin`)?.remove(),g.textContent=e,_.textContent=t,Error(e)}async function v(){let e=document.createElement(`canvas`);e.classList.add(`webgpu`),m(`canvas-wrap`).appendChild(e);let t=new URLSearchParams(location.search).get(`gl`)===`1`,r=null;if(!t&&navigator.gpu)try{r=await n.create(e)}catch(e){console.warn(`WebGPU init failed, falling back to WebGL2:`,e)}if(!r)try{r=await o.create(e)}catch(e){ee(`WebGPU or WebGL2 required`,`Neither WebGPU nor WebGL2 could be initialized in this browser. `+String(e))}let i=r,a=performance.now(),l=null,u=null,d=null,g=!1,v=0,y=!1,b=.5,x=60,S=new URLSearchParams(location.search),te=S.get(`bare`)===`1`,ne=S.get(`embed`)===`1`,re=te?devicePixelRatio:Math.min(devicePixelRatio,2);if(te)for(let e of[`hud`,`panel`,`bar`])m(e).style.display=`none`;if(ne){for(let e of[`hud`,`panel`])m(e).style.display=`none`;document.documentElement.classList.add(`embed`)}let C=new p(e,f({position:[0,-1.2,-2.9],target:[0,-.9,.3]})),ie=!1,w=0,T=!1,E=null,D=[],O=-1,k=!1,A=-1,j=null,ae=0,oe=performance.now(),M=m(`timeline`),N=M.getContext(`2d`),P=`juggle_2s`,F=new Map,I=[],L=!1,R=-1;async function se(e){let t=F.get(e);if(t)return t;try{let t=await fetch(`/frames/${P}/frame_${String(e).padStart(4,`0`)}.splat`);if(!t.ok)return null;let n=await t.arrayBuffer();for(F.set(e,n),I.push(e);I.length>6;)F.delete(I.shift());return n}catch{return null}}function ce(e){!y||!E||!L||se(e).then(t=>{if(t&&y&&e===O&&E){let n=t.slice(0);E.postMessage({type:`origframe`,frame:e,buffer:n},[n])}})}function z(e){!l||!E||!T||k||e===O&&A<0||(k=!0,E.postMessage({type:`frame`,frame:e}))}function le(e){u&&u.count>=0&&ue===e||(u?.dispose(),d?.dispose(),d=null,u=i.createSet(),ue=e,y&&de(),J())}let ue=-1;function de(){d||=i.createSet()}function B(e){w++;let t=w;E?.terminate(),T=!1,D=[],O=-1,k=!1,A=-1,j=null,R=-1;let n=new Worker(new URL(``+new URL(`worker-DaxXKHdp.js`,import.meta.url).href,``+import.meta.url),{type:`module`});E=n,n.onmessage=e=>{if(t!==w)return;let r=e.data;if(r.type===`meta`){l=r;for(let e=0;e<l.gops.length;e++)D.push(!1);m(`m-total`).textContent=(l.fileSize/1e6).toFixed(1),m(`m-splats`).textContent=l.n.toLocaleString(),m(`m-dyn`).textContent=`gop ${l.gop} · ${i.backend===`webgl2`?`WebGL2`:`WebGPU`}`,m(`m-bounds`).textContent=`±${(l.bounds.pos_m*1e3).toFixed(1)}mm pos · ${l.bounds.rgb===0?`exact`:`±${l.bounds.rgb}`} color · ${l.bounds.rot===0?`exact`:`±${l.bounds.rot}/128`} rot${l.denoised?` · denoised`:``}`,m(`m-ratio`).textContent=`${(l.fileSize/1e6).toFixed(1)} MB ← ${(l.n*l.t*32/1e6).toFixed(0)} MB raw (${(l.n*l.t*32/l.fileSize).toFixed(1)}×)`,_.textContent=`${l.n.toLocaleString()} splats · ${l.t} frames @ ${l.fps} fps`,W&&fetch(W).then(e=>e.arrayBuffer()).then(e=>{t!==w||!E||(E.postMessage({type:`perm`,perm:e},[e]),L=!0)}).catch(()=>L=!1)}else if(r.type===`static`){if(!l)return;m(`m-static`).textContent=`${r.staticMs.toFixed(0)} ms`,le(l.n),u.uploadTexture(r.texdata,r.texwidth,r.texheight),T=!0,V?(G(V),V=null,H=!1):H&&=(G(fe(l)),!1),h.classList.add(`hidden`),m(`m-ttfv`).textContent===`…`&&(m(`m-ttfv`).textContent=`${(performance.now()-a).toFixed(0)} ms`),O=-1,z(Math.min(l.t-1,Math.floor(v*l.fps)))}else if(r.type===`frame`){if(!u)return;ne&&!ie&&(ie=!0,Q(!0)),O=r.frame,r.approximate||(A=-1);let e=new Uint32Array(r.band,0,2048*r.rows*4);u.uploadTexRows(e,r.rowStart,r.rows),n.postMessage({type:`return`,kind:`band`,buffer:r.band},[r.band]),k=!1,m(`m-frame`).textContent=String(r.frame),m(`m-decode`).textContent=r.decodeMs.toFixed(1),y&&R!==r.frame&&ce(r.frame)}else if(r.type===`miss`)k=!1,A=r.gop;else if(r.type===`buffered`)D[r.gop]=!0,m(`m-loaded`).textContent=(r.bytesLoaded/1e6).toFixed(1);else if(r.type===`sorted`){if(!u)return;let e=new Uint32Array(r.indices,0,r.count);u.setIndices(e,r.count),j=e.slice(0),d&&y&&d.setIndices(j,r.count),n.postMessage({type:`return`,kind:`sort`,buffer:r.indices},[r.indices]),m(`m-sort`).textContent=r.sortMs.toFixed(1)}else if(r.type===`origtex`){if(d&&y){let e=new Uint32Array(r.texdata,0,r.texwidth*r.texheight*4);d.uploadTexture(e,r.texwidth,r.texheight),j&&d.setIndices(j,j.length),R=r.frame}n.postMessage({type:`return`,kind:`origtex`,buffer:r.texdata},[r.texdata])}else r.type===`dump`?window.__dumpResult=r.stats:r.type===`error`&&ee(`Stream error`,r.message)},n.postMessage({type:`load`,url:new URL(e,location.href).href})}let V=null,H=!0,U=new Map,W=null;function G(e){e&&(C.viewMatrix=f(e),x=e.fov??60)}function fe(e){let[t,,n,r,,i]=e.aabb,a=t<0&&r>0&&n<0&&i>0,o=a?0:(t+r)/2,s=a?-.9:(e.aabb[1]+e.aabb[4])/2,c=a?.3:(n+i)/2;return{position:[o,s-.3,c-2.9],target:[o,s,c],fov:60}}let K=m(`divider`),q=m(`compare-btn`);function J(){let e=b*2-1;y&&u&&d?(d.setClip(e,-1),u.setClip(e,1),K.style.display=`block`,K.style.left=`${b*100}%`):(u?.setClip(0,0),d?.setClip(0,0),K.style.display=`none`)}function pe(e){if(!(!u||!l)){if(e&&!L){m(`enc-status`).textContent=`compare needs the dev API`;return}y=e,q.classList.toggle(`on`,e),e?(de(),d.visible=!0,R=-1,ce(Math.max(0,O))):d&&(d.visible=!1),J()}}q.onclick=()=>pe(!y);{let e=K.querySelector(`.grip`),t=!1;e.addEventListener(`pointerdown`,n=>{t=!0;try{e.setPointerCapture(n.pointerId)}catch{}}),addEventListener(`pointermove`,e=>{t&&(b=Math.min(.98,Math.max(.02,e.clientX/innerWidth)),J())}),addEventListener(`pointerup`,()=>t=!1)}let Y=m(`s-seq`);async function me(){try{let e=await fetch(`/api/sequences`);if(!e.ok)return;let{sequences:t}=await e.json();if(!t.length)return;Y.innerHTML=``;for(let e of t){let t=document.createElement(`option`);t.value=e.id;let n=(e.frames/e.fps).toFixed(1);t.textContent=`${e.id.replace(/_2s$/,``)} · ${n}s · ${(e.splats/1e3).toFixed(0)}k`,Y.appendChild(t),U.set(e.id,e.camera)}P=(t.find(e=>e.id===`flame_2s`)??t[0]).id,Y.value=P,V=U.get(P)??null}catch{}}Y.onchange=()=>{P=Y.value,F.clear(),I=[],L=!1,v=0,g=!1,Z.textContent=`▶`,V=U.get(P)??null,H=!0,be()};let X={pos:m(`s-pos`),col:m(`s-col`),rot:m(`s-rot`),scl:m(`s-scl`),gop:m(`s-gop`),dn:m(`s-dn`),z:m(`s-z`)},he=()=>{m(`o-pos`).textContent=`±${X.pos.value} mm`,m(`o-col`).textContent=X.col.value===`0`?`exact`:`±${X.col.value}/255`,m(`o-rot`).textContent=X.rot.value===`0`?`exact`:`±${X.rot.value}/128`,m(`o-scl`).textContent=`±${X.scl.value}%`,m(`o-gop`).textContent=`${X.gop.value} fr`};for(let e of[X.pos,X.col,X.rot,X.scl,X.gop])e.addEventListener(`input`,he);he();function ge(){return new URLSearchParams({seq:P,pos_mm:X.pos.value,color_levels:X.col.value,rot_steps:X.rot.value,scale_pct:X.scl.value,gop:X.gop.value,denoise:X.dn.checked?`1`:`0`,zstd:X.z.value}).toString()}function _e(e){let t=e.report,n=t.verify,r=t.static_fracs;m(`enc-stats`).innerHTML=`<div><b>${(t.output.bytes/1e6).toFixed(1)} MB</b> · <b>${t.output.ratio.toFixed(1)}×</b> smaller · encoded in ${t.times_s.total.toFixed(1)} s${e.cached?` (cached)`:``}</div><div class="s">static: pos ${(r.pos*100).toFixed(0)}% rot ${(r.rot*100).toFixed(0)}% color ${(r.rgb*100).toFixed(0)}%</div>`+(n?`<div class="s">verified ✓ max err: ${n.pos_mm.toFixed(2)}mm · ${n.rgb_levels}/255 · ${n.rot_units}/128</div>`:``)+(t.denoise?`<div class="s">denoise dev: mean ${t.denoise.mean_dev.toFixed(1)}, p99 ${t.denoise.p99_dev.toFixed(0)}</div>`:``)}let ve=m(`copy-btn`);ve.onclick=()=>{let e=`--pos-mm ${X.pos.value} --color-levels ${X.col.value} --rot-steps ${X.rot.value} --scale-pct ${X.scl.value} --gop ${X.gop.value}${X.dn.checked?` --denoise-colors`:``} --zstd-level ${X.z.value}`,t=()=>{let t=document.getElementById(`flags-out`);t||(t=document.createElement(`div`),t.id=`flags-out`,t.style.cssText=`margin-top:6px;padding:6px 8px;background:#0d1117;border:1px solid #2c3540;border-radius:4px;user-select:all;word-break:break-all;color:#9ecbff;`,m(`enc-stats`).before(t)),t.textContent=e};navigator.clipboard.writeText(e).then(()=>{m(`enc-status`).textContent=`copied ✓`,t(),setTimeout(()=>{m(`enc-status`).textContent===`copied ✓`&&(m(`enc-status`).textContent=``)},2e3)}).catch(()=>{m(`enc-status`).textContent=`select & copy below`,t()})};let ye=m(`encode-btn`);async function be(){ye.disabled=!0,m(`enc-status`).textContent=`encoding…`;try{let e=await fetch(`/api/encode?${ge()}`);if(!e.ok)throw Error(`api ${e.status}`);let t=await e.json();if(t.error)throw Error(t.error);return _e(t),W=t.perm,L=!1,B(t.url),m(`enc-status`).textContent=`${(t.wallMs/1e3).toFixed(1)} s`,!0}catch(e){return m(`enc-status`).textContent=`failed`,console.error(e),!1}finally{ye.disabled=!1}}ye.onclick=()=>void be();function xe(){if(!l)return;let e=Math.max(1,Math.round(M.clientWidth*2)),t=Math.max(1,Math.round(M.clientHeight*2));(M.width!==e||M.height!==t)&&(M.width=e,M.height=t);let n=M.width,r=M.height;N.clearRect(0,0,n,r),N.fillStyle=`#1a212a`,N.beginPath(),N.roundRect(0,r/2-7,n,14,7),N.fill();let i=l.t/l.fps;N.fillStyle=`#3a4654`,l.gops.forEach((e,t)=>{if(!D[t])return;let a=e.t0/i*n,o=Math.min(e.f1+1,l.t)/l.fps/i*n;N.fillRect(a,r/2-7,o-a,14)}),N.fillStyle=`#4da3ff`,N.beginPath(),N.roundRect(0,r/2-7,Math.max(14,v/i*n),14,7),N.fill();let a=v/i*n;N.fillStyle=`#fff`,N.beginPath(),N.arc(a,r/2,9,0,Math.PI*2),N.fill(),A>=0&&(N.strokeStyle=`#4da3ff`,N.lineWidth=3,N.beginPath(),N.arc(a,r/2,13,performance.now()/200,performance.now()/200+4),N.stroke())}function Se(e){if(!l)return;let t=M.getBoundingClientRect();v=Math.min(1,Math.max(0,(e-t.left)/t.width))*(l.t/l.fps),z(Math.min(l.t-1,Math.floor(v*l.fps)))}let Ce=!1;M.addEventListener(`pointerdown`,e=>{Ce=!0;try{M.setPointerCapture(e.pointerId)}catch{}Se(e.clientX)}),M.addEventListener(`pointermove`,e=>Ce&&Se(e.clientX)),M.addEventListener(`pointerup`,()=>Ce=!1);let Z=m(`play`);function Q(e){g=e,Z.textContent=e?`⏸`:`▶`}Z.onclick=()=>Q(!g),addEventListener(`keydown`,e=>{e.code===`Space`?(e.preventDefault(),Q(!g)):e.code===`KeyC`&&pe(!y)});function we(){e.width=Math.round(innerWidth*re),e.height=Math.round(innerHeight*re)}addEventListener(`resize`,we),we(),window.__cam=(e,t,n,r,i,a)=>G({position:[e,t,n],target:[r,i,a],fov:x}),window.__play=e=>Q(e),window.__compare=e=>pe(e),window.__vm=()=>C.viewMatrix.slice(0),window.__setvm=e=>C.viewMatrix=e;let $=null;window.__setfocal=(e,t)=>$=[e,t],window.__frameShown=()=>O,window.__dump=()=>E?.postMessage({type:`dump`}),window.__seek=e=>{l&&(Q(!1),v=(e+.5)/l.fps,z(e))};let Te=S.get(`file`);if(Te){m(`panel`).style.display=`none`,q.style.display=`none`;let e=S.get(`cam`);if(e){let t=e.split(`,`).map(Number);t.length>=6&&t.every(Number.isFinite)&&(V={position:t.slice(0,3),target:t.slice(3,6),fov:t[6]||60})}B(Te)}else if(await me(),!await be()){m(`panel`).style.display=`none`,q.style.display=`none`;let e=`juggle.splat4d`;try{let t=await fetch(`demo.json`);if(t.ok){let n=await t.json();e=n.file??e,V=n.camera??null}}catch{}B(e)}let Ee=performance.now();function De(){requestAnimationFrame(De);let e=performance.now(),t=(e-Ee)/1e3;if(Ee=e,C.update(),l&&u&&T){let e=l.t/l.fps;g&&A<0&&(v=(v+Math.min(t,.1))%e);let n=Math.min(l.t-1,Math.floor(v*l.fps));n!==O&&z(n);let r=$?$[1]:.5*innerHeight/Math.tan(x*Math.PI/360),a=$?$[0]:r,o=s(a,r,innerWidth,innerHeight),f=c(o,C.viewMatrix);E?.postMessage({type:`view`,viewProj:new Float32Array(f)}),u.setCamera(o,C.viewMatrix,a,r,innerWidth,innerHeight),d&&y&&d.setCamera(o,C.viewMatrix,a,r,innerWidth,innerHeight),m(`clock`).textContent=`${v.toFixed(2)} / ${e.toFixed(2)}`,xe(),i.render(y&&d?[d,u]:[u])}else i.render([]);ae++,e-oe>1e3&&(m(`m-fps`).textContent=String(ae),ae=0,oe=e)}De()}v().catch(e=>console.error(e));