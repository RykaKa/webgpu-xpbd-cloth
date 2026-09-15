// ============================================================
// cloth.js — Cloth simulation XPBD on WebGPU
// ============================================================
//
// Implementation features:
//   - XPBD Small Steps: SUBSTEPS substeps per frame
//   - Gauss-Seidel with constraint coloring (eliminates races)
//   - 8 constraint groups, no shared vertices within each
//   - 4 corners pinned, central vertex moves along a sine
//   - Gravity toggled by checkbox
//
// Physical constants tuned for cotton/linen cloth:
//   - COMPLIANCE = 1e-5 (stretch stiffness)
//   - DAMPING    = 0.99 (velocity damping)
//   - GRAVITY    = -9.81 m/s² (real physical)
// ============================================================


// ---- Simulation constants ----
const CLOTH_SIZE = 1.0;           // cloth size: 1 m × 1 m
const GRID_N = 20;            // 20 × 20 cells = 21 × 21 vertices
const VERTEX_COUNT = (GRID_N + 1) * (GRID_N + 1);

const SUBSTEPS = 20;   // substeps per frame
const COMPLIANCE = 5e-5; // stiff cloth
const DAMPING = 0.999; // natural decay
const GRAVITY_Y = -9.81; // m/s²
const DT = 1.0 / 60.0;
const SUB_DT = DT / SUBSTEPS;

const WAVE_AMPLITUDE = 0.1;        // center wave amplitude, m
const WAVE_FREQUENCY = 1.0;         // angular frequency, rad/s

const VERTEX_FLOATS = 12;           // 12 floats = 48 bytes per vertex


// ---- Special vertex indices ----
const CORNER_INDICES = [
    0,
    GRID_N,
    GRID_N * (GRID_N + 1),
    (GRID_N + 1) * (GRID_N + 1) - 1
];

const CENTER_INDEX =
    Math.floor((GRID_N + 1) / 2) * (GRID_N + 1) +
    Math.floor((GRID_N + 1) / 2);


// ============================================================
// Matrix helpers
// ============================================================

function mat4Perspective(fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far * nf, -1,
        0, 0, near * far * nf, 0
    ]);
}

function mat4LookAt(eye, target, up) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const cross = (a, b) => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const norm = (v) => {
        const l = Math.hypot(v[0], v[1], v[2]);
        return [v[0] / l, v[1] / l, v[2] / l];
    };
    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
    ]);
}

function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let s = 0;
            for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
            out[col * 4 + row] = s;
        }
    }
    return out;
}


// ============================================================
// Cloth data generation with constraint coloring
// ============================================================
//
// Coloring guarantees that inside one constraint group
// no two constraints share the same vertex.
// This lets us solve a group in parallel without races.
//
// Groups:
//   0: horizontal edges, x % 2 == 0
//   1: horizontal edges, x % 2 == 1
//   2: vertical edges, y % 2 == 0
//   3: vertical edges, y % 2 == 1
//   4: "\" diagonals, x % 2 == 0
//   5: "\" diagonals, x % 2 == 1
//   6: "/" diagonals, x % 2 == 0
//   7: "/" diagonals, x % 2 == 1
//
// 8 groups total.

const GROUP_COUNT = 16;

function generateClothData() {
    const step = CLOTH_SIZE / GRID_N;
    const half = CLOTH_SIZE / 2;

    // ---- Vertices ----
    const vertexData = new Float32Array(VERTEX_COUNT * VERTEX_FLOATS);
    for (let y = 0; y <= GRID_N; y++) {
        for (let x = 0; x <= GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const off = i * VERTEX_FLOATS;

            vertexData[off + 0] = x * step - half;   // pos.x
            vertexData[off + 1] = 0.0;               // pos.y
            vertexData[off + 2] = y * step - half;   // pos.z
            vertexData[off + 3] = 0.0;               // prevPos.x
            vertexData[off + 4] = 0.0;               // vel.x
            vertexData[off + 5] = 0.0;               // vel.y
            vertexData[off + 6] = 0.0;               // vel.z
            vertexData[off + 7] = 0.0;               // prevPos.y
            vertexData[off + 8] = 1.0;               // invMass
            vertexData[off + 9] = 0.0;               // prevPos.z
            vertexData[off + 10] = 0.0;              // padding
            vertexData[off + 11] = 0.0;              // padding
        }
    }
    // Pinned corners: invMass = 0
    for (const idx of CORNER_INDICES) {
        vertexData[idx * VERTEX_FLOATS + 8] = 0.0;
    }

    // ---- Constraints, split into 8 groups ----
    const groupArrays = [];
    for (let g = 0; g < GROUP_COUNT; g++) groupArrays.push([]);

    // Groups 0, 1: horizontal
    for (let y = 0; y <= GRID_N; y++) {
        for (let x = 0; x < GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const j = y * (GRID_N + 1) + (x + 1);
            const g = x % 2;
            groupArrays[g].push(i, j, step);
        }
    }

    // Groups 2, 3: vertical
    for (let y = 0; y < GRID_N; y++) {
        for (let x = 0; x <= GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const j = (y + 1) * (GRID_N + 1) + x;
            const g = 2 + (y % 2);
            groupArrays[g].push(i, j, step);
        }
    }

    // Groups 4, 5: "\" diagonals — colored by x % 2
    for (let y = 0; y < GRID_N; y++) {
        for (let x = 0; x < GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const j = (y + 1) * (GRID_N + 1) + (x + 1);
            const g = 4 + (x % 2);
            groupArrays[g].push(i, j, Math.sqrt(2) * step);
        }
    }

    // Groups 6, 7: "/" diagonals — colored by x % 2
    for (let y = 0; y < GRID_N; y++) {
        for (let x = 0; x < GRID_N; x++) {
            const i = y * (GRID_N + 1) + (x + 1);
            const j = (y + 1) * (GRID_N + 1) + x;
            const g = 6 + (x % 2);
            groupArrays[g].push(i, j, Math.sqrt(2) * step);
        }
    }

    // ---- Bend constraints (resistance to bending) ----
    // Between vertices two cells apart. 8 new groups (8..15).
    // Coloring: same logic as stretch — x % 2 or y % 2.

    // Groups 8, 9: horizontal bend — (x, y) to (x+2, y)
    for (let y = 0; y <= GRID_N; y++) {
        for (let x = 0; x + 2 <= GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const j = y * (GRID_N + 1) + (x + 2);
            const g = 8 + (x % 2);
            groupArrays[g].push(i, j, 2 * step);
        }
    }

    // Groups 10, 11: vertical bend — (x, y) to (x, y+2)
    for (let y = 0; y + 2 <= GRID_N; y++) {
        for (let x = 0; x <= GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const j = (y + 2) * (GRID_N + 1) + x;
            const g = 10 + (y % 2);
            groupArrays[g].push(i, j, 2 * step);
        }
    }

    // Groups 12, 13: "\" bend — (x, y) to (x+2, y+2)
    for (let y = 0; y + 2 <= GRID_N; y++) {
        for (let x = 0; x + 2 <= GRID_N; x++) {
            const i = y * (GRID_N + 1) + x;
            const j = (y + 2) * (GRID_N + 1) + (x + 2);
            const g = 12 + (x % 2);
            groupArrays[g].push(i, j, 2 * Math.sqrt(2) * step);
        }
    }

    // Groups 14, 15: "/" bend — (x+2, y) to (x, y+2)
    for (let y = 0; y + 2 <= GRID_N; y++) {
        for (let x = 0; x + 2 <= GRID_N; x++) {
            const i = y * (GRID_N + 1) + (x + 2);
            const j = (y + 2) * (GRID_N + 1) + x;
            const g = 14 + (x % 2);
            groupArrays[g].push(i, j, 2 * Math.sqrt(2) * step);
        }
    }

    // ---- Render indices ----
    const indices = [];
    for (let y = 0; y < GRID_N; y++) {
        for (let x = 0; x < GRID_N; x++) {
            const tl = y * (GRID_N + 1) + x;
            const tr = y * (GRID_N + 1) + (x + 1);
            const bl = (y + 1) * (GRID_N + 1) + x;
            const br = (y + 1) * (GRID_N + 1) + (x + 1);

            indices.push(tl, bl, tr);
            indices.push(tr, bl, br);
        }
    }

    // ---- Packing ----
    const groups = groupArrays.map(arr => ({
        data: new Float32Array(arr),
        count: arr.length / 3
    }));

    const totalConstraints = groups.reduce((sum, g) => sum + g.count, 0);

    return {
        vertexData,
        groups,
        totalConstraints,
        indices: new Uint32Array(indices)
    };
}


// ============================================================
// WGSL: simulation
// ============================================================

const SIMULATION_SHADER = /* wgsl */`
struct SimParams {
    substepDt       : f32,
    gravity         : f32,
    damping         : f32,
    compliance      : f32,
    constraintCount : u32,
    vertexCount     : u32,
    centerIndex     : u32,
    time            : f32,
    waveAmplitude   : f32,
    waveFrequency   : f32,
    _pad0           : f32,
    _pad1           : f32,
};

struct Vertex {
    pos      : vec3f,
    prevPosX : f32,
    vel      : vec3f,
    prevPosY : f32,
    invMass  : f32,
    prevPosZ : f32,
    _pad0    : f32,
    _pad1    : f32,
};

@group(0) @binding(0) var<uniform>             params      : SimParams;
@group(0) @binding(1) var<storage, read_write> vertices    : array<Vertex>;
@group(0) @binding(2) var<storage, read>       constraints : array<f32>;
@group(0) @binding(3) var<storage, read_write> lambdas     : array<f32>;


@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx >= params.vertexCount) { return; }

    var v = vertices[idx];
        if (v.invMass <= 0.0) {
        vertices[idx] = v;
        return;
    }

    v.vel.y += params.gravity * params.substepDt;
    v.vel *= params.damping;
    v.pos += v.vel * params.substepDt;

    // Store position AFTER gravity integration, but BEFORE solve.
    v.prevPosX = v.pos.x;
    v.prevPosY = v.pos.y;
    v.prevPosZ = v.pos.z;

    vertices[idx] = v;
}


@compute @workgroup_size(256)
fn solveConstraints(@builtin(global_invocation_id) gid : vec3u) {
    let cIdx = gid.x;
    if (cIdx >= params.constraintCount) { return; }

    let i = u32(constraints[cIdx * 3 + 0]);
    let j = u32(constraints[cIdx * 3 + 1]);
    let restLen = constraints[cIdx * 3 + 2];

    var vi = vertices[i];
    var vj = vertices[j];

    let wSum = vi.invMass + vj.invMass;
    if (wSum <= 0.0) { return; }

    let diff = vi.pos - vj.pos;
    let dist = length(diff);
    if (dist < 1e-8) { return; }

    let C = dist - restLen;
    let alphaTilde = params.compliance / (params.substepDt * params.substepDt);

    var lambda = lambdas[cIdx];
    let deltaLambda = (-C - alphaTilde * lambda) / (wSum + alphaTilde);
    lambda += deltaLambda;
    lambdas[cIdx] = lambda;

    let n = diff / dist;
    let correction = n * deltaLambda;

    if (vi.invMass > 0.0) {
        vi.pos += correction * vi.invMass;
        vertices[i] = vi;
    }
    if (vj.invMass > 0.0) {
        vj.pos -= correction * vj.invMass;
        vertices[j] = vj;
    }
}


@compute @workgroup_size(256)
fn updateVelocities(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx >= params.vertexCount) { return; }

    var v = vertices[idx];


    // // Forcefully move central vertex: smooth sine (per task) + periodic impulse
    // // to excite traveling waves.
    // if (idx == params.centerIndex) {
    //     let phase = params.time * params.waveFrequency;
    //     let baseY = params.waveAmplitude * sin(phase);

    //     // Periodic impulse every IMPULSE_PERIOD seconds.
    //     // The impulse is a short, sharp bump that excites all vibration modes.
    //     let impulsePeriod = 3.0;     // seconds between impulses
    //     let impulseWidth  = 0.15;    // impulse duration, seconds
    //     let impulseAmp    = 0.20;    // impulse height, meters

    //     let tMod = params.time - floor(params.time / impulsePeriod) * impulsePeriod;
    //     var impulseY = 0.0;
    //     if (tMod < impulseWidth) {
    //         // Half-sine bump: rises and falls within impulseWidth.
    //         let s = tMod / impulseWidth;              // 0..1
    //         impulseY = impulseAmp * sin(s * 3.14159265);
    //     }

    //     v.pos.y = baseY + impulseY;
    //     v.vel.x = 0.0;
    //     v.vel.y = params.waveAmplitude * params.waveFrequency * cos(phase);
    //     v.vel.z = 0.0;
    //     vertices[idx] = v;
    //     return;
    // }

    // Forcefully move central vertex along a sine (per task specification).
    if (idx == params.centerIndex) {
        let phase = params.time * params.waveFrequency;
        v.pos.y = params.waveAmplitude * sin(phase);
        v.vel.x = 0.0;
        v.vel.y = params.waveAmplitude * params.waveFrequency * cos(phase);
        v.vel.z = 0.0;
        vertices[idx] = v;
        return;
    }

    // Velocity from position delta (PBD-style).
    if (v.invMass > 0.0) {
        let prev = vec3f(v.prevPosX, v.prevPosY, v.prevPosZ);
        // v.vel = (v.pos - prev) / params.substepDt;
        let newVel = (v.pos - prev) / params.substepDt;
        v.vel = mix(v.vel, newVel, 0.5);   // 0.5 old + 0.5 new velocity
    }
    vertices[idx] = v;
}


@compute @workgroup_size(256)
fn resetLambdas(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx >= params.constraintCount) { return; }
    lambdas[idx] = 0.0;
}
`;


// ============================================================
// WGSL: rendering
// ============================================================

const RENDER_SHADER = /* wgsl */`
struct Uniforms {
    mvp         : mat4x4f,   // 64 bytes
    centerIndex : u32,       // 4 bytes
    _pad0       : u32,       // 4 bytes
    _pad1       : u32,       // 4 bytes
    _pad2       : u32,       // 4 bytes
};

struct Vertex {
    pos      : vec3f,
    prevPosX : f32,
    vel      : vec3f,
    prevPosY : f32,
    invMass  : f32,
    prevPosZ : f32,
    _pad0    : f32,
    _pad1    : f32,
};

@group(0) @binding(0) var<uniform>       uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       worldPos : vec3f,
    @location(1)       isPinned : f32,
    @location(2)       isCenter : f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vidx : u32) -> VertexOutput {
    let v = vertices[vidx];
    var out : VertexOutput;
    out.position = uniforms.mvp * vec4f(v.pos, 1.0);
    out.worldPos = v.pos;
    out.isPinned = select(0.0, 1.0, v.invMass <= 0.0);
    out.isCenter = select(0.0, 1.0, vidx == uniforms.centerIndex);
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let N = vec3f(0.0, 1.0, 0.0);
    let L = normalize(vec3f(0.5, -1.0, 0.5));
    let diff = max(dot(N, -L), 0.0);
    let lighting = 0.3 + diff * 0.7;

    // let e1 = smoothstep(0.0, 0.02, abs(fract(in.worldPos.x * 20.0) - 0.5));
    // let e2 = smoothstep(0.0, 0.02, abs(fract(in.worldPos.z * 20.0) - 0.5));
    // let edge = min(e1, e2);

    // var baseColor = vec3f(0.65, 0.70, 0.75);
    // if (in.isPinned > 0.5) {
    //     baseColor = vec3f(0.9, 0.2, 0.2);   // red for pinned
    // }
    // if (in.isCenter > 0.5) {
    //     baseColor = vec3f(0.2, 0.4, 1.0);   // blue for central
    // }
    // baseColor = mix(baseColor * 0.3, baseColor, edge);

    // Grid along X and Z (sides of the squares)
    let e1 = smoothstep(0.0, 0.02, abs(fract(in.worldPos.x * 20.0) - 0.5));
    let e2 = smoothstep(0.0, 0.02, abs(fract(in.worldPos.z * 20.0) - 0.5));

    // Diagonal inside each cell.
    // Geometrically, each cell is split by the diagonal bl -> tr,
    // i.e. from (0, 1) to (1, 0). Equation: cellX + cellZ = 1.
    let cellX = fract(in.worldPos.x * 20.0);
    let cellZ = fract(in.worldPos.z * 20.0);
    let diagAnti = abs(cellX + cellZ - 1.0);
    let e3 = smoothstep(0.0, 0.02, diagAnti);

    // Final darkening: square + diagonal.
    let edge = min(min(e1, e2), e3);

    var baseColor = vec3f(0.65, 0.70, 0.75);
    if (in.isPinned > 0.5) {
        baseColor = vec3f(0.9, 0.2, 0.2);   // красный для закреплённых
    }
    if (in.isCenter > 0.5) {
        baseColor = vec3f(0.2, 0.4, 1.0);   // синий для центральной
    }
    baseColor = mix(baseColor * 0.3, baseColor, edge);

    return vec4f(baseColor * lighting, 1.0);
}
`;


// ============================================================
// Utils
// ============================================================

function createBuffer(device, data, usage, label) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage,
        mappedAtCreation: true,
        label
    });
    new (data.constructor)(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
}

function createStorageBuffer(device, byteSize, label) {
    return device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label
    });
}


// ============================================================
// main
// ============================================================

async function main() {
    const canvas = document.getElementById('gpuCanvas');
    const gravityToggle = document.getElementById('gravityToggle');
    const statsEl = document.getElementById('stats');

    console.log('[cloth] start');

    if (!navigator.gpu) throw new Error('WebGPU not supported.');

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    // Debug: log generated cloth stats to verify topology and constraint counts.
    const cloth = generateClothData();
    console.log('[cloth] vertices =', cloth.vertexData.length);
    console.log('[cloth] indices =', cloth.indices.length);
    console.log('[cloth] constraints =', cloth.totalConstraints);
    console.log('[cloth] groups =', cloth.groups.map(g => g.count));

    const vertexBufferSize = cloth.vertexData.byteLength;

    // ---- Vertex buffer ----
    const vertexBuffer = createBuffer(
        device,
        cloth.vertexData,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        'vertexBuffer'
    );

    // ---- One constraint and lambda buffer per group ----
    const constraintBuffers = cloth.groups.map((g, i) =>
        createBuffer(device, g.data,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            `constraints_${i}`)
    );

    const lambdaBuffers = cloth.groups.map((g, i) =>
        createStorageBuffer(device, g.count * 4, `lambdas_${i}`)
    );

    const indexBuffer = createBuffer(
        device,
        cloth.indices,
        GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        'indices'
    );

    const simParamsBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const renderUniformBuffer = device.createBuffer({
        size: 80,   // 64 (mvp) + 4 (centerIndex) + 12 (padding)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective(Math.PI / 4, aspect, 0.1, 100);
    const view = mat4LookAt([1.1, 1.1, 1.1], [0, -0.3, 0], [0, 1, 0]);
    const mvp = mat4Multiply(proj, view);
    const renderUniformData = new Float32Array(20);   // 20 floats = 80 bytes
    renderUniformData.set(mvp, 0);
    // centerIndex is u32, but we write it through Float32Array.
    // WebGPU has no writeBuffer for u32 by index, so we use DataView.
    const u32View = new Uint32Array(renderUniformData.buffer);
    u32View[16] = CENTER_INDEX;   // position 16 floats = 64 bytes = 16 u32
    device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformData);

    const simModule = device.createShaderModule({ code: SIMULATION_SHADER });
    const renderModule = device.createShaderModule({ code: RENDER_SHADER });

    const simBindGroupLayout = device.createBindGroupLayout({
        label: 'simBindGroupLayout',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
        ]
    });
    const simPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [simBindGroupLayout]
    });

    const integratePipeline = device.createComputePipeline({
        layout: simPipelineLayout,
        compute: { module: simModule, entryPoint: 'integrate' }
    });
    const solvePipeline = device.createComputePipeline({
        layout: simPipelineLayout,
        compute: { module: simModule, entryPoint: 'solveConstraints' }
    });
    const updateVelPipeline = device.createComputePipeline({
        layout: simPipelineLayout,
        compute: { module: simModule, entryPoint: 'updateVelocities' }
    });
    const resetLambdaPipeline = device.createComputePipeline({
        layout: simPipelineLayout,
        compute: { module: simModule, entryPoint: 'resetLambdas' }
    });

    const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: renderModule, entryPoint: 'vs_main', buffers: [] },
        fragment: {
            module: renderModule,
            entryPoint: 'fs_main',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' }
    });

    // ---- One bind group per constraint group ----
    const simBindGroups = cloth.groups.map((g, i) =>
        device.createBindGroup({
            layout: simBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: simParamsBuffer } },
                { binding: 1, resource: { buffer: vertexBuffer } },
                { binding: 2, resource: { buffer: constraintBuffers[i] } },
                { binding: 3, resource: { buffer: lambdaBuffers[i] } }
            ]
        })
    );

    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: renderUniformBuffer } },
            { binding: 1, resource: { buffer: vertexBuffer } }
        ]
    });

    let time = 0;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    function frame() {
        const now = performance.now();
        frameCount++;
        if (now - lastFpsTime > 500) {
            const fps = frameCount / ((now - lastFpsTime) / 1000);
            statsEl.textContent = `${fps.toFixed(0)} FPS`;
            frameCount = 0;
            lastFpsTime = now;
        }

        time += DT;

        const gravity = gravityToggle.checked ? GRAVITY_Y : 0.0;
        const simParams = new Float32Array(12);
        simParams[0] = SUB_DT;
        simParams[1] = gravity;
        simParams[2] = DAMPING;
        simParams[3] = COMPLIANCE;
        simParams[7] = time;
        simParams[8] = WAVE_AMPLITUDE;
        simParams[9] = WAVE_FREQUENCY;

        const u32View = new Uint32Array(simParams.buffer);
        u32View[4] = cloth.totalConstraints;
        u32View[5] = VERTEX_COUNT;
        u32View[6] = CENTER_INDEX;

        device.queue.writeBuffer(simParamsBuffer, 0, simParams);

        const encoder = device.createCommandEncoder();

        for (let s = 0; s < SUBSTEPS; s++) {
            // --- resetLambdas across all groups ---
            for (let g = 0; g < GROUP_COUNT; g++) {
                const pass = encoder.beginComputePass();
                pass.setPipeline(resetLambdaPipeline);
                pass.setBindGroup(0, simBindGroups[g]);
                pass.dispatchWorkgroups(Math.ceil(cloth.groups[g].count / 256));
                pass.end();
            }

            // --- integrate (once for all vertices) ---
            {
                const pass = encoder.beginComputePass();
                pass.setPipeline(integratePipeline);
                pass.setBindGroup(0, simBindGroups[0]);   // any of them, only vertexBuffer matters
                pass.dispatchWorkgroups(Math.ceil(VERTEX_COUNT / 256));
                pass.end();
            }

            // --- solveConstraints over 8 groups ---
            for (let g = 0; g < GROUP_COUNT; g++) {
                const pass = encoder.beginComputePass();
                pass.setPipeline(solvePipeline);
                pass.setBindGroup(0, simBindGroups[g]);
                pass.dispatchWorkgroups(Math.ceil(cloth.groups[g].count / 256));
                pass.end();
            }

            // --- updateVelocities (once for all vertices) ---
            {
                const pass = encoder.beginComputePass();
                pass.setPipeline(updateVelPipeline);
                pass.setBindGroup(0, simBindGroups[0]);
                pass.dispatchWorkgroups(Math.ceil(VERTEX_COUNT / 256));
                pass.end();
            }
        }

        const textureView = context.getCurrentTexture().createView();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.05, g: 0.05, b: 0.12, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.setIndexBuffer(indexBuffer, 'uint32');
        renderPass.drawIndexed(cloth.indices.length);
        renderPass.end();

        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
    console.log('[cloth] cycle started');
}

main().catch(err => {
    console.error('[cloth] ERROR:', err);
    document.body.innerHTML = `<div style="color:#e94560;padding:40px;text-align:center">
        <h2>Error</h2><p>${err.message}</p></div>`;
});