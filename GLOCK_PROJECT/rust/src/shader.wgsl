// ============================================================
// GLOCK-19 CINEMATIC PBR SHADER  (WGSL / wgpu)
// ============================================================
// Uniform: view_proj матриця + camera_pos (xyz + w=pad)
// Vertex attrs: position(0), normal(1)
// ============================================================

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz = world-space camera, w = pad
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal:    vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.normal    = model.normal;
    out.world_pos = model.position;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

// ── PBR math helpers ────────────────────────────────────────

// GGX / Trowbridge-Reitz нормальний розподіл
fn D_GGX(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom + 0.0001);
}

// Schlick-GGX геометрична маскировка (пряме освітлення)
fn G_SchlickGGX(n_dot_v: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn G_Smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return G_SchlickGGX(n_dot_v, roughness) * G_SchlickGGX(n_dot_l, roughness);
}

// Fresnel-Schlick апроксимація
fn F_Schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Повний Cook-Torrance specular BRDF (одна точка освітлення)
fn cook_torrance(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, roughness: f32,
    light_color: vec3<f32>
) -> vec3<f32> {
    let h        = normalize(v + l);
    let n_dot_v  = max(dot(n, v), 0.0001);
    let n_dot_l  = max(dot(n, l), 0.0);
    let n_dot_h  = max(dot(n, h), 0.0);
    let h_dot_v  = max(dot(h, v), 0.0);

    let f0       = mix(vec3<f32>(0.04), albedo, metallic);
    let F        = F_Schlick(h_dot_v, f0);
    let D        = D_GGX(n_dot_h, roughness);
    let G        = G_Smith(n_dot_v, n_dot_l, roughness);

    let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);

    // Energ conservation: kS = Fresnel, kD = diffuse
    let kS       = F;
    let kD       = (1.0 - kS) * (1.0 - metallic);
    let diffuse  = albedo / 3.14159265;

    return (kD * diffuse + specular) * light_color * n_dot_l;
}

// Мінімальна IBL ambient: spherical harmonics L0+L1 для студійного HDRI
fn ibl_ambient(n: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    // SH warm-top / cool-bottom aprrox (нейтральний studio environment)
    let sky    = vec3<f32>(0.25, 0.30, 0.40);   // Cold sky hemisphere
    let ground = vec3<f32>(0.12, 0.10, 0.08);   // Warm ground reflection
    let t      = n.y * 0.5 + 0.5;
    let irradiance = mix(ground, sky, t);

    // Diffuse IBL
    let diffuse_ibl = albedo * irradiance * (1.0 - metallic);

    // Specular IBL (simplified — approx envmap lod via roughness)
    let r       = reflect(-n, n); // dummy – we don't have real envmap
    let env_lod = roughness * roughness;
    let env_col = mix(vec3<f32>(0.8, 0.85, 0.95), vec3<f32>(0.1, 0.1, 0.12), env_lod);
    let F_amb   = F_Schlick(max(dot(n, n), 0.0), f0); // очень грубо для ambient
    let spec_ibl = env_col * F_amb * (1.0 - roughness * 0.6) * metallic;

    return (diffuse_ibl + spec_ibl) * 0.35; // scaled AO
}

// ACES filmic tone-map
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Fragment ─────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var n         = normalize(in.normal);
    let view_dir  = normalize(camera.camera_pos.xyz - in.world_pos); // ✓ правильний V

    // ── Геометрична діагностика кутів (Edge Detection через похідні) ──
    let dx       = dpdx(in.world_pos);
    let dy       = dpdy(in.world_pos);
    let geom_n   = normalize(cross(dx, dy));
    let edge_t   = clamp(1.0 - dot(n, geom_n), 0.0, 1.0);
    let is_edge  = step(0.12, edge_t) * smoothstep(0.12, 0.55, edge_t);

    // ── Матеріал (Y-спліт: slide = метал, frame = полімер) ──
    var albedo    = vec3<f32>(0.038, 0.038, 0.040); // чорний нейлон рамки
    var roughness = 0.82;
    var metallic  = 0.0;

    let is_metal  = step(0.55, in.world_pos.y); // slide + barrel вище за рамку
    if (is_metal > 0.5) {
        // Tennifer (Ferritic Nitro-Carburizing) — майже чорна воронована сталь
        albedo    = vec3<f32>(0.08, 0.085, 0.092);
        roughness = 0.42;
        metallic  = 0.92;

        // Wear scratches — жива сталь на гранях (більш яскрава)
        if (is_edge > 0.0) {
            let wear = is_edge * smoothstep(0.0, 1.0, is_edge);
            albedo    = mix(albedo, vec3<f32>(0.72, 0.72, 0.75), wear * 0.85);
            roughness = mix(roughness, 0.15, wear * 0.9);
            metallic  = mix(metallic, 1.0, wear * 0.7);
        }
    } else {
        // Polymer frame — Stippling texture approximation via normal variation
        let stip  = fract(sin(dot(in.world_pos * 18.0, vec3<f32>(12.9898, 78.233, 45.164))) * 43758.5) * 0.04;
        roughness = clamp(roughness + stip, 0.0, 1.0);
    }

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // ── Освітлення: 3 cinematic sources ──────────────────────
    // Key  — тепле студійне з лівого верху
    let L1       = normalize(vec3<f32>(1.2, 2.0, 0.8));
    let C1       = vec3<f32>(1.10, 1.02, 0.90) * 3.5;
    // Fill — холодне з протилежного боку
    let L2       = normalize(vec3<f32>(-1.0, 0.5, -1.2));
    let C2       = vec3<f32>(0.35, 0.42, 0.60) * 1.2;
    // Rim  — контровий збоку знизу (підкреслює форму пістолета)
    let L3       = normalize(vec3<f32>(0.0, -1.0, 1.5));
    let C3       = vec3<f32>(0.60, 0.58, 0.55) * 0.8;

    var Lo = vec3<f32>(0.0);
    Lo += cook_torrance(n, view_dir, L1, albedo, metallic, roughness, C1);
    Lo += cook_torrance(n, view_dir, L2, albedo, metallic, roughness, C2);
    Lo += cook_torrance(n, view_dir, L3, albedo, metallic, roughness, C3);

    // ── IBL Ambient ──────────────────────────────────────────
    let ambient  = ibl_ambient(n, albedo, metallic, roughness, f0);

    // ── Combine HDR ──────────────────────────────────────────
    let hdr      = Lo + ambient;

    // ── Tone-map + Gamma ─────────────────────────────────────
    let mapped   = aces_filmic(hdr);
    let ldr      = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(ldr, 1.0);
}