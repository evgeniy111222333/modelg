# ═══════════════════════════════════════════════════════════════════════
#  GLOCK-19  Software PBR Ray-Marcher  v6.0
#  — Full Cook-Torrance GGX PBR
#  — Multi-light studio setup (5 lights)
#  — Ambient Occlusion + Soft Shadows
#  — Grip stipple via normal perturbation (no geometry cost)
#  — Tennifer steel + polymer frame + polished bore materials
#  — ACES tone mapping + film grain + vignette
#  — Single-batch rendering (numpy internal parallelism)
#  — 800 × 800 main renders
# ═══════════════════════════════════════════════════════════════════════
import os, sys, time
# Let numpy use its own internal threading for max SIMD throughput
# (Python-level threading causes contention — single batch is faster)
import numpy as np

sys.path.append(os.path.dirname(__file__))
from render_cameras import CameraConfig
from sdf_glock import GlockSDF
from mesh_exporter import MeshExporter

# ──────────────────────────────────────────────────────────────────────
#  PBR MATH
# ──────────────────────────────────────────────────────────────────────
def _normalize(v):
    n = np.linalg.norm(v.astype(np.float32), axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-8)).astype(np.float32)

def _D_GGX(ndh, roughness):
    a  = roughness * roughness          # (N,1)
    a2 = a * a
    d  = ndh * ndh * (a2 - 1.0) + 1.0
    return a2 / (np.pi * d * d + 1e-7)  # (N,1)

def _G_SchlickGGX(ndv, roughness):
    k = (roughness + 1.0)**2 / 8.0
    return ndv / (ndv*(1.0-k) + k + 1e-7)

def _G_Smith(ndv, ndl, roughness):
    return _G_SchlickGGX(ndv, roughness) * _G_SchlickGGX(ndl, roughness)

def _F_Schlick(cos_t, f0):
    return f0 + (1.0 - f0) * (1.0 - np.clip(cos_t, 0, 1))**5

def _cook_torrance(n, v, l_dir, albedo, metallic, roughness, light_col):
    """All arrays: n,v shape (N,3); metallic,roughness shape (N,1); albedo (N,3)."""
    h   = _normalize(v + l_dir)                                   # (N,3)
    ndv = np.clip((n*v).sum(-1,keepdims=True),   1e-4, 1.0)       # (N,1)
    ndl = np.clip((n*l_dir).sum(-1,keepdims=True), 0.0, 1.0)      # (N,1)
    ndh = np.clip((n*h).sum(-1,keepdims=True),   0.0, 1.0)        # (N,1)
    hdv = np.clip((h*v).sum(-1,keepdims=True),   0.0, 1.0)        # (N,1)

    f0  = 0.04*(1-metallic) + albedo*metallic                      # (N,3)
    F   = _F_Schlick(hdv, f0)                                      # (N,3)
    D   = _D_GGX(ndh, roughness)                                   # (N,1)
    G   = _G_Smith(ndv, ndl, roughness)                            # (N,1)

    spec = (D * G) * F / (4.0*ndv*ndl + 1e-7)                     # (N,3) ← fixed broadcast
    kS   = F
    kD   = (1.0 - kS) * (1.0 - metallic)
    diff = albedo / np.pi
    return (kD*diff + spec) * light_col * ndl

def _ibl_ambient(n, albedo, metallic, roughness, f0):
    """Trilinear studio IBL: cool sky + warm ground + neutral walls."""
    t     = np.clip(n[...,1:2]*0.5+0.5, 0, 1)
    sky   = np.array([0.20,0.24,0.34])
    gnd   = np.array([0.16,0.13,0.10])
    side  = np.array([0.12,0.12,0.14])
    irr   = gnd + t*(sky-gnd) + (1.0-np.abs(n[...,1:2]))*side*0.28
    diff_ibl = albedo * irr * (1.0 - metallic)

    env_lod = roughness**1.4
    env_col = (np.array([0.82,0.86,0.94])*(1-env_lod)
              + np.array([0.06,0.06,0.08])*env_lod)
    ndv  = np.clip(n[...,1:2], 0, 1)
    F_ab = _F_Schlick(ndv, f0)
    spec_ibl = env_col * F_ab * (1.0 - roughness*0.68) * metallic

    return (diff_ibl + spec_ibl) * 0.48

def _aces(x):
    a,b,c,d,e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b))/(x*(c*x+d)+e), 0, 1)

# ──────────────────────────────────────────────────────────────────────
#  NORMAL PERTURBATION  (grip stippling via analytic bump)
# ──────────────────────────────────────────────────────────────────────
def _perturb_normal_stipple(n, p, metal_t):
    """Add high-freq grip-texture bump to normals without geometry cost."""
    freq   = 18.5
    px, py, pz = p[:,0], p[:,1], p[:,2]
    # Pseudo-random hash for pyramidal stipple
    h1 = np.sin(px*17.3 + py*53.7 + pz*29.1)*43758.5 % 1.0
    h2 = np.sin(px*41.2 - py*31.8 + pz*61.4)*34578.3 % 1.0
    h3 = np.sin(-px*23.6+ py*71.2 + pz*13.9)*29471.2 % 1.0
    # Scale by inverse metalness so only polymer gets texture
    poly_mask = (1.0 - np.clip(metal_t*1.5, 0, 1))[:,np.newaxis]
    grip_zone = np.clip(-(py + 0.5)*0.5, 0, 1)[:,np.newaxis]   # stronger at grip bottom
    strength  = 0.14 * poly_mask * grip_zone
    dn = np.stack([(h1-0.5)*strength[:,0],
                   (h2-0.5)*strength[:,0],
                   (h3-0.5)*strength[:,0]], axis=-1)
    return _normalize(n + dn)

# ──────────────────────────────────────────────────────────────────────
#  AMBIENT OCCLUSION
# ──────────────────────────────────────────────────────────────────────
def _calc_ao(sdf_fn, p, n, steps=6, max_dist=0.70):
    occ   = np.zeros(p.shape[0], dtype=np.float32)
    scale = np.float32(1.0)
    p32   = p.astype(np.float32)
    n32   = n.astype(np.float32)
    for i in range(1, steps+1):
        t_ao  = np.float32(max_dist * i / steps)
        p_ao  = p32 + n32 * t_ao
        d_ao  = sdf_fn(p_ao).astype(np.float32)
        occ  += scale * np.maximum(np.float32(0.0), t_ao - d_ao)
        scale *= np.float32(0.48)
    ao = np.clip(np.float32(1.0) - occ * np.float32(1.8), np.float32(0.0), np.float32(1.0))
    return ao ** np.float32(1.35)

# ──────────────────────────────────────────────────────────────────────
#  SOFT SHADOWS
# ──────────────────────────────────────────────────────────────────────
def _soft_shadow(sdf_fn, ro, rd, t_min=0.06, t_max=5.5, k=12.0, steps=16):
    N  = ro.shape[0]
    sh = np.ones(N,  dtype=np.float32)
    t  = np.full(N,  t_min, dtype=np.float32)
    ph = np.full(N,  1e10,  dtype=np.float32)
    ro32 = ro.astype(np.float32)
    rd32 = rd.astype(np.float32)
    for _ in range(steps):
        alive = (t < t_max)
        if not alive.any(): break
        p   = ro32[alive] + rd32[alive] * t[alive, np.newaxis]
        d   = sdf_fn(p).astype(np.float32)
        y2  = d * d / (2.0 * np.maximum(ph[alive], 1e-6))
        ph[alive] = d
        dist = np.sqrt(np.maximum(t[alive]**2 - y2, 1e-7))
        sh[alive] = np.minimum(sh[alive], k * dist / np.maximum(t[alive], 1e-5))
        t[alive] += np.maximum(d, 0.008)
        blocked   = alive.copy()
        blocked[alive] &= (d < 0.0015)
        sh[blocked] = 0.0
    return np.clip(sh, 0.0, 1.0)

# ──────────────────────────────────────────────────────────────────────
#  SDF NORMAL — tetrahedron method (4 SDF evals vs 6 for central diff)
# ──────────────────────────────────────────────────────────────────────
_TET = np.array([[1,-1,-1],[-1,-1,1],[-1,1,-1],[1,1,1]], dtype=np.float32)

def _sdf_normal(sdf_fn, p, eps=0.0026):
    e = _TET * eps                                            # (4,3)
    n  = _TET[0] * sdf_fn(p + e[0])[:,np.newaxis]
    n += _TET[1] * sdf_fn(p + e[1])[:,np.newaxis]
    n += _TET[2] * sdf_fn(p + e[2])[:,np.newaxis]
    n += _TET[3] * sdf_fn(p + e[3])[:,np.newaxis]
    return _normalize(n.astype(np.float32))

# ──────────────────────────────────────────────────────────────────────
#  RAY MARCHER
# ──────────────────────────────────────────────────────────────────────
def _march_rays(sdf_fn, ray_o, ray_d, max_steps=96, t_min=0.05,
                t_max=28.0, tol=0.0010):
    N    = ray_o.shape[0]
    t    = np.full(N, t_min, dtype=np.float32)
    hit  = np.zeros(N, dtype=bool)
    ro32 = ray_o.astype(np.float32)
    rd32 = ray_d.astype(np.float32)
    for _ in range(max_steps):
        alive = (~hit) & (t < t_max)
        if not alive.any(): break
        p = ro32[alive] + rd32[alive] * t[alive, np.newaxis]
        d = sdf_fn(p).astype(np.float32)
        t[alive]  += np.maximum(d, 3e-4)
        hit[alive] |= (d < tol)
    return t, hit

# ──────────────────────────────────────────────────────────────────────
#  MATERIAL SYSTEM  (position-based zones)
# ──────────────────────────────────────────────────────────────────────
def _material(world_pos):
    """Returns albedo(N,3), roughness(N,1), metallic(N,1)."""
    x = world_pos[:,0]; y = world_pos[:,1]; z = world_pos[:,2]

    # Metallic zone: slide+barrel at y > 0.32 (after ty=0.90 transform,
    # slide spans from y=0.38 to y=1.42)
    metal_t = np.clip((y - 0.38) / 0.24, 0.0, 1.0)      # (N,)

    # Bore/crown polished zone
    in_bore = (np.sqrt((y-1.13)**2 + z**2) < 0.21).astype(np.float32)

    # Controls zone (slide stop, takedown, mag release) — slightly lighter metal
    ctrl_mask = (
        ((np.abs(x - 0.36) < 0.70) & (y > 0.040) & (y < 0.125)) |   # slide stop
        ((np.abs(x + 0.96) < 0.52) & (y > -0.075) & (y < -0.038)) |  # takedown
        ((np.abs(x + 1.45) < 0.08) & (y > -0.165) & (y < -0.100))    # mag release
    ).astype(np.float32)

    # ── Tennifer (FNC) slide: very dark, micro-rough ──────────────────
    albedo_metal = np.array([0.062, 0.066, 0.076])  # near-black tenifer
    rough_metal  = 0.34

    # ── Polymer frame: matte glass-filled nylon ───────────────────────
    albedo_poly  = np.array([0.025, 0.025, 0.029])
    rough_poly   = 0.84

    # ── Controls: Parkerized / slightly lighter steel ─────────────────
    albedo_ctrl  = np.array([0.09, 0.09, 0.10])
    rough_ctrl   = 0.42

    metal_t3 = metal_t[:,np.newaxis]                                   # (N,1)
    albedo   = albedo_poly*(1-metal_t3) + albedo_metal*metal_t3        # (N,3)
    roughness= (rough_poly*(1-metal_t) + rough_metal*metal_t)[:,np.newaxis]  # (N,1)
    metallic = np.clip(metal_t*0.95, 0., 1.)[:,np.newaxis]             # (N,1)

    # Mix in controls  (ctrl3 already (N,1) — no extra newaxis!)
    ctrl3 = ctrl_mask[:,np.newaxis]                         # (N,1)
    albedo    = albedo*(1-ctrl3)    + albedo_ctrl*ctrl3     # (N,3)
    roughness = roughness*(1-ctrl3) + rough_ctrl*ctrl3      # (N,1)
    metallic  = metallic*(1-ctrl3)  + 0.88*ctrl3           # (N,1)

    # ── Polymer grip stippling (procedural roughness, NOT geometry) ────
    #    Pyramidal hash — adds ±0.06 roughness variation
    p18  = world_pos * 20.0
    stip = np.abs(np.sin(p18[:,0]*13.1+p18[:,1]*78.9+p18[:,2]*45.3)) * 43758.5 % 1.0
    # FIX: ensure stip is (N,1) before broadcasting with roughness (N,1)
    stip_2d = stip[:,np.newaxis] * 0.055 * (1.0 - metal_t[:,np.newaxis])
    roughness = np.clip(roughness + stip_2d, 0., 1.)

    # ── Slide machining micro-scratches (along X axis) ────────────────
    scr_freq = x * 36.0
    scr      = (np.abs(np.sin(scr_freq*7.12))
              * np.abs(np.sin(scr_freq*13.5+1.08)) * 0.5)
    scr_mask = ((scr > 0.44).astype(np.float32) * metal_t)[:,np.newaxis]
    roughness = roughness - scr_mask*0.16
    albedo    = albedo    + scr_mask*np.array([0.09,0.09,0.10])

    # ── Polished bore / crown ─────────────────────────────────────────
    bore3 = in_bore[:,np.newaxis]
    albedo    = albedo*(1-bore3)    + np.array([0.13,0.13,0.145])*bore3
    roughness = roughness*(1-bore3) + 0.16*bore3
    metallic  = np.clip(metallic + bore3*0.90, 0., 1.)

    roughness = np.clip(roughness, 0.04, 1.0)
    return albedo, roughness, metallic

# ──────────────────────────────────────────────────────────────────────
#  STUDIO LIGHT RIG  (5 lights: key, fill, back, rim, top-fill)
# ──────────────────────────────────────────────────────────────────────
LIGHTS = [
    # (direction_xyz, color_rgb × intensity)
    ( np.array([ 1.10, 2.20, 0.85]), np.array([1.08,1.00,0.88]) * 3.80 ),  # Key
    ( np.array([-0.95, 0.60,-1.10]), np.array([0.32,0.40,0.58]) * 1.40 ),  # Fill (cool)
    ( np.array([ 0.00,-1.10, 1.40]), np.array([0.58,0.56,0.52]) * 0.90 ),  # Back / edge
    ( np.array([-0.50, 0.20,-0.80]), np.array([0.72,0.68,0.64]) * 0.55 ),  # Rim
    ( np.array([ 0.10, 3.50,-0.30]), np.array([0.68,0.72,0.82]) * 1.10 ),  # Overhead softbox
]
# Normalize light directions
LIGHTS = [(_normalize(l[np.newaxis])[0], c) for l,c in LIGHTS]

# ──────────────────────────────────────────────────────────────────────
#  RENDER FRAME
# ──────────────────────────────────────────────────────────────────────
def render_frame(sdf_model, cam_pos, cam_target, W=800, H=800,
                 fov_deg=42.0, chunk=0, n_threads=1):
    """chunk=0 → single full-image batch (fastest: numpy internal threading).
    chunk>0 → split into sub-batches (use only if RAM limited)."""
    t_start = time.perf_counter()

    cam_pos_np    = np.array(cam_pos,    dtype=np.float32)
    cam_target_np = np.array(cam_target, dtype=np.float32)

    forward = _normalize((cam_target_np - cam_pos_np)[np.newaxis])[0]
    wup = np.array([0., 1., 0.], dtype=np.float32)
    if abs(np.dot(forward, wup)) > 0.98:
        wup = np.array([0., 0., 1.], dtype=np.float32)
    right   = _normalize(np.cross(forward, wup)[np.newaxis])[0]
    up      = np.cross(right, forward).astype(np.float32)

    fy   = np.float32(np.tan(np.radians(fov_deg)*0.5))
    ar   = np.float32(W / H)
    u    = np.linspace(-1.0, 1.0, W, dtype=np.float32) * fy * ar
    v    = np.linspace(-1.0, 1.0, H, dtype=np.float32) * fy
    uu, vv = np.meshgrid(u, v)
    directions = (uu[..., np.newaxis] * right
                - vv[..., np.newaxis] * up
                + forward)
    N     = W * H
    ray_d = _normalize(directions.reshape(N, 3)).astype(np.float32)
    ray_o = np.broadcast_to(cam_pos_np, (N, 3)).copy().astype(np.float32)

    _fast = sdf_model.make_evaluator()
    def sdf_batch(p): return _fast(p[:,0], p[:,1], p[:,2])

    bg_top = np.array([0.055,0.072,0.108], dtype=np.float32)
    bg_bot = np.array([0.018,0.018,0.022], dtype=np.float32)
    t_bg   = np.clip(ray_d[:,1]*0.5+0.5, 0, 1)
    color  = (bg_bot + t_bg[:,np.newaxis]*(bg_top-bg_bot)).astype(np.float32)

    # ── Single-batch render (fast path) ───────────────────────────────
    def _render_batch(ro, rd):
        t, hit = _march_rays(sdf_batch, ro, rd, max_steps=110, tol=0.0009)
        if not hit.any():
            return None, hit
        p_hit  = (ro[hit] + rd[hit]*t[hit, np.newaxis]).astype(np.float32)
        n_raw  = _sdf_normal(sdf_batch, p_hit)
        flip   = ((n_raw*(-rd[hit])).sum(-1) < 0).astype(np.float32)[:,np.newaxis]
        n_hit  = n_raw*(1-2*flip)

        albedo, roughness, metallic = _material(p_hit)
        metal_t = np.clip((p_hit[:,1]-0.38)/0.24, 0., 1.)
        n_hit   = _perturb_normal_stipple(n_hit, p_hit, metal_t)

        v_dir = _normalize(cam_pos_np - p_hit)
        f0    = (0.04*(1-metallic) + albedo*metallic).astype(np.float32)
        p_bias= (p_hit + n_hit*0.022).astype(np.float32)

        ao     = _calc_ao(sdf_batch, p_hit, n_hit, steps=6, max_dist=0.75)
        ao_col = ao[:,np.newaxis]

        Lo = np.zeros_like(p_hit, dtype=np.float32)
        for l_dir, c_light in LIGHTS:
            l_broad = np.broadcast_to(l_dir, p_hit.shape)
            if c_light.max() > 1.5:
                l_rep = np.broadcast_to(l_dir, p_bias.shape).copy()
                sh    = _soft_shadow(sdf_batch, p_bias, l_rep, k=12.0, steps=16)
                sh_c  = sh[:,np.newaxis]
            else:
                sh_c  = np.float32(1.0)
            Lo += _cook_torrance(n_hit,v_dir,l_broad,albedo,metallic,roughness,c_light)*sh_c

        ambient = _ibl_ambient(n_hit, albedo, metallic, roughness, f0)
        hdr = (Lo*ao_col + ambient*ao_col).astype(np.float32)
        ldr = _aces(hdr)**(np.float32(1/2.2))
        return np.clip(ldr, 0, 1).astype(np.float32), hit

    bsz = N if chunk <= 0 else chunk
    for start in range(0, N, bsz):
        end  = min(start+bsz, N)
        c, hit = _render_batch(ray_o[start:end], ray_d[start:end])
        if c is not None:
            color[start:end][hit] = c

    img = color.reshape(H, W, 3)

    # ── Post-processing ───────────────────────────────────────────────
    rng   = np.random.default_rng(42)
    grain = rng.standard_normal((H,W,1)).astype(np.float32)*0.012
    img   = np.clip(img + grain, 0, 1)

    cx    = (np.arange(W)/(W-1)-0.5)*2
    cy    = (np.arange(H)/(H-1)-0.5)*2
    xx,yy = np.meshgrid(cx,cy)
    vig   = np.clip(1.0 - (xx**2+yy**2)*0.45, 0.35, 1.0)[:,:,np.newaxis].astype(np.float32)
    img   = img * vig

    lum   = img.mean(axis=-1, keepdims=True)
    img   = np.clip(lum + (img-lum)*1.12, 0, 1)

    img_u8  = (np.clip(img,0,1)*255).astype(np.uint8)
    elapsed = time.perf_counter()-t_start
    return img_u8, elapsed

# ──────────────────────────────────────────────────────────────────────
#  PIPELINE
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(regen_mesh=False, render_views=True):
    project_root = os.path.dirname(os.path.dirname(__file__))
    assets_dir   = os.path.join(project_root, "assets")
    output_dir   = os.path.join(assets_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=== GLOCK-19 CINEMA PIPELINE v5 ===")

    camera_manager = CameraConfig(assets_dir)
    camera_manager.generate_views()

    t0 = time.perf_counter()
    glock = GlockSDF()
    print(f"GlockSDF built in {time.perf_counter()-t0:.3f}s")

    if regen_mesh:
        obj_path = os.path.join(assets_dir, "glock_procedural.obj")
        exporter = MeshExporter(glock, resolution=320, grid_bounds=5.2)
        exporter.generate_obj(obj_path)

    if render_views:
        try:
            from PIL import Image
        except ImportError:
            print("[WARN] Pillow missing — skipping renders"); return

        # Cinematic camera rig — gun center ≈ (-0.3, -1.1, 0)
        # (slide top ≈ y=1.42, mag bottom ≈ y=-3.90, center ≈ y=-1.24)
        CTR = (-0.25, -1.05, 0.0)
        VIEWS = [
            # name,          cam_pos,              target
            ("side_L",       (-0.25,  0.30,  9.2),  CTR),
            ("side_R",       (-0.25,  0.30, -9.2),  CTR),
            ("front",        ( 9.0,   0.80,  0.0),  CTR),
            ("rear",         (-8.8,   0.50,  0.0),  CTR),
            ("iso_45",       ( 5.2,   3.00,  6.2),  CTR),
            ("iso_top",      (-0.25, 10.0,   0.15), CTR),
            ("macro_barrel", ( 5.2,   2.10,  2.4),  ( 3.5, 1.8, 0.0)),
            ("macro_grip",   (-4.2,  -1.4,   4.2),  (-2.1,-2.9, 0.0)),
            ("macro_trigger",(-1.0,  -0.8,   5.0),  (-0.5,-0.6, 0.0)),
            ("hero_cinema",  ( 3.2,   2.80,  7.8),  CTR),
            ("3q_slide",     ( 2.0,   2.50,  6.5),  ( 0.5, 0.8, 0.0)),
            ("hero_dark",    (-1.5,   1.50, -7.0),  CTR),
        ]

        RES = 800
        for name, cam_pos, target in VIEWS:
            print(f"  [RENDER] {name} ...", end=" ", flush=True)
            img_arr, el = render_frame(glock, cam_pos, target, W=RES, H=RES)
            out = os.path.join(output_dir, f"sw_{name}.png")
            Image.fromarray(img_arr,"RGB").save(out)
            mpps = (RES*RES/el)/1e6
            print(f"{el:.1f}s  ({mpps:.2f} Mpix/s) → {out}")

        # 3×4 contact sheet
        SHEET_VIEWS = VIEWS[:12]
        cols,rows = 4,3
        sheet = Image.new("RGB",(RES*cols, RES*rows),(8,8,8))
        for idx,(name,_,_) in enumerate(SHEET_VIEWS):
            p = os.path.join(output_dir, f"sw_{name}.png")
            if os.path.exists(p):
                tile = Image.open(p)
                sheet.paste(tile, ((idx%cols)*RES, (idx//cols)*RES))
        sheet.save(os.path.join(output_dir,"sw_contact_sheet.png"))
        print(f"  [SHEET] → {os.path.join(output_dir,'sw_contact_sheet.png')}")

    print("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    run_pipeline(regen_mesh=False, render_views=True)