# ═══════════════════════════════════════════════════════════════════════════
#  GLOCK 19 Gen4 — Cinema-Grade SDF  v4.0
#  Scale: 1 unit ≈ 25 mm
#  Axes: X = barrel forward (+X) | Y = up (+Y) | Z = shooter-left (+Z)
#  Real Glock 19 dims: 187mm × 128mm × 30mm  →  7.5 × 5.1 × 1.2 units
# ═══════════════════════════════════════════════════════════════════════════
import math
import numpy as np


class Vector3:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x: float, y: float, z: float):
        self.x = x; self.y = y; self.z = z


class SDFNode:
    def evaluate(self, point): raise NotImplementedError
    def evaluate_np(self, x, y, z): raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
#  PRIMITIVE SDFs
# ══════════════════════════════════════════════════════════════════════

class RoundBox(SDFNode):
    def __init__(self, size: Vector3, radius: float):
        self.size = size; self.radius = radius
    def evaluate(self, p):
        qx = abs(p.x)-self.size.x; qy = abs(p.y)-self.size.y; qz = abs(p.z)-self.size.z
        return (math.sqrt(max(qx,0)**2+max(qy,0)**2+max(qz,0)**2)
                + min(max(qx,max(qy,qz)),0.) - self.radius)
    def evaluate_np(self, x, y, z):
        qx=np.abs(x)-self.size.x; qy=np.abs(y)-self.size.y; qz=np.abs(z)-self.size.z
        return (np.sqrt(np.maximum(qx,0)**2+np.maximum(qy,0)**2+np.maximum(qz,0)**2)
                + np.minimum(np.maximum(qx,np.maximum(qy,qz)),0.) - self.radius)

class Sphere(SDFNode):
    def __init__(self, r): self.r = r
    def evaluate(self, p): return math.sqrt(p.x**2+p.y**2+p.z**2) - self.r
    def evaluate_np(self, x, y, z): return np.sqrt(x**2+y**2+z**2) - self.r

class CylinderX(SDFNode):
    def __init__(self, radius): self.radius = radius
    def evaluate(self, p): return math.sqrt(p.y**2+p.z**2) - self.radius
    def evaluate_np(self, x, y, z): return np.sqrt(y**2+z**2) - self.radius

class CylinderY(SDFNode):
    def __init__(self, radius): self.radius = radius
    def evaluate(self, p): return math.sqrt(p.x**2+p.z**2) - self.radius
    def evaluate_np(self, x, y, z): return np.sqrt(x**2+z**2) - self.radius

class CappedCylX(SDFNode):
    """Finite capped cylinder along X axis."""
    def __init__(self, radius, half_len): self.r=radius; self.hl=half_len
    def evaluate(self, p):
        dyz=math.sqrt(p.y**2+p.z**2)-self.r; dx=abs(p.x)-self.hl
        return min(max(dyz,dx),0.)+math.sqrt(max(dyz,0)**2+max(dx,0)**2)
    def evaluate_np(self, x, y, z):
        dyz=np.sqrt(y**2+z**2)-self.r; dx=np.abs(x)-self.hl
        return np.minimum(np.maximum(dyz,dx),0.)+np.sqrt(np.maximum(dyz,0)**2+np.maximum(dx,0)**2)

class CappedCylY(SDFNode):
    """Finite capped cylinder along Y axis."""
    def __init__(self, radius, half_h): self.r=radius; self.hh=half_h
    def evaluate(self, p):
        dxz=math.sqrt(p.x**2+p.z**2)-self.r; dy=abs(p.y)-self.hh
        return min(max(dxz,dy),0.)+math.sqrt(max(dxz,0)**2+max(dy,0)**2)
    def evaluate_np(self, x, y, z):
        dxz=np.sqrt(x**2+z**2)-self.r; dy=np.abs(y)-self.hh
        return np.minimum(np.maximum(dxz,dy),0.)+np.sqrt(np.maximum(dxz,0)**2+np.maximum(dy,0)**2)


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMS + CSG
# ══════════════════════════════════════════════════════════════════════

class Transform(SDFNode):
    def __init__(self, child, tx=0., ty=0., tz=0., rot_z_deg=0.):
        self.child=child; self.tx=tx; self.ty=ty; self.tz=tz
        ang=math.radians(-rot_z_deg)
        self.cz=math.cos(ang); self.sz=math.sin(ang)
    def evaluate(self, p):
        px=p.x-self.tx; py=p.y-self.ty; pz=p.z-self.tz
        return self.child.evaluate(Vector3(px*self.cz-py*self.sz, px*self.sz+py*self.cz, pz))
    def evaluate_np(self, x, y, z):
        px=x-self.tx; py=y-self.ty; pz=z-self.tz
        return self.child.evaluate_np(px*self.cz-py*self.sz, px*self.sz+py*self.cz, pz)

class CSGUnion(SDFNode):
    def __init__(self,a,b): self.a=a; self.b=b
    def evaluate(self,p): return min(self.a.evaluate(p), self.b.evaluate(p))
    def evaluate_np(self,x,y,z): return np.minimum(self.a.evaluate_np(x,y,z), self.b.evaluate_np(x,y,z))

class CSGDifference(SDFNode):
    def __init__(self,s,c): self.s=s; self.c=c
    def evaluate(self,p): return max(self.s.evaluate(p), -self.c.evaluate(p))
    def evaluate_np(self,x,y,z): return np.maximum(self.s.evaluate_np(x,y,z), -self.c.evaluate_np(x,y,z))

class CSGIntersection(SDFNode):
    def __init__(self,a,b): self.a=a; self.b=b
    def evaluate(self,p): return max(self.a.evaluate(p), self.b.evaluate(p))
    def evaluate_np(self,x,y,z): return np.maximum(self.a.evaluate_np(x,y,z), self.b.evaluate_np(x,y,z))

class SmoothUnion(SDFNode):
    def __init__(self,a,b,k=0.08): self.a=a; self.b=b; self.k=k
    def evaluate(self,p):
        da=self.a.evaluate(p); db=self.b.evaluate(p)
        h=max(0.,min(1.,0.5+0.5*(db-da)/self.k))
        return da*h+db*(1-h)-self.k*h*(1-h)
    def evaluate_np(self,x,y,z):
        da=self.a.evaluate_np(x,y,z); db=self.b.evaluate_np(x,y,z)
        h=np.clip(0.5+0.5*(db-da)/self.k,0.,1.)
        return da*h+db*(1-h)-self.k*h*(1-h)

class SmoothDiff(SDFNode):
    """Smooth subtraction: smooth_max(solid, -cutter)."""
    def __init__(self,solid,cutter,k=0.04): self.solid=solid; self.cutter=cutter; self.k=k
    def evaluate(self,p):
        ds=self.solid.evaluate(p); dc=self.cutter.evaluate(p)
        h=max(0.,min(1.,0.5+0.5*(ds+dc)/self.k))
        return ds*h-dc*(1-h)+self.k*h*(1-h)
    def evaluate_np(self,x,y,z):
        ds=self.solid.evaluate_np(x,y,z); dc=self.cutter.evaluate_np(x,y,z)
        h=np.clip(0.5+0.5*(ds+dc)/self.k,0.,1.)
        return ds*h-dc*(1-h)+self.k*h*(1-h)


# ══════════════════════════════════════════════════════════════════════
#  GLOCK 19 GEN4 — COMPLETE CINEMA SDF  v4
# ══════════════════════════════════════════════════════════════════════

class GlockSDF(SDFNode):
    def __init__(self):
        upper   = self._build_upper()
        upper_t = Transform(upper, ty=0.90)   # seat slide/barrel on frame

        lower   = self._build_lower()
        self.root = CSGUnion(upper_t, lower)

    # ── UPPER ASSEMBLY ────────────────────────────────────────────────
    def _build_upper(self):
        return CSGUnion(self._build_slide(), self._build_barrel())

    def _build_slide(self):
        # Main slide body: 186mm L × 25.5mm H × 28mm W  (real Glock 19)
        slide = RoundBox(Vector3(3.82, 0.605, 0.558), radius=0.048)

        # Front chamfers — iconic Glock diagonal muzzle cut (both sides)
        for sz in [+1, -1]:
            cut = Transform(RoundBox(Vector3(0.58, 0.76, 0.52), radius=0.0),
                            tx=3.54, ty=0.36, tz=sz*0.70, rot_z_deg=-22.)
            slide = CSGDifference(slide, cut)

        # Top flat bevel (subtle crown on slide top-rear)
        top_bev = Transform(RoundBox(Vector3(1.60, 0.040, 0.62), radius=0.018),
                            tx=-2.20, ty=0.638)
        slide = CSGDifference(slide, top_bev)

        # Bottom void — slide rides over barrel / frame rails
        void = Transform(RoundBox(Vector3(3.94, 0.502, 0.352), radius=0.042),
                         tx=0., ty=-0.144)
        slide = CSGDifference(slide, void)

        # Ejection port (right side -Z, chamfered opening)
        eject = Transform(RoundBox(Vector3(0.64, 0.498, 0.490), radius=0.042),
                          tx=0.62, ty=0.44, tz=-0.272)
        slide = CSGDifference(slide, eject)

        # Extractor spring tab (right side, proud nub)
        ext_body = Transform(RoundBox(Vector3(0.248, 0.034, 0.022), radius=0.010),
                             tx=0.64, ty=0.422, tz=-0.554)
        ext_tip  = Transform(Sphere(0.031), tx=0.88, ty=0.422, tz=-0.554)
        slide = CSGUnion(slide, ext_body)
        slide = CSGUnion(slide, ext_tip)

        # Loaded chamber indicator bump (tiny nub, rear-extractor)
        lci = Transform(RoundBox(Vector3(0.060, 0.022, 0.016), radius=0.012),
                        tx=0.16, ty=0.614, tz=-0.508)
        slide = CSGUnion(slide, lci)

        # Rear serrations: 14 grooves, sharper V-profile
        for i in range(14):
            s = Transform(RoundBox(Vector3(0.030, 0.552, 1.28), radius=0.016),
                          tx=-3.58+i*0.148, ty=0.10)
            slide = CSGDifference(slide, s)

        # Front serrations: 10 grooves (Gen4/5 forward slide press)
        for i in range(10):
            s = Transform(RoundBox(Vector3(0.025, 0.552, 1.28), radius=0.014),
                          tx=2.12+i*0.148, ty=0.10)
            slide = CSGDifference(slide, s)

        # Rear plate / backplate (striker housing)
        rp = Transform(RoundBox(Vector3(0.044, 0.572, 0.552), radius=0.018),
                       tx=-3.81, ty=0.06)
        slide = CSGUnion(slide, rp)
        # Backplate retaining notch
        bp_notch = Transform(RoundBox(Vector3(0.018, 0.032, 0.068), radius=0.008),
                             tx=-3.82, ty=0.42)
        slide = CSGDifference(slide, bp_notch)

        # Optic/sight dovetail groove (top-rear)
        op = Transform(RoundBox(Vector3(1.12, 0.022, 0.238), radius=0.007),
                       tx=-2.42, ty=0.618)
        slide = CSGDifference(slide, op)

        # Striker/firing pin channel (tiny centered groove on top)
        fp_ch = Transform(RoundBox(Vector3(2.80, 0.015, 0.042), radius=0.006),
                          tx=-0.50, ty=0.624)
        slide = CSGDifference(slide, fp_ch)

        # Front sight (tall blade with white-dot recess)
        fs = Transform(RoundBox(Vector3(0.038, 0.128, 0.058), radius=0.012),
                       tx=3.52, ty=0.690)
        fd = Transform(Sphere(0.022), tx=3.52, ty=0.726)
        fs = CSGDifference(fs, fd)
        slide = CSGUnion(slide, fs)

        # Rear sight (U-notch, two ears with white-dot recesses)
        for tz, sx in [(+0.212, 1), (-0.212, -1)]:
            ear = Transform(RoundBox(Vector3(0.064, 0.098, 0.076), radius=0.012),
                            tx=-3.10, ty=0.690, tz=tz)
            dot = Transform(Sphere(0.020), tx=-3.10, ty=0.716, tz=tz)
            ear = CSGDifference(ear, dot)
            slide = CSGUnion(slide, ear)
        notch = Transform(RoundBox(Vector3(0.068, 0.096, 0.114), radius=0.016),
                          tx=-3.10, ty=0.698)
        slide = CSGDifference(slide, notch)

        return slide

    def _build_barrel(self):
        # Chamber block (upper-rear, sits above frame at breech)
        chamber = Transform(RoundBox(Vector3(0.50, 0.44, 0.424), radius=0.026),
                            tx=0.60, ty=0.24)

        # Hood (locking lug block on top, trapezoidal)
        hood = Transform(RoundBox(Vector3(0.112, 0.070, 0.240), radius=0.010),
                         tx=1.06, ty=0.420)

        # Barrel body (round, bounded by crisp half-length)
        b_cyl   = Transform(CylinderX(0.246), ty=0.24)
        b_bound = Transform(RoundBox(Vector3(2.10, 0.268, 0.268), radius=0.004),
                            tx=1.78, ty=0.24)
        b_body  = CSGIntersection(b_cyl, b_bound)

        # Fluted barrel exterior: 6 shallow axial grooves (visual detail)
        for ang_deg in range(0, 360, 60):
            ang = math.radians(ang_deg)
            fy2 = math.sin(ang) * 0.214
            fz  = math.cos(ang) * 0.214
            flute = Transform(RoundBox(Vector3(2.05, 0.028, 0.028), radius=0.022),
                              tx=1.78, ty=0.24+fy2, tz=fz)
            b_body = CSGDifference(b_body, flute)

        # Muzzle crown (recessed polished ring at muzzle face)
        cr_out  = Transform(RoundBox(Vector3(0.078, 0.276, 0.276), radius=0.018),
                            tx=3.81, ty=0.24)
        cr_icyl = Transform(CylinderX(0.196), ty=0.24)
        cr_imsk = Transform(RoundBox(Vector3(0.124, 0.214, 0.214), radius=0.010),
                            tx=3.81, ty=0.24)
        crown   = CSGDifference(cr_out, CSGIntersection(cr_icyl, cr_imsk))

        # Bore (9mm ⌀9mm → r=0.176)
        bore_c  = Transform(CylinderX(0.176), ty=0.24)
        bore_m  = Transform(RoundBox(Vector3(2.18, 0.194, 0.194), radius=0.004),
                            tx=1.74, ty=0.24)
        bore    = CSGIntersection(bore_c, bore_m)

        # Recoil spring guide rod (visible at muzzle, below barrel)
        rod_cyl  = Transform(CylinderX(0.072), ty=-0.128)
        rod_mask = Transform(RoundBox(Vector3(2.40, 0.084, 0.084), radius=0.006),
                             tx=1.50, ty=-0.128)
        rod_tip  = Transform(Sphere(0.084), tx=3.80, ty=-0.128)
        rod = CSGIntersection(rod_cyl, rod_mask)
        rod = CSGUnion(rod, rod_tip)

        barrel = CSGUnion(chamber, b_body)
        barrel = CSGUnion(barrel, hood)
        barrel = CSGDifference(barrel, bore)
        barrel = CSGUnion(barrel, crown)
        barrel = CSGUnion(barrel, rod)
        return barrel

    # ── LOWER ASSEMBLY ────────────────────────────────────────────────
    def _build_lower(self):
        return CSGUnion(self._build_frame(), self._build_trigger())

    def _build_frame(self):
        # Main dust-cover / rail section
        frame = Transform(RoundBox(Vector3(3.76, 0.365, 0.575), radius=0.072),
                          tx=0.04, ty=0.04)

        # Slide guide rails (steel inserts, two parallel)
        for rz in [+0.380, -0.380]:
            g = Transform(RoundBox(Vector3(3.58, 0.040, 0.036), radius=0.009),
                          tx=0.04, ty=0.374, tz=rz)
            frame = CSGUnion(frame, g)

        # Picatinny rail (MIL-STD-1913)
        rail_slot = Transform(RoundBox(Vector3(1.82, 0.062, 0.143), radius=0.007),
                              tx=0.88, ty=-0.289)
        frame = CSGDifference(frame, rail_slot)
        for rz in [+0.162, -0.162]:
            rw = Transform(RoundBox(Vector3(1.80, 0.042, 0.020), radius=0.004),
                           tx=0.88, ty=-0.278, tz=rz)
            frame = CSGUnion(frame, rw)
        for ri in range(4):
            xs = Transform(RoundBox(Vector3(0.046, 0.052, 0.172), radius=0.005),
                           tx=0.12+ri*0.54, ty=-0.276)
            frame = CSGDifference(frame, xs)

        # Trigger guard (anatomic shape with squared front hook)
        g_out  = Transform(RoundBox(Vector3(0.900, 0.530, 0.278), radius=0.108),
                           tx=-0.08, ty=-0.635)
        g_in   = Transform(RoundBox(Vector3(0.640, 0.395, 0.360), radius=0.112),
                           tx=-0.18, ty=-0.635)
        guard  = CSGDifference(g_out, g_in)
        # Squared front hook for support-hand index finger
        g_hook_cut = Transform(RoundBox(Vector3(0.28, 0.12, 0.35), radius=0.04),
                               tx=-0.92, ty=-0.82)
        g_hook     = Transform(RoundBox(Vector3(0.058, 0.098, 0.268), radius=0.024),
                               tx=-0.954, ty=-0.732)
        guard = CSGDifference(guard, g_hook_cut)
        guard = CSGUnion(guard, g_hook)
        frame = CSGUnion(frame, guard)

        # Ergonomic undercut below trigger guard
        ucut = Transform(RoundBox(Vector3(0.42, 0.312, 0.70), radius=0.212),
                         tx=-1.14, ty=-0.532)
        frame = CSGDifference(frame, ucut)

        # Spine (frame-to-grip transition)
        spine = Transform(RoundBox(Vector3(0.55, 0.82, 0.562), radius=0.088),
                          tx=-1.38, ty=-0.562)
        frame = SmoothUnion(frame, spine, k=0.058)

        # Grip body with 20.5° rake
        _gx, _gy, _gr = -2.26, -2.08, -20.5
        grip_main = Transform(RoundBox(Vector3(1.12, 1.88, 0.548), radius=0.092),
                              tx=_gx, ty=_gy, rot_z_deg=_gr)
        grip_back = Transform(RoundBox(Vector3(1.08, 1.82, 0.090), radius=0.038),
                              tx=_gx-0.31, ty=_gy, rot_z_deg=_gr)
        grip = SmoothUnion(grip_main, grip_back, k=0.060)

        # 3 finger grooves on front strap (Gen3 style)
        for ly in [0.82, 0.20, -0.42]:
            groove_loc = Transform(RoundBox(Vector3(0.070, 0.072, 0.520), radius=0.046),
                                   tx=1.04, ty=ly)
            groove_wld = Transform(groove_loc, tx=_gx, ty=_gy, rot_z_deg=_gr)
            grip = CSGDifference(grip, groove_wld)

        # Mag-well flare
        mw  = Transform(RoundBox(Vector3(1.06, 0.220, 0.540), radius=0.058),
                        tx=_gx+0.13, ty=-4.16, rot_z_deg=_gr)
        mwf = Transform(RoundBox(Vector3(1.06, 0.095, 0.582), radius=0.038),
                        tx=_gx+0.13, ty=-4.35, rot_z_deg=_gr)
        mag_sys = SmoothUnion(mw, mwf, k=0.038)

        # Magazine baseplate (visible at bottom)
        bp  = Transform(RoundBox(Vector3(1.10, 0.152, 0.562), radius=0.046),
                        tx=_gx+0.13, ty=-4.528, rot_z_deg=_gr)
        bpr = Transform(RoundBox(Vector3(1.06, 0.040, 0.448), radius=0.014),
                        tx=_gx+0.13, ty=-4.692, rot_z_deg=_gr)
        mag_base = SmoothUnion(bp, bpr, k=0.024)

        frame = SmoothUnion(frame, grip,    k=0.040)
        frame = SmoothUnion(frame, mag_sys, k=0.048)
        frame = SmoothUnion(frame, mag_base,k=0.028)

        # ── Controls ────────────────────────────────────────────────

        # Mag release button (left side, oval, knurled)
        mr = Transform(RoundBox(Vector3(0.063, 0.116, 0.073), radius=0.031),
                       tx=-1.55, ty=-0.152, tz=0.578)
        for bi in range(3):
            bump = Transform(Sphere(0.014), tx=-1.55+(bi-1)*0.038, ty=-0.134, tz=0.600)
            mr = CSGUnion(mr, bump)
        frame = CSGUnion(frame, mr)

        # Slide stop lever (left side, above trigger guard)
        sl_body = Transform(RoundBox(Vector3(0.698, 0.066, 0.027), radius=0.025),
                            tx=0.32, ty=0.064, tz=0.586)
        sl_pad  = Transform(RoundBox(Vector3(0.113, 0.116, 0.035), radius=0.031),
                            tx=-0.26, ty=0.090, tz=0.588)
        sl_pin  = Transform(CappedCylY(0.038, 0.019), tx=0.32, ty=0.064, tz=0.602)
        slide_stop = SmoothUnion(sl_body, sl_pad, k=0.038)
        slide_stop = CSGUnion(slide_stop, sl_pin)
        frame = CSGUnion(frame, slide_stop)

        # Takedown lever (left side, below slide stop)
        tk_body = Transform(RoundBox(Vector3(0.518, 0.050, 0.023), radius=0.019),
                            tx=-1.06, ty=-0.063, tz=0.586)
        tk_pin  = Transform(CappedCylY(0.035, 0.017), tx=-1.06, ty=-0.063, tz=0.600)
        frame = CSGUnion(frame, CSGUnion(tk_body, tk_pin))

        # Locking block & trigger pins (visible on both sides)
        for px, py in [(-0.18, -0.10), (-1.06, -0.28)]:
            for sz in [+1, -1]:
                pin = Transform(CappedCylY(0.029, 0.015), tx=px, ty=py, tz=sz*0.583)
                frame = CSGUnion(frame, pin)

        return frame

    def _build_trigger(self):
        # Curved blade
        blade = Transform(RoundBox(Vector3(0.113, 0.283, 0.130), radius=0.036),
                          tx=-0.44, ty=-0.510, rot_z_deg=9.5)
        # Bottom hook
        hook  = Transform(RoundBox(Vector3(0.078, 0.047, 0.118), radius=0.027),
                          tx=-0.570, ty=-0.778)
        # Safety lever tab (inside blade)
        saf   = Transform(RoundBox(Vector3(0.028, 0.168, 0.044), radius=0.009),
                          tx=-0.445, ty=-0.500, rot_z_deg=9.5)
        # Safety pivot bar
        piv   = Transform(RoundBox(Vector3(0.009, 0.026, 0.088), radius=0.005),
                          tx=-0.445, ty=-0.382)
        # Hollow center of blade (realistic aperture)
        hole  = Transform(RoundBox(Vector3(0.063, 0.172, 0.053), radius=0.019),
                          tx=-0.44, ty=-0.510, rot_z_deg=9.5)
        trig = SmoothUnion(blade, hook, k=0.038)
        trig = CSGUnion(trig, saf)
        trig = CSGUnion(trig, piv)
        trig = CSGDifference(trig, hole)
        return trig

    def evaluate(self, point): return self.root.evaluate(point)
    def evaluate_np(self, x, y, z): return self.root.evaluate_np(x, y, z)


# ═══════════════════════════════════════════════════════════════════════════
#  CODE-GENERATION COMPILER  — flat vectorized SDF  (no object dispatch)
#  Compiles the 263-node tree into a single Python function with ~600
#  sequential numpy statements, eliminating all Python method-dispatch
#  overhead.  Gives 5–15× speedup over the recursive tree.
# ═══════════════════════════════════════════════════════════════════════════

def _ec_roundbox(self, xe, ye, ze, L, c):
    sx=self.size.x; sy=self.size.y; sz=self.size.z; r=self.radius
    c[0]+=1; n=c[0]; qx,qy,qz,v=f"_qx{n}",f"_qy{n}",f"_qz{n}",f"_v{n}"
    L+=[ f"    {qx}=np.abs({xe})-{sx:.9g}",
         f"    {qy}=np.abs({ye})-{sy:.9g}",
         f"    {qz}=np.abs({ze})-{sz:.9g}",
         f"    {v}=(np.sqrt(np.maximum({qx},0)**2+np.maximum({qy},0)**2"
         f"+np.maximum({qz},0)**2)+np.minimum(np.maximum({qx},"
         f"np.maximum({qy},{qz})),0.)-{r:.9g})" ]
    return v

def _ec_sphere(self, xe, ye, ze, L, c):
    c[0]+=1; n=c[0]; v=f"_v{n}"
    L.append(f"    {v}=np.sqrt(({xe})**2+({ye})**2+({ze})**2)-{self.r:.9g}")
    return v

def _ec_cylx(self, xe, ye, ze, L, c):
    c[0]+=1; n=c[0]; v=f"_v{n}"
    L.append(f"    {v}=np.sqrt(({ye})**2+({ze})**2)-{self.radius:.9g}")
    return v

def _ec_cyly(self, xe, ye, ze, L, c):
    c[0]+=1; n=c[0]; v=f"_v{n}"
    L.append(f"    {v}=np.sqrt(({xe})**2+({ze})**2)-{self.radius:.9g}")
    return v

def _ec_capcylx(self, xe, ye, ze, L, c):
    c[0]+=1; n=c[0]; dyz=f"_dyz{n}"; dx=f"_dx{n}"; v=f"_v{n}"
    L+=[ f"    {dyz}=np.sqrt(({ye})**2+({ze})**2)-{self.r:.9g}",
         f"    {dx}=np.abs({xe})-{self.hl:.9g}",
         f"    {v}=(np.minimum(np.maximum({dyz},{dx}),0.)"
         f"+np.sqrt(np.maximum({dyz},0)**2+np.maximum({dx},0)**2))" ]
    return v

def _ec_capcyly(self, xe, ye, ze, L, c):
    c[0]+=1; n=c[0]; dxz=f"_dxz{n}"; dy=f"_dy{n}"; v=f"_v{n}"
    L+=[ f"    {dxz}=np.sqrt(({xe})**2+({ze})**2)-{self.r:.9g}",
         f"    {dy}=np.abs({ye})-{self.hh:.9g}",
         f"    {v}=(np.minimum(np.maximum({dxz},{dy}),0.)"
         f"+np.sqrt(np.maximum({dxz},0)**2+np.maximum({dy},0)**2))" ]
    return v

def _ec_transform(self, xe, ye, ze, L, c):
    pxe = f"({xe}-{self.tx:.9g})" if self.tx != 0 else xe
    pye = f"({ye}-{self.ty:.9g})" if self.ty != 0 else ye
    pze = f"({ze}-{self.tz:.9g})" if self.tz != 0 else ze
    if self.sz != 0:
        c[0]+=1; n=c[0]; rx=f"_rx{n}"; ry=f"_ry{n}"
        L+=[ f"    {rx}={pxe}*{self.cz:.9g}-{pye}*{self.sz:.9g}",
             f"    {ry}={pxe}*{self.sz:.9g}+{pye}*{self.cz:.9g}" ]
        return self.child.emit_code(rx, ry, pze, L, c)
    else:
        return self.child.emit_code(pxe, pye, pze, L, c)

def _ec_union(self, xe, ye, ze, L, c):
    va=self.a.emit_code(xe,ye,ze,L,c); vb=self.b.emit_code(xe,ye,ze,L,c)
    c[0]+=1; n=c[0]; v=f"_v{n}"
    L.append(f"    {v}=np.minimum({va},{vb})")
    return v

def _ec_diff(self, xe, ye, ze, L, c):
    vs=self.s.emit_code(xe,ye,ze,L,c); vc2=self.c.emit_code(xe,ye,ze,L,c)
    c[0]+=1; n=c[0]; v=f"_v{n}"
    L.append(f"    {v}=np.maximum({vs},-({vc2}))")
    return v

def _ec_intersect(self, xe, ye, ze, L, c):
    va=self.a.emit_code(xe,ye,ze,L,c); vb=self.b.emit_code(xe,ye,ze,L,c)
    c[0]+=1; n=c[0]; v=f"_v{n}"
    L.append(f"    {v}=np.maximum({va},{vb})")
    return v

def _ec_smooth_union(self, xe, ye, ze, L, c):
    vda=self.a.emit_code(xe,ye,ze,L,c); vdb=self.b.emit_code(xe,ye,ze,L,c)
    c[0]+=1; n=c[0]; vh=f"_h{n}"; v=f"_v{n}"; k=self.k
    L+=[ f"    {vh}=np.clip(0.5+0.5*({vdb}-{vda})/{k:.9g},0.,1.)",
         f"    {v}={vda}*{vh}+{vdb}*(1.-{vh})-{k:.9g}*{vh}*(1.-{vh})" ]
    return v

def _ec_smooth_diff(self, xe, ye, ze, L, c):
    vds=self.solid.emit_code(xe,ye,ze,L,c); vdc=self.cutter.emit_code(xe,ye,ze,L,c)
    c[0]+=1; n=c[0]; vh=f"_h{n}"; v=f"_v{n}"; k=self.k
    L+=[ f"    {vh}=np.clip(0.5+0.5*({vds}+{vdc})/{k:.9g},0.,1.)",
         f"    {v}={vds}*{vh}-{vdc}*(1.-{vh})+{k:.9g}*{vh}*(1.-{vh})" ]
    return v

# ── Attach emit_code methods to all SDF classes ──────────────────────
RoundBox.emit_code        = _ec_roundbox
Sphere.emit_code          = _ec_sphere
CylinderX.emit_code       = _ec_cylx
CylinderY.emit_code       = _ec_cyly
CappedCylX.emit_code      = _ec_capcylx
CappedCylY.emit_code      = _ec_capcyly
Transform.emit_code       = _ec_transform
CSGUnion.emit_code        = _ec_union
CSGDifference.emit_code   = _ec_diff
CSGIntersection.emit_code = _ec_intersect
SmoothUnion.emit_code     = _ec_smooth_union
SmoothDiff.emit_code      = _ec_smooth_diff


def compile_flat(root_node):
    """Compile an SDF tree into a single flat Python/NumPy function.

    Eliminates all Python method-dispatch overhead present in the
    recursive SDFNode tree.  The generated function contains only
    sequential numpy assignments — fast bytecode with no attribute
    lookups.

    Returns
    -------
    fn  : callable  fn(x, y, z) → np.ndarray[float32]
    src : str       generated source (for debugging / caching)
    """
    lines = ["def _sdf_eval(x, y, z):"]
    c = [0]
    result_var = root_node.emit_code("x", "y", "z", lines, c)
    lines.append(f"    return {result_var}.astype(np.float32)")
    src = "\n".join(lines)
    ns = {"np": np}
    exec(compile(src, "<sdf_compiled>", "exec"), ns)
    return ns["_sdf_eval"], src


def _glock_make_evaluator(self):
    """Return cached compiled flat evaluator for this GlockSDF instance."""
    if not hasattr(self, "_fast_eval"):
        import time as _t
        t0 = _t.perf_counter()
        self._fast_eval, self._fast_src = compile_flat(self.root)
        dt = (_t.perf_counter()-t0)*1e3
        nlines = self._fast_src.count("\n")
        print(f"  [SDF] flat evaluator compiled: {nlines} lines in {dt:.1f}ms")
    return self._fast_eval

GlockSDF.make_evaluator = _glock_make_evaluator


if __name__ == '__main__':
    import time
    t0 = time.perf_counter()
    g = GlockSDF()
    print(f"Build: {(time.perf_counter()-t0)*1e3:.1f}ms")
    for p in [(3.8,1.15,0), (0.,0.9,0), (-2.5,-2.5,0)]:
        d = g.evaluate(Vector3(*p))
        print(f"  d{p} = {d:.4f}")