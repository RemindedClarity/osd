#!/usr/bin/env python3
"""
================================================================
ORANGE SWEATER DUDE v5.0 — CINEMATIC UPGRADE
================================================================
v5.0 UPGRADES:
  - Vectorized warp_stretch & xray materials (no per-pixel loops)
  - Fixed disintegration particle seeding (smooth paths)
  - Parallax depth layers on environment particles
  - Cinematic lighting: rim light, god rays, AO, vignette
  - Smooth camera motion with easing functions
  - Dynamic teleport: multi-phase explosion + shockwave
  - Color grading with S-curve + chromatic aberration
  - Motion blur trails on particles

OSD MATERIAL TRANSFORMATIONS:
  Jurassic    -> MOLTEN LAVA (flowing, dripping)
  Egypt       -> SAND DISSOLVING (grains blowing off)
  Medieval    -> PURE FIRE (embers breaking off)
  Wild West   -> DUST GHOST (semi-transparent)
  Arcade      -> PIXELATED (RGB split, chunky)
  Cyberpunk   -> GLITCH CORRUPTED (scanlines, artifacts)
  Nuclear     -> X-RAY GHOST (transparent green glow)
  Warp        -> STAR STRETCHED (motion blur radial)

CINEMATIC BACKGROUNDS:
  Jurassic    -> Meteor strike, shockwave, ground cracks
  Egypt       -> Sandstorm wall approaching, lightning
  Medieval    -> Castle collapsing, siege weapons
  Wild West   -> Tornado forming, debris flying
  Arcade      -> Giant arcade screen (Pac-Man, Tetris)
  Cyberpunk   -> Flying cars, animated billboards
  Nuclear     -> Mushroom cloud expanding, shockwave
  Warp        -> Wormhole opening, reality tearing

CAMERA EFFECTS:
  - Screen shake with easing (ramp up/down)
  - Chromatic aberration on teleports + glitch material
  - Radial blur on explosions
  - Heat distortion on fire
  - Dutch angle tilts
  - Slow zoom drift per era
  - Gentle sway/pan

TELEPORT: OSD particles gather inward -> burst explosion -> shockwave
         -> slow drift -> snap back -> reform

RENDER TIME: ~15-25 minutes (vectorized pipeline)
================================================================
"""

import numpy as np
import math
import os
import time
import subprocess
from PIL import Image


# ============================================================
# CONFIG
# ============================================================
WORLD_SEED     = 7
DURATION       = 30.0
FPS            = 30
OSD_FILE       = "osd_character.png"
OUTPUT         = "osd_monday.mp4"

RW, RH         = 540, 960
FW, FH         = 1080, 1920
TEMP_BG        = "_osd_bg.mp4"
TEMP_UP        = "_osd_up.mp4"

N_WORLDS_TOTAL = 8
N_WORLDS_PICK  = 4
WORLD_DUR      = DURATION / N_WORLDS_PICK
FLASH_DUR      = 0.20  # Teleport flash duration
EXPLODE_DUR    = 0.15  # Disintegration duration

GCX = RW // 2
GCY = int(RH * 0.60)

_yy, _xx = np.mgrid[0:RH, 0:RW]
DIST     = np.sqrt((_xx - GCX)**2 + (_yy - GCY)**2).astype(np.float64)
ANGLE    = np.arctan2(_yy - GCY, _xx - GCX)


# ============================================================
# EASING FUNCTIONS (Step 4)
# ============================================================
def ease_in_out_cubic(t):
    """Smooth ease in/out."""
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0

def ease_out_quad(t):
    """Ease out quadratic."""
    return t * (2.0 - t)

def ease_in_quad(t):
    """Ease in quadratic."""
    return t * t


# ============================================================
# COLOR GRADING (Step 6)
# ============================================================
# Pre-compute S-curve LUT (crushed blacks, boosted highlights)
_lut_x = np.arange(256, dtype=np.float64) / 255.0
# S-curve: lift blacks to ~10, compress highlights above 240
_lut_y = np.clip(10.0 + 230.0 * (3.0 * _lut_x**2 - 2.0 * _lut_x**3), 0, 255)
COLOR_GRADE_LUT = _lut_y.astype(np.uint8)

# Per-channel gamma adjustments (slight warm shift)
_r_lut = np.clip(10.0 + 232.0 * (3.0 * _lut_x**2 - 2.0 * _lut_x**3), 0, 255).astype(np.uint8)
_g_lut = np.clip(9.0 + 228.0 * (3.0 * _lut_x**2 - 2.0 * _lut_x**3), 0, 255).astype(np.uint8)
_b_lut = np.clip(8.0 + 225.0 * (3.0 * _lut_x**2 - 2.0 * _lut_x**3), 0, 255).astype(np.uint8)
COLOR_GRADE_LUT_R = _r_lut
COLOR_GRADE_LUT_G = _g_lut
COLOR_GRADE_LUT_B = _b_lut

def apply_color_grading(frame_u8):
    """Apply S-curve color grading to final frame."""
    result = np.empty_like(frame_u8)
    result[:, :, 0] = COLOR_GRADE_LUT_R[frame_u8[:, :, 0]]
    result[:, :, 1] = COLOR_GRADE_LUT_G[frame_u8[:, :, 1]]
    result[:, :, 2] = COLOR_GRADE_LUT_B[frame_u8[:, :, 2]]
    return result

def apply_chromatic_aberration(frame_u8, intensity):
    """Offset R and B channels for chromatic aberration."""
    if intensity < 0.01:
        return frame_u8
    offset = int(np.clip(intensity * 6, 1, 8))
    result = frame_u8.copy()
    # Red channel shifts right
    result[:, offset:, 0] = frame_u8[:, :-offset, 0]
    # Blue channel shifts left
    result[:, :-offset, 2] = frame_u8[:, offset:, 2]
    return result


# ============================================================
# CINEMATIC LIGHTING (Step 3) - pre-compute masks
# ============================================================
# Vignette mask
_vig_cx, _vig_cy = RW / 2.0, RH / 2.0
_vig_dist = np.sqrt((_xx - _vig_cx)**2 + (_yy - _vig_cy)**2)
_vig_max = math.sqrt(_vig_cx**2 + _vig_cy**2)
VIGNETTE_MASK = np.clip(_vig_dist / _vig_max, 0, 1) ** 1.8
VIGNETTE_MASK = VIGNETTE_MASK.reshape(RH, RW, 1)

# Ambient occlusion (darken bottom more aggressively)
AO_MASK = np.linspace(0.0, 0.35, RH).reshape(-1, 1, 1) ** 1.5

# God ray angular masks (6 rays from glow center)
GOD_RAY_MASKS = []
for ray_i in range(8):
    ray_angle = ray_i * (2 * np.pi / 8)
    angle_diff = np.abs(ANGLE - ray_angle)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    ray_mask = np.clip(1.0 - angle_diff / 0.12, 0, 1)
    ray_falloff = np.clip(1.0 - DIST / 400, 0, 1) ** 0.5
    GOD_RAY_MASKS.append((ray_mask * ray_falloff).reshape(RH, RW, 1))


# ============================================================
# LOAD OSD IMAGE ONCE
# ============================================================
if os.path.exists(OSD_FILE):
    osd_img = Image.open(OSD_FILE).convert('RGBA')
    OSD_W, OSD_H = osd_img.size
    osd_rgba = np.array(osd_img, dtype=np.float64)
    # Scale to target height (55% of frame)
    target_h = int(RH * 0.55)
    scale = target_h / OSD_H
    new_w = int(OSD_W * scale)
    osd_img_scaled = osd_img.resize((new_w, target_h), Image.Resampling.LANCZOS)
    osd_rgba_scaled = np.array(osd_img_scaled, dtype=np.float64)
    OSD_SCALED_W, OSD_SCALED_H = new_w, target_h
    # Position (centered horizontally, lower third)
    OSD_X = (RW - OSD_SCALED_W) // 2
    OSD_Y = RH - OSD_SCALED_H - 90
    HAS_OSD = True

    # Pre-compute rim light mask from alpha edges (Step 3)
    _osd_alpha = osd_rgba_scaled[:, :, 3]
    # Compute alpha gradient magnitude for edge detection
    _grad_y = np.zeros_like(_osd_alpha)
    _grad_x = np.zeros_like(_osd_alpha)
    _grad_y[1:-1, :] = np.abs(_osd_alpha[2:, :] - _osd_alpha[:-2, :])
    _grad_x[:, 1:-1] = np.abs(_osd_alpha[:, 2:] - _osd_alpha[:, :-2])
    OSD_EDGE_MASK = np.sqrt(_grad_y**2 + _grad_x**2)
    OSD_EDGE_MASK = np.clip(OSD_EDGE_MASK / (OSD_EDGE_MASK.max() + 1e-8), 0, 1)
    # Bias toward top-right for rim light direction
    _rim_yy, _rim_xx = np.mgrid[0:OSD_SCALED_H, 0:OSD_SCALED_W]
    _rim_dir = np.clip((_rim_xx / OSD_SCALED_W) * 0.7 + (1.0 - _rim_yy / OSD_SCALED_H) * 0.3, 0, 1)
    OSD_RIM_MASK = OSD_EDGE_MASK * _rim_dir
else:
    HAS_OSD = False
    osd_rgba_scaled = None
    OSD_SCALED_W, OSD_SCALED_H = 0, 0
    OSD_X, OSD_Y = 0, 0
    OSD_EDGE_MASK = None
    OSD_RIM_MASK = None


# ============================================================
# ERA DEFINITIONS
# ============================================================
WORLDS = [
    {
        "name": "Jurassic Era",
        "bg_color": np.array([18, 12, 8], dtype=np.float64),
        "glow_col": np.array([255, 100, 40], dtype=np.float64),
        "material": "lava",
        "bg_fx": "meteor_strike",
        "camera_shake": 0.15,
        "particles_n": 800,
        "particle_colors": [np.array([80, 70, 60]), np.array([60, 50, 40])],
    },
    {
        "name": "Ancient Egypt",
        "bg_color": np.array([35, 28, 18], dtype=np.float64),
        "glow_col": np.array([230, 180, 100], dtype=np.float64),
        "material": "sand",
        "bg_fx": "sandstorm",
        "camera_shake": 0.10,
        "particles_n": 750,
        "particle_colors": [np.array([210, 180, 130]), np.array([180, 140, 90])],
    },
    {
        "name": "Medieval Siege",
        "bg_color": np.array([12, 8, 6], dtype=np.float64),
        "glow_col": np.array([255, 140, 40], dtype=np.float64),
        "material": "fire",
        "bg_fx": "castle_collapse",
        "camera_shake": 0.18,
        "particles_n": 780,
        "particle_colors": [np.array([255, 180, 80]), np.array([200, 100, 40])],
    },
    {
        "name": "Wild West",
        "bg_color": np.array([32, 24, 16], dtype=np.float64),
        "glow_col": np.array([220, 160, 100], dtype=np.float64),
        "material": "dust",
        "bg_fx": "tornado",
        "camera_shake": 0.12,
        "particles_n": 820,
        "particle_colors": [np.array([200, 170, 130]), np.array([170, 130, 90])],
    },
    {
        "name": "1980s Arcade",
        "bg_color": np.array([4, 2, 12], dtype=np.float64),
        "glow_col": np.array([255, 0, 255], dtype=np.float64),
        "material": "pixelated",
        "bg_fx": "arcade_screen",
        "camera_shake": 0.05,
        "particles_n": 650,
        "particle_colors": [np.array([255, 0, 255]), np.array([0, 255, 255])],
    },
    {
        "name": "Cyberpunk 2077",
        "bg_color": np.array([6, 4, 18], dtype=np.float64),
        "glow_col": np.array([255, 50, 150], dtype=np.float64),
        "material": "glitch",
        "bg_fx": "flying_cars",
        "camera_shake": 0.08,
        "particles_n": 720,
        "particle_colors": [np.array([240, 60, 160]), np.array([0, 210, 255])],
    },
    {
        "name": "Nuclear Apocalypse",
        "bg_color": np.array([8, 12, 6], dtype=np.float64),
        "glow_col": np.array([100, 255, 80], dtype=np.float64),
        "material": "xray",
        "bg_fx": "mushroom_cloud",
        "camera_shake": 0.20,
        "particles_n": 790,
        "particle_colors": [np.array([140, 160, 140]), np.array([100, 120, 100])],
    },
    {
        "name": "Warp Speed",
        "bg_color": np.array([2, 2, 8], dtype=np.float64),
        "glow_col": np.array([200, 220, 255], dtype=np.float64),
        "material": "warp_stretch",
        "bg_fx": "wormhole",
        "camera_shake": 0.25,
        "particles_n": 900,
        "particle_colors": [np.array([240, 245, 255]), np.array([180, 200, 255])],
    },
]

# Select random eras
np.random.seed(WORLD_SEED)
SELECTED_INDICES = np.random.choice(N_WORLDS_TOTAL, N_WORLDS_PICK, replace=False)
SELECTED_WORLDS  = [WORLDS[i] for i in SELECTED_INDICES]


# ============================================================
# PARTICLE SYSTEMS (Step 2: parallax depth layers)
# ============================================================
def init_particles(wi):
    w = SELECTED_WORLDS[wi]
    n = w["particles_n"]
    np.random.seed(WORLD_SEED + wi * 997 + 3000)

    # Depth layers: 0.5x (background), 1.0x (mid), 1.5x (foreground)
    depth = np.random.choice([0.5, 1.0, 1.5], n, p=[0.35, 0.40, 0.25])

    # Size scales with depth (foreground=larger)
    base_size = np.random.choice([2, 3, 3, 4, 4, 5], n).astype(np.float64)
    size = np.round(base_size * (0.6 + depth * 0.4)).astype(int)
    size = np.clip(size, 1, 7)

    # Alpha scales with depth (foreground=brighter)
    base_alpha = np.random.uniform(0.5, 1.0, n)
    alpha = base_alpha * (0.4 + depth * 0.4)

    # Velocity scales with depth
    base_vx = np.random.uniform(-60, 60, n)
    base_vy = np.random.uniform(-120, 120, n)

    return {
        "n": n,
        "sy": np.random.uniform(0, RH, n),
        "sx": np.random.uniform(0, RW, n),
        "vx": base_vx * depth,
        "vy": base_vy * depth,
        "size": size,
        "alpha": alpha,
        "col_t": np.random.uniform(0, 1, n),
        "depth": depth,
    }

PSYS = [init_particles(i) for i in range(N_WORLDS_PICK)]


# ============================================================
# DISINTEGRATION PARTICLES (Step 1 fix + Step 5 dynamic teleport)
# ============================================================
def init_disintegration_particles():
    """Sample 800 points from OSD alpha mask with pre-generated explosion vectors."""
    if not HAS_OSD:
        return None

    alpha = osd_rgba_scaled[:, :, 3]
    ys, xs = np.where(alpha > 128)

    if len(ys) == 0:
        return None

    n = min(800, len(ys))
    np.random.seed(WORLD_SEED + 5555)
    indices = np.random.choice(len(ys), n, replace=False)

    # Pre-generate per-particle explosion angles and speeds (Step 1 + Step 5)
    angles = np.random.uniform(0, 2 * np.pi, n)
    speeds = np.random.uniform(100, 350, n)

    return {
        "n": n,
        "sx": xs[indices].astype(float),
        "sy": ys[indices].astype(float),
        "colors": osd_rgba_scaled[ys[indices], xs[indices], :3],
        "angles": angles,
        "speeds": speeds,
    }

DISINT_PARTICLES = init_disintegration_particles()


# ============================================================
# HELPERS
# ============================================================
def get_world_and_tn(t):
    t  = t % DURATION
    wi = int(t // WORLD_DUR) % N_WORLDS_PICK
    tn = (t % WORLD_DUR) / WORLD_DUR
    return wi, tn

def teleport_phase(t):
    """Returns phase: 'explode' (0-0.15s), 'flash' (0.15-0.20s), 'reform' (0.20-0.35s), 'normal'"""
    t_in = (t % DURATION) % WORLD_DUR
    dist = min(t_in, WORLD_DUR - t_in)

    if dist < EXPLODE_DUR:
        return 'explode', dist / EXPLODE_DUR
    elif dist < FLASH_DUR:
        return 'flash', (dist - EXPLODE_DUR) / (FLASH_DUR - EXPLODE_DUR)
    elif dist < (FLASH_DUR + EXPLODE_DUR):
        return 'reform', (dist - FLASH_DUR) / EXPLODE_DUR
    else:
        return 'normal', 0.0


# ============================================================
# MATERIAL SHADERS FOR OSD (Step 1: vectorized xray + warp_stretch)
# ============================================================
def apply_material(rgba, material, tn, phase, prog):
    """Apply material transformation to OSD RGBA array."""
    result = rgba.copy()
    h, w = rgba.shape[0], rgba.shape[1]

    if phase == 'explode' or phase == 'reform':
        fade = 1.0 - prog if phase == 'explode' else prog
        result[:, :, 3] *= fade
        return result

    if material == "lava":
        # Vectorized flowing lava texture
        y_range = np.arange(h)
        waves = (12 * np.sin(2*np.pi*tn*2 + y_range * 0.03)).astype(int)
        for y in range(h):
            result[y] = np.roll(result[y], waves[y], axis=0)
        alpha_mask = result[:, :, 3] > 0
        result[alpha_mask, 0] = np.minimum(255, result[alpha_mask, 0] + 80)
        result[alpha_mask, 1] = np.minimum(255, result[alpha_mask, 1] + 30)

    elif material == "sand":
        edge_fade = np.linspace(1.0, 0.3, w).reshape(1, -1)
        wind = math.sin(2*np.pi*tn)
        if wind > 0:
            edge_fade = np.flip(edge_fade)
        result[:, :, 3] *= edge_fade
        result[:, :, :3] *= 0.9
        result[:, :, :3] += np.array([40, 35, 25]) * 0.1

    elif material == "fire":
        # Vectorized ember flicker at top
        y_top = min(30, h)
        y_range = np.arange(y_top).reshape(-1, 1)
        flicker = 0.6 + 0.4 * np.sin(2*np.pi*tn*5 + y_range)
        result[:y_top, :, 3] *= flicker
        alpha_mask = result[:, :, 3] > 0
        result[alpha_mask, 0] = np.minimum(255, result[alpha_mask, 0] * 1.2)
        result[alpha_mask, 1] *= 0.7
        result[alpha_mask, 2] *= 0.4

    elif material == "dust":
        result[:, :, 3] *= 0.6
        # Vectorized box blur using cumulative sum
        blur_size = 3
        pad = blur_size // 2
        padded = np.pad(result[:, :, 3], pad, mode='edge')
        # Horizontal pass
        cs = np.cumsum(padded, axis=1)
        h_blur = cs[:, blur_size:] - cs[:, :-blur_size]
        # Vertical pass
        cs2 = np.cumsum(h_blur, axis=0)
        blurred = cs2[blur_size:, :] - cs2[:-blur_size, :]
        result[:, :, 3] = blurred / (blur_size * blur_size)

    elif material == "pixelated":
        block = 8
        # Vectorized pixelation via reshape
        bh = (h // block) * block
        bw = (w // block) * block
        cropped = result[:bh, :bw].reshape(h // block, block, w // block, block, 4)
        block_avg = cropped.mean(axis=(1, 3), keepdims=True)
        result[:bh, :bw] = block_avg.repeat(block, axis=1).repeat(block, axis=3).reshape(bh, bw, 4)
        # RGB split
        result[1:, :, 0] = result[:-1, :, 0]
        result[:, 1:, 2] = result[:, :-1, 2]

    elif material == "glitch":
        # Vectorized scanlines
        scanline_rows = np.arange(0, h, 3)
        result[scanline_rows, :, :3] *= 0.7
        np.random.seed(int(tn * 100))
        for i in range(5):
            y = np.random.randint(0, h-5)
            shift = np.random.randint(-15, 15)
            result[y:y+5] = np.roll(result[y:y+5], shift, axis=1)
        result[:, 2:, 0] = result[:, :-2, 0]
        result[:, :-2, 2] = result[:, 2:, 2]

    elif material == "xray":
        # VECTORIZED: green transparent glow (Step 1 fix)
        result[:, :, 3] *= 0.5
        result[:, :, :3] = result[:, :, :3] * 0.3 + np.array([80, 255, 100]) * 0.7
        # Inner glow: add green boost where alpha > threshold (vectorized)
        alpha_mask = result[:, :, 3] > 10
        result[:, :, 1] = np.where(alpha_mask, np.minimum(255, result[:, :, 1] + 50), result[:, :, 1])

    elif material == "warp_stretch":
        # VECTORIZED: radial motion blur toward edges (Step 1 fix)
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        dy = yy - cy
        dx = xx - cx
        dist_map = np.sqrt(dx**2 + dy**2)
        angle_map = np.arctan2(dy.astype(np.float64), dx.astype(np.float64))

        stretch = (dist_map * 0.15 * math.sin(2*np.pi*tn)).astype(np.float64)
        new_dist = dist_map + stretch
        x_src = (cx + new_dist * np.cos(angle_map)).astype(int)
        y_src = (cy + new_dist * np.sin(angle_map)).astype(int)

        # Clamp source coords
        x_src = np.clip(x_src, 0, w - 1)
        y_src = np.clip(y_src, 0, h - 1)

        # Only apply where alpha > threshold and dist > 10
        apply_mask = (result[:, :, 3] > 10) & (dist_map > 10)
        sampled = result[y_src, x_src]
        result = np.where(apply_mask[:, :, np.newaxis], sampled * 0.5 + result * 0.5, result)

    # Apply rim light to OSD (Step 3)
    if OSD_RIM_MASK is not None and phase == 'normal':
        rim_h = min(h, OSD_RIM_MASK.shape[0])
        rim_w = min(w, OSD_RIM_MASK.shape[1])
        rim_strength = OSD_RIM_MASK[:rim_h, :rim_w, np.newaxis]
        pulse = 0.7 + 0.3 * math.sin(2*np.pi*tn*1.5)
        result[:rim_h, :rim_w, :3] = np.minimum(
            255,
            result[:rim_h, :rim_w, :3] + rim_strength * 120 * pulse
        )

    return result


# ============================================================
# CINEMATIC BACKGROUND EFFECTS
# ============================================================
def render_bg_fx(frame, fx_type, tn, wi):
    """Render cinematic background effects."""

    if fx_type == "meteor_strike":
        mx = int(RW * 0.75 - tn * RW * 0.5)
        my = int(tn * RH * 0.6)
        # Vectorized meteor glow circles
        radii = np.arange(15, 60, 5)
        angles = np.linspace(0, 2*np.pi, 24)
        r_grid, a_grid = np.meshgrid(radii, angles)
        pts_x = (mx + r_grid * np.cos(a_grid)).astype(int).ravel()
        pts_y = (my + r_grid * np.sin(a_grid)).astype(int).ravel()
        pts_alpha = (0.08 * (1.0 - r_grid / 60)).ravel()
        valid = (pts_x >= 0) & (pts_x < RW) & (pts_y >= 0) & (pts_y < RH)
        meteor_col = np.array([255, 120, 40])
        for c in range(3):
            np.add.at(frame[:, :, c], (pts_y[valid], pts_x[valid]), meteor_col[c] * pts_alpha[valid])
        if tn > 0.6:
            impact_r = (tn - 0.6) * 800
            ring = np.clip(1.0 - np.abs(DIST - impact_r) / 20, 0, 1)
            frame += ring.reshape(RH, RW, 1) * np.array([200, 100, 50]) * 0.15
        if tn > 0.7:
            np.random.seed(WORLD_SEED + wi * 100)
            for i in range(5):
                x_start = np.random.randint(0, RW)
                y_start = int(RH * 0.75)
                steps = np.arange(80)
                cx = (x_start + np.cumsum(np.random.uniform(-2, 2, 80))).astype(int)
                cy = (y_start + steps * 2).astype(int)
                valid = (cx >= 0) & (cx < RW) & (cy >= 0) & (cy < RH)
                frame[cy[valid], cx[valid]] = [80, 60, 50]

    elif fx_type == "sandstorm":
        wall_x = int(RW * 1.2 - tn * RW * 1.5)
        y_range = np.arange(RH)
        waves = (30 * np.sin(2*np.pi*tn*3 + y_range * 0.02)).astype(int)
        x_positions = wall_x + waves
        sand_col = np.array([180, 140, 90])
        for y in range(RH):
            x_pos = x_positions[y]
            if x_pos < RW:
                start = max(0, x_pos)
                fade = np.clip((RW - x_pos) / 100, 0, 1)
                frame[y, start:] += sand_col * fade * 0.25
        if math.sin(2*np.pi*tn*7) > 0.95:
            frame += np.array([230, 200, 150]) * 0.3

    elif fx_type == "castle_collapse":
        castle_x = int(RW * 0.75)
        castle_y = int(RH * 0.70)
        shake = int(15 * math.sin(2*np.pi*tn*10))
        # Vectorized castle tower rectangle
        dy_range = np.arange(-100 - int(tn*50), 0)
        dx_range = np.arange(-30, 30)
        dy_g, dx_g = np.meshgrid(dy_range, dx_range, indexing='ij')
        bx = (castle_x + dx_g + shake).ravel()
        by = (castle_y + dy_g + abs(shake)).ravel()
        valid = (bx >= 0) & (bx < RW) & (by >= 0) & (by < RH)
        frame[by[valid], bx[valid]] = [25, 20, 18]
        # Vectorized fire circles
        radii = np.arange(10, 45, 4)
        angles = np.linspace(0, 2*np.pi, 20)
        r_grid, a_grid = np.meshgrid(radii, angles)
        fx_pts = (castle_x + r_grid * np.cos(a_grid)).astype(int).ravel()
        fy_pts = (castle_y - 100 + r_grid * np.sin(a_grid) * 0.4).astype(int).ravel()
        f_alpha = (0.15 * (1.0 - r_grid / 45) * (0.7 + 0.3 * math.sin(2*np.pi*tn*5))).ravel()
        valid = (fx_pts >= 0) & (fx_pts < RW) & (fy_pts >= 0) & (fy_pts < RH)
        fire_col = np.array([255, 140, 40])
        for c in range(3):
            np.add.at(frame[:, :, c], (fy_pts[valid], fx_pts[valid]), fire_col[c] * f_alpha[valid])

    elif fx_type == "tornado":
        tornado_x = int(RW * 0.3 + 50 * math.sin(2*np.pi*tn))
        y_range = np.arange(int(RH * 0.3), RH)
        widths = (10 + (y_range - RH * 0.3) * 0.3).astype(int)
        spirals = (widths * np.sin(2*np.pi*tn*5 + y_range * 0.05)).astype(int)
        x_centers = tornado_x + spirals
        tornado_col = np.array([140, 110, 80])
        for idx, y in enumerate(y_range):
            w_val = max(1, widths[idx])
            xc = x_centers[idx]
            dx = np.arange(-w_val, w_val)
            x_arr = xc + dx
            valid = (x_arr >= 0) & (x_arr < RW)
            fade = (1.0 - np.abs(dx) / w_val) * 0.20
            frame[y, x_arr[valid]] += tornado_col * fade[valid].reshape(-1, 1)

    elif fx_type == "arcade_screen":
        pacman_x = int((tn * RW * 2) % RW)
        pacman_y = int(RH * 0.3)
        # Vectorized pac-man circle
        radii = np.arange(2, 18)
        angles = np.linspace(0.3, 2*np.pi - 0.3, 20)
        r_g, a_g = np.meshgrid(radii, angles)
        px_pts = (pacman_x + r_g * np.cos(a_g)).astype(int).ravel()
        py_pts = (pacman_y + r_g * np.sin(a_g)).astype(int).ravel()
        valid = (px_pts >= 0) & (px_pts < RW) & (py_pts >= 0) & (py_pts < RH)
        frame[py_pts[valid], px_pts[valid]] = [255, 255, 0]
        # Vectorized ghosts
        ghost_colors = [[255, 0, 0], [0, 255, 255], [255, 100, 200]]
        radii_g = np.arange(2, 16)
        angles_g = np.linspace(0, np.pi, 12)
        rg_g, ag_g = np.meshgrid(radii_g, angles_g)
        for i in range(3):
            ghost_x = int(pacman_x - 60 - i * 50) % RW
            gx_pts = (ghost_x + rg_g * np.cos(ag_g)).astype(int).ravel()
            gy_pts = (pacman_y + rg_g * np.sin(ag_g)).astype(int).ravel()
            valid = (gx_pts >= 0) & (gx_pts < RW) & (gy_pts >= 0) & (gy_pts < RH)
            frame[gy_pts[valid], gx_pts[valid]] = ghost_colors[i]

    elif fx_type == "flying_cars":
        np.random.seed(WORLD_SEED + wi * 200)
        dy_range = np.arange(-6, 6)
        dx_range = np.arange(-15, 15)
        dy_g, dx_g = np.meshgrid(dy_range, dx_range, indexing='ij')
        dy_flat, dx_flat = dy_g.ravel(), dx_g.ravel()
        for i in range(4):
            car_x = int((tn * RW * 3 + i * 200) % (RW + 100) - 50)
            car_y = int(RH * 0.2 + i * 80)
            car_color = np.array([255, 50, 150] if i % 2 == 0 else [0, 200, 255])
            bx = car_x + dx_flat
            by = car_y + dy_flat
            valid = (bx >= 0) & (bx < RW) & (by >= 0) & (by < RH)
            frame[by[valid], bx[valid]] = car_color
            # Vectorized trail
            trail_x = car_x - np.arange(30) * 3
            trail_valid = (trail_x >= 0) & (trail_x < RW) & (0 <= car_y < RH)
            if np.any(trail_valid):
                fade = (1.0 - np.arange(30) / 30)[trail_valid] * 0.4
                frame[car_y, trail_x[trail_valid]] += car_color * fade.reshape(-1, 1)

    elif fx_type == "mushroom_cloud":
        cloud_x = int(RW * 0.5)
        cloud_y_base = int(RH * 0.7 - tn * 200)
        # Vectorized stem
        stem_y = np.arange(max(0, cloud_y_base), RH)
        stem_x = np.arange(max(0, cloud_x - 20), min(RW, cloud_x + 20))
        if len(stem_y) > 0 and len(stem_x) > 0:
            frame[np.ix_(stem_y, stem_x)] = [60, 50, 40]
        if tn > 0.3:
            top_r = int(100 + tn * 150)
            radii = np.arange(10, top_r, 5)
            angles = np.linspace(0, 2*np.pi, 30)
            r_g, a_g = np.meshgrid(radii, angles)
            cx_pts = (cloud_x + r_g * np.cos(a_g)).astype(int).ravel()
            cy_pts = (cloud_y_base - 50 + r_g * np.sin(a_g) * 0.5).astype(int).ravel()
            c_alpha = (0.08 * (1.0 - r_g / top_r)).ravel()
            valid = (cx_pts >= 0) & (cx_pts < RW) & (cy_pts >= 0) & (cy_pts < RH)
            cloud_col = np.array([100, 80, 60])
            for c in range(3):
                np.add.at(frame[:, :, c], (cy_pts[valid], cx_pts[valid]), cloud_col[c] * c_alpha[valid])
        glow = np.clip(1.0 - DIST / (400 * (0.5 + tn)), 0, 1) ** 3
        frame += glow.reshape(RH, RW, 1) * np.array([80, 200, 60]) * 0.12

    elif fx_type == "wormhole":
        center_x, center_y = RW // 2, RH // 2
        # Vectorized spiral
        i_arr = np.arange(100)
        w_angles = (i_arr / 100) * 4 * np.pi + tn * 4 * np.pi
        w_radii = 50 + i_arr * 3
        wx = (center_x + w_radii * np.cos(w_angles)).astype(int)
        wy = (center_y + w_radii * np.sin(w_angles) * 0.6).astype(int)
        w_fade = (1.0 - i_arr / 100) * 0.5
        valid = (wx >= 0) & (wx < RW) & (wy >= 0) & (wy < RH)
        worm_col = np.array([200, 220, 255])
        for c in range(3):
            np.add.at(frame[:, :, c], (wy[valid], wx[valid]), worm_col[c] * w_fade[valid])
        horizon_r = 80 + 30 * math.sin(2*np.pi*tn*2)
        ring = np.clip(1.0 - np.abs(DIST - horizon_r) / 15, 0, 1)
        frame += ring.reshape(RH, RW, 1) * np.array([220, 230, 255]) * 0.3


# ============================================================
# CAMERA EFFECTS (Step 4: smooth motion with easing)
# ============================================================
def apply_camera_shake(frame, shake_amount, tn, phase, prog):
    """Apply eased screen shake."""
    if shake_amount < 0.01:
        return frame

    # Eased shake envelope: ramp up during explode, ramp down during reform
    if phase == 'explode':
        envelope = ease_in_quad(prog)
    elif phase == 'flash':
        envelope = 1.0
    elif phase == 'reform':
        envelope = 1.0 - ease_out_quad(prog)
    else:
        envelope = 0.0

    if envelope < 0.01:
        return frame

    shake_x = int(shake_amount * 20 * math.sin(2*np.pi*tn*15) * envelope)
    shake_y = int(shake_amount * 15 * math.cos(2*np.pi*tn*17) * envelope)

    shifted = np.roll(frame, shake_x, axis=1)
    shifted = np.roll(shifted, shake_y, axis=0)
    return shifted


def apply_camera_zoom_sway(frame, tn, wi):
    """Apply subtle slow zoom drift and gentle sway per era (Step 4)."""
    # Slow zoom: 1.0x to 1.03x with eased sine
    zoom_t = ease_in_out_cubic((math.sin(2*np.pi*tn*0.5) + 1.0) / 2.0)
    zoom = 1.0 + 0.03 * zoom_t

    # Gentle sway/pan
    sway_x = int(8 * math.sin(2*np.pi*tn*0.7 + wi * 1.5))
    sway_y = int(5 * math.sin(2*np.pi*tn*0.5 + wi * 2.3))

    if zoom <= 1.001 and abs(sway_x) < 1 and abs(sway_y) < 1:
        return frame

    h, w = frame.shape[:2]
    # Compute crop region for zoom
    new_h = int(h / zoom)
    new_w = int(w / zoom)
    y_off = (h - new_h) // 2 + sway_y
    x_off = (w - new_w) // 2 + sway_x

    # Clamp offsets
    y_off = max(0, min(y_off, h - new_h))
    x_off = max(0, min(x_off, w - new_w))

    cropped = frame[y_off:y_off+new_h, x_off:x_off+new_w]

    # Resize back using nearest neighbor (fast, no PIL needed for small zoom)
    # Use simple repeat-based upscale for speed
    y_idx = np.linspace(0, new_h - 1, h).astype(int)
    x_idx = np.linspace(0, new_w - 1, w).astype(int)
    return cropped[np.ix_(y_idx, x_idx)]


# ============================================================
# MAIN RENDER
# ============================================================
def render_frame_background(t):
    """Render background only (no OSD)."""
    t = t % DURATION
    wi, tn = get_world_and_tn(t)
    w = SELECTED_WORLDS[wi]

    frame = np.empty((RH, RW, 3), dtype=np.float64)
    frame[:] = w["bg_color"]

    # Background gradient
    y_blend = np.linspace(0, 0.18, RH).reshape(-1, 1, 1)
    frame += y_blend * w["glow_col"] * 0.55

    # Cinematic BG effects
    render_bg_fx(frame, w["bg_fx"], tn, wi)

    # Glow around OSD position
    pulse = 0.70 + 0.30 * math.sin(2*np.pi*tn*2)
    g1 = np.clip(1.0 - DIST / (280 * pulse), 0, 1) ** 2.0
    frame += g1.reshape(RH, RW, 1) * w["glow_col"] * 0.25

    # God rays from glow center (Step 3)
    ray_pulse = 0.5 + 0.5 * math.sin(2*np.pi*tn*1.2)
    for ray_mask in GOD_RAY_MASKS:
        frame += ray_mask * w["glow_col"] * 0.06 * ray_pulse

    # Environment particles with parallax depth + motion blur trails (vectorized)
    ps = PSYS[wi]
    n = ps["n"]

    # Vectorized position computation
    py_all = (ps["sy"] + ps["vy"] * tn * WORLD_DUR) % RH
    px_all = (ps["sx"] + ps["vx"] * tn * WORLD_DUR) % RW
    ix_all = px_all.astype(int)
    iy_all = py_all.astype(int)

    # Vectorized alpha and color
    i_arr = np.arange(n)
    alpha_all = ps["alpha"] * (0.6 + 0.4 * np.sin(2*np.pi*tn*3 + i_arr))
    ct = ps["col_t"]
    col_all = (w["particle_colors"][0].reshape(1, 3) * (1 - ct.reshape(-1, 1))
             + w["particle_colors"][1].reshape(1, 3) * ct.reshape(-1, 1)) * alpha_all.reshape(-1, 1)

    sizes = ps["size"]

    # Render size-1 particles via scatter
    mask_s1 = sizes == 1
    if np.any(mask_s1):
        s1_x = np.clip(ix_all[mask_s1], 0, RW - 1)
        s1_y = np.clip(iy_all[mask_s1], 0, RH - 1)
        s1_col = col_all[mask_s1]
        for c in range(3):
            np.add.at(frame[:, :, c], (s1_y, s1_x), s1_col[:, c])

    # Render size>=2 particles with soft circles
    mask_s2 = sizes >= 2
    idx_s2 = np.where(mask_s2)[0]
    for i in idx_s2:
        s = int(sizes[i])
        ix, iy = ix_all[i], iy_all[i]
        y0, y1 = max(0, iy - s), min(RH, iy + s + 1)
        x0, x1 = max(0, ix - s), min(RW, ix + s + 1)
        if y0 < y1 and x0 < x1:
            yl = np.arange(y0, y1).reshape(-1, 1)
            xl = np.arange(x0, x1).reshape(1, -1)
            d = np.sqrt((xl - ix)**2 + (yl - iy)**2)
            m = np.clip(1.0 - d / (s + 0.5), 0, 1)
            frame[y0:y1, x0:x1] += m[:, :, np.newaxis] * col_all[i]

    # Vectorized motion blur trails
    speeds = np.sqrt(ps["vx"]**2 + ps["vy"]**2)
    fast_mask = speeds > 30
    if np.any(fast_mask):
        f_ix = ix_all[fast_mask]
        f_iy = iy_all[fast_mask]
        f_speeds = speeds[fast_mask]
        f_dx = ps["vx"][fast_mask] / f_speeds
        f_dy = ps["vy"][fast_mask] / f_speeds
        f_col = col_all[fast_mask]
        trail_lens = np.minimum(5, (f_speeds / 40).astype(int))
        for step in range(1, 6):
            step_mask = trail_lens >= step
            if not np.any(step_mask):
                break
            tx = (f_ix[step_mask] - (f_dx[step_mask] * step * 2).astype(int))
            ty = (f_iy[step_mask] - (f_dy[step_mask] * step * 2).astype(int))
            valid = (tx >= 0) & (tx < RW) & (ty >= 0) & (ty < RH)
            if np.any(valid):
                trail_alpha = (1.0 - step / (trail_lens[step_mask][valid] + 1)) * 0.5
                t_col = f_col[step_mask][valid] * trail_alpha.reshape(-1, 1)
                for c in range(3):
                    np.add.at(frame[:, :, c], (ty[valid], tx[valid]), t_col[:, c])

    # Ambient occlusion - darken bottom (Step 3)
    frame *= (1.0 - AO_MASK)

    # Vignette - darken corners (Step 3)
    frame *= (1.0 - VIGNETTE_MASK * 0.45)

    # Camera effects
    phase, prog = teleport_phase(t)

    # Smooth camera zoom/sway (Step 4) - applied before shake
    frame_clipped = np.clip(frame, 0, 255).astype(np.uint8)
    frame_clipped = apply_camera_zoom_sway(frame_clipped, tn, wi).astype(np.float64)

    if phase in ('explode', 'flash', 'reform'):
        shake = w["camera_shake"] * (prog if phase != 'reform' else (1.0 - prog))
        frame_clipped = apply_camera_shake(frame_clipped, shake, tn, phase, prog)

    # Flash overlay
    if phase == 'flash':
        frame_clipped = frame_clipped.astype(np.float64) + np.array([220, 230, 255]) * prog * 0.95
        frame_clipped = np.clip(frame_clipped, 0, 255)

    frame_u8 = np.clip(frame_clipped, 0, 255).astype(np.uint8)

    # Color grading (Step 6)
    frame_u8 = apply_color_grading(frame_u8)

    # Chromatic aberration on teleports (Step 6)
    if phase in ('explode', 'flash', 'reform'):
        ca_intensity = prog if phase == 'explode' else (1.0 - prog if phase == 'reform' else 1.0)
        frame_u8 = apply_chromatic_aberration(frame_u8, ca_intensity * 0.8)

    return frame_u8


def render_osd_layer(t):
    """Render OSD with material effects (returns RGBA)."""
    if not HAS_OSD:
        return None

    t = t % DURATION
    wi, tn = get_world_and_tn(t)
    w = SELECTED_WORLDS[wi]
    phase, prog = teleport_phase(t)

    # Dynamic teleport (Step 5): multi-phase explosion
    if phase in ('explode', 'reform') and DISINT_PARTICLES is not None:
        canvas = np.zeros((RH, RW, 4), dtype=np.float64)
        dp = DISINT_PARTICLES
        n = dp["n"]
        angles = dp["angles"]
        speeds = dp["speeds"]

        if phase == 'explode':
            # Multi-phase: energy gather (prog < 0.2) -> burst (prog 0.2-1.0)
            if prog < 0.2:
                # Particles pull inward briefly
                gather_t = prog / 0.2
                gather_dist = (1.0 - ease_out_quad(gather_t)) * 15
                px = OSD_X + dp["sx"] - gather_dist * np.cos(angles)
                py = OSD_Y + dp["sy"] - gather_dist * np.sin(angles)
                particle_alpha = 255.0
            else:
                # Fast burst outward
                burst_t = (prog - 0.2) / 0.8
                eased_t = ease_out_quad(burst_t)
                dist = eased_t * speeds * 0.8
                px = OSD_X + dp["sx"] + dist * np.cos(angles)
                py = OSD_Y + dp["sy"] + dist * np.sin(angles)
                particle_alpha = 255.0 * (1.0 - eased_t * 0.7)
        else:
            # Reform: slow drift inward -> snap back
            reform_t = ease_in_out_cubic(prog)
            dist = (1.0 - reform_t) * speeds * 0.5
            px = OSD_X + dp["sx"] + dist * np.cos(angles + np.pi)
            py = OSD_Y + dp["sy"] + dist * np.sin(angles + np.pi)
            particle_alpha = 255.0 * reform_t

        # Scale alpha by distance from center for glow effect
        cx_osd = OSD_X + OSD_SCALED_W / 2
        cy_osd = OSD_Y + OSD_SCALED_H / 2
        dist_from_center = np.sqrt((px - cx_osd)**2 + (py - cy_osd)**2)
        glow_alpha = np.clip(1.0 - dist_from_center / 400, 0.3, 1.0)

        ix = px.astype(int)
        iy = py.astype(int)
        valid = (ix >= 0) & (ix < RW) & (iy >= 0) & (iy < RH)

        # Vectorized particle scatter
        v_ix, v_iy = ix[valid], iy[valid]
        v_colors = dp["colors"][valid]
        if isinstance(particle_alpha, np.ndarray):
            v_alpha = particle_alpha[valid] * glow_alpha[valid]
        else:
            v_alpha = particle_alpha * glow_alpha[valid]
        canvas[v_iy, v_ix, :3] = v_colors
        canvas[v_iy, v_ix, 3] = v_alpha

        # Vectorized motion blur trails
        if phase == 'explode' and prog > 0.2:
            trail_dx = -np.cos(angles) * 3
            trail_dy = -np.sin(angles) * 3
            for step in range(1, 4):
                tx = (ix + trail_dx * step).astype(int)
                ty = (iy + trail_dy * step).astype(int)
                t_valid = (tx >= 0) & (tx < RW) & (ty >= 0) & (ty < RH)
                if np.any(t_valid):
                    if isinstance(particle_alpha, np.ndarray):
                        t_alpha = particle_alpha[t_valid] * glow_alpha[t_valid] * (1.0 - step / 4.0) * 0.5
                    else:
                        t_alpha = particle_alpha * glow_alpha[t_valid] * (1.0 - step / 4.0) * 0.5
                    canvas[ty[t_valid], tx[t_valid], :3] = np.maximum(
                        canvas[ty[t_valid], tx[t_valid], :3], dp["colors"][t_valid] * 0.7)
                    canvas[ty[t_valid], tx[t_valid], 3] = np.maximum(
                        canvas[ty[t_valid], tx[t_valid], 3], t_alpha)

        # Expanding shockwave ring during teleport (Step 5)
        if phase == 'explode' and prog > 0.3:
            ring_r = (prog - 0.3) * 500
            ring_mask = np.clip(1.0 - np.abs(DIST - ring_r) / 8, 0, 1)
            ring_intensity = (1.0 - (prog - 0.3) / 0.7) * 200
            canvas[:, :, 0] += ring_mask * w["glow_col"][0] / 255 * ring_intensity
            canvas[:, :, 1] += ring_mask * w["glow_col"][1] / 255 * ring_intensity
            canvas[:, :, 2] += ring_mask * w["glow_col"][2] / 255 * ring_intensity
            canvas[:, :, 3] = np.maximum(canvas[:, :, 3], ring_mask * ring_intensity)

        return np.clip(canvas, 0, 255).astype(np.uint8)

    # Normal rendering with material
    material_rgba = apply_material(osd_rgba_scaled, w["material"], tn, phase, prog)

    # Create full canvas
    canvas = np.zeros((RH, RW, 4), dtype=np.float64)
    canvas[OSD_Y:OSD_Y+OSD_SCALED_H, OSD_X:OSD_X+OSD_SCALED_W] = material_rgba

    return canvas.astype(np.uint8)


# ============================================================
# BUILD PIPELINE
# ============================================================
def build():
    try:
        from moviepy import VideoClip, CompositeVideoClip, ImageClip
    except ImportError:
        from moviepy.editor import VideoClip, CompositeVideoClip, ImageClip

    t0 = time.time()
    n_frames = int(DURATION * FPS)
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  ORANGE SWEATER DUDE v5.0 -- CINEMATIC UPGRADE")
    print(f"{sep}")
    print(f"  Character     : {'loaded ' + OSD_FILE if HAS_OSD else 'missing'}")
    print(f"  Eras Selected : {N_WORLDS_PICK}/{N_WORLDS_TOTAL}")
    for i, w in enumerate(SELECTED_WORLDS):
        print(f"                  [{i}] {w['name']:20s} -> {w['material']:15s} | {w['bg_fx']}")
    print(f"  Frames        : {n_frames}")
    print(f"  Resolution    : {RW}x{RH} -> upscale -> {FW}x{FH}")
    print(f"  Effects       : Material shaders, cinematic BG, camera shake")
    print(f"                  Parallax particles, rim light, god rays, vignette")
    print(f"                  Color grading, chromatic aberration, motion blur")
    print(f"  Teleport      : Multi-phase particle explosion + shockwave")
    print(f"{sep}\n")

    # Render background frames
    print("  Rendering background frames...")
    bg_frames = np.zeros((n_frames, RH, RW, 3), dtype=np.uint8)
    for i in range(n_frames):
        bg_frames[i] = render_frame_background(i / FPS)
        if (i+1) % FPS == 0 or (i+1) == n_frames:
            pct = (i+1) / n_frames * 100
            eta = (time.time()-t0) / (i+1) * (n_frames - i - 1)
            print(f"      {pct:5.1f}%  |  {time.time()-t0:6.1f}s  |  ETA: {eta:5.0f}s")
    print(f"  Background rendered ({time.time()-t0:.1f}s)\n")

    # Write background video
    print("  Writing background video...")
    def get_bg_frame(t):
        return bg_frames[int(t * FPS) % n_frames]

    bg_clip = VideoClip(get_bg_frame, duration=DURATION)
    bg_clip = bg_clip.with_fps(FPS)
    bg_clip.write_videofile(TEMP_BG, fps=FPS, codec='libx264', audio=False, logger=None)
    bg_clip.close()
    del bg_frames
    print(f"  Background video written ({time.time()-t0:.1f}s)\n")

    # Upscale
    print(f"  Upscaling to {FW}x{FH}...")
    subprocess.run([
        'ffmpeg', '-y', '-i', TEMP_BG,
        '-vf', f'scale={FW}:{FH}:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-r', str(FPS), TEMP_UP
    ], check=True, capture_output=True)
    print(f"  Upscaled ({time.time()-t0:.1f}s)\n")

    # OSD overlay with material effects
    if HAS_OSD:
        print("  Rendering OSD material transformations...")
        from moviepy import VideoFileClip

        # Render OSD frames
        osd_frames = []
        for i in range(n_frames):
            osd_layer = render_osd_layer(i / FPS)
            if osd_layer is not None:
                osd_frames.append(osd_layer)
            if (i+1) % (FPS*2) == 0:
                pct = (i+1) / n_frames * 100
                print(f"      OSD: {pct:5.1f}%")

        print(f"  OSD frames rendered ({time.time()-t0:.1f}s)\n")

        # Composite
        print("  Compositing final video...")
        bg_video = VideoFileClip(TEMP_UP)

        def make_osd_frame(t):
            idx = int(t * FPS) % len(osd_frames)
            return osd_frames[idx]

        osd_clip = VideoClip(make_osd_frame, duration=DURATION, is_mask=False)
        osd_clip = osd_clip.with_fps(FPS)

        # Scale OSD to final resolution
        scale_factor = FH / RH

        def scaled_osd_frame(t):
            frame = make_osd_frame(t)
            from PIL import Image
            img = Image.fromarray(frame)
            new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
            img_scaled = img.resize(new_size, Image.Resampling.LANCZOS)
            return np.array(img_scaled)

        osd_clip_scaled = VideoClip(scaled_osd_frame, duration=DURATION)
        osd_clip_scaled = osd_clip_scaled.with_fps(FPS)

        final = CompositeVideoClip([bg_video, osd_clip_scaled], size=(FW, FH))
        final.write_videofile(OUTPUT, fps=FPS, codec='libx264', audio=False, logger=None)
        final.close()
        bg_video.close()

        print(f"  Final composite complete ({time.time()-t0:.1f}s)\n")
    else:
        if os.path.exists(OUTPUT):
            os.remove(OUTPUT)
        os.rename(TEMP_UP, OUTPUT)
        print("  Background-only saved\n")

    # Cleanup
    for f in [TEMP_BG, TEMP_UP]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    total = time.time() - t0
    print(f"{sep}")
    print(f"  {OUTPUT}")
    print(f"     {FW}x{FH}  |  {DURATION}s seamless loop  |  {total/60:.1f} min total")
    print(f"  CINEMATIC UPGRADE COMPLETE")
    print(f"{sep}\n")


if __name__ == "__main__":
    build()
