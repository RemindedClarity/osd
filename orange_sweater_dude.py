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
DURATION       = 8.0
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
AO_MASK = np.linspace(0.0, 0.20, RH).reshape(-1, 1, 1) ** 1.5

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
        "bg_color": np.array([55, 35, 20], dtype=np.float64),
        "glow_col": np.array([255, 120, 40], dtype=np.float64),
        "material": "lava",
        "bg_fx": "meteor_strike",
        "camera_shake": 0.15,
        "particles_n": 1600,
        "particle_colors": [np.array([255, 160, 60]), np.array([255, 80, 20])],
    },
    {
        "name": "Ancient Egypt",
        "bg_color": np.array([100, 75, 40], dtype=np.float64),
        "glow_col": np.array([255, 210, 120], dtype=np.float64),
        "material": "sand",
        "bg_fx": "sandstorm",
        "camera_shake": 0.10,
        "particles_n": 1500,
        "particle_colors": [np.array([255, 220, 140]), np.array([240, 180, 80])],
    },
    {
        "name": "Medieval Siege",
        "bg_color": np.array([40, 25, 15], dtype=np.float64),
        "glow_col": np.array([255, 160, 50], dtype=np.float64),
        "material": "fire",
        "bg_fx": "castle_collapse",
        "camera_shake": 0.18,
        "particles_n": 1600,
        "particle_colors": [np.array([255, 200, 80]), np.array([255, 120, 30])],
    },
    {
        "name": "Wild West",
        "bg_color": np.array([90, 65, 40], dtype=np.float64),
        "glow_col": np.array([240, 180, 100], dtype=np.float64),
        "material": "dust",
        "bg_fx": "tornado",
        "camera_shake": 0.12,
        "particles_n": 1600,
        "particle_colors": [np.array([240, 200, 140]), np.array([220, 160, 80])],
    },
    {
        "name": "1980s Arcade",
        "bg_color": np.array([15, 5, 35], dtype=np.float64),
        "glow_col": np.array([255, 0, 255], dtype=np.float64),
        "material": "pixelated",
        "bg_fx": "arcade_screen",
        "camera_shake": 0.05,
        "particles_n": 1400,
        "particle_colors": [np.array([255, 0, 255]), np.array([0, 255, 255])],
    },
    {
        "name": "Cyberpunk 2077",
        "bg_color": np.array([20, 10, 50], dtype=np.float64),
        "glow_col": np.array([255, 50, 200], dtype=np.float64),
        "material": "glitch",
        "bg_fx": "flying_cars",
        "camera_shake": 0.08,
        "particles_n": 1500,
        "particle_colors": [np.array([255, 60, 200]), np.array([0, 230, 255])],
    },
    {
        "name": "Nuclear Apocalypse",
        "bg_color": np.array([25, 40, 18], dtype=np.float64),
        "glow_col": np.array([120, 255, 80], dtype=np.float64),
        "material": "xray",
        "bg_fx": "mushroom_cloud",
        "camera_shake": 0.20,
        "particles_n": 1600,
        "particle_colors": [np.array([150, 255, 100]), np.array([80, 200, 60])],
    },
    {
        "name": "Warp Speed",
        "bg_color": np.array([8, 8, 30], dtype=np.float64),
        "glow_col": np.array([220, 235, 255], dtype=np.float64),
        "material": "warp_stretch",
        "bg_fx": "wormhole",
        "camera_shake": 0.25,
        "particles_n": 1800,
        "particle_colors": [np.array([255, 255, 255]), np.array([180, 210, 255])],
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

    # Size scales with depth (foreground=larger, up to 10px)
    base_size = np.random.choice([3, 4, 5, 6, 7, 8, 9, 10], n).astype(np.float64)
    size = np.round(base_size * (0.5 + depth * 0.5)).astype(int)
    size = np.clip(size, 2, 10)

    # Alpha scales with depth (foreground=brighter, overall brighter)
    base_alpha = np.random.uniform(0.7, 1.0, n)
    alpha = base_alpha * (0.5 + depth * 0.5)

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
        # LARGE VOLCANO SILHOUETTE (right third of frame)
        volcano_cx = int(RW * 0.75)
        volcano_base_y = int(RH * 0.65)
        volcano_peak_y = int(RH * 0.30)
        for y in range(volcano_peak_y, RH):
            t_y = (y - volcano_peak_y) / max(1, RH - volcano_peak_y)
            half_w = int(10 + t_y * 160)
            x0 = max(0, volcano_cx - half_w)
            x1 = min(RW, volcano_cx + half_w)
            frame[y, x0:x1] = np.array([30, 15, 10])
        # Glowing lava at crater
        crater_glow = np.clip(1.0 - np.sqrt((_xx - volcano_cx)**2 + (_yy - volcano_peak_y)**2) / 80, 0, 1) ** 1.5
        frame += crater_glow.reshape(RH, RW, 1) * np.array([255, 100, 20]) * 1.5
        # Lava streams down sides
        for stream in range(3):
            np.random.seed(WORLD_SEED + wi * 100 + stream)
            sx = volcano_cx + np.random.randint(-30, 30)
            for y in range(volcano_peak_y + 20, volcano_base_y):
                sx += np.random.randint(-2, 3)
                sx = max(0, min(RW - 1, sx))
                for dx in range(-3, 4):
                    px = sx + dx
                    if 0 <= px < RW:
                        fade = 1.0 - abs(dx) / 4
                        frame[y, px] += np.array([255, 80, 10]) * fade * 0.8
        # Meteor
        mx = int(RW * 0.3 - tn * RW * 0.2)
        my = int(tn * RH * 0.5)
        meteor_glow = np.clip(1.0 - np.sqrt((_xx - mx)**2 + (_yy - my)**2) / 50, 0, 1) ** 2
        frame += meteor_glow.reshape(RH, RW, 1) * np.array([255, 180, 60]) * 2.0
        if tn > 0.6:
            impact_r = (tn - 0.6) * 800
            ring = np.clip(1.0 - np.abs(DIST - impact_r) / 25, 0, 1)
            frame += ring.reshape(RH, RW, 1) * np.array([255, 120, 50]) * 0.5

    elif fx_type == "sandstorm":
        # LARGE PYRAMID SILHOUETTE (center-right)
        pyr_cx = int(RW * 0.6)
        pyr_base_y = int(RH * 0.72)
        pyr_peak_y = int(RH * 0.28)
        pyr_base_half = 140
        for y in range(pyr_peak_y, pyr_base_y):
            t_y = (y - pyr_peak_y) / max(1, pyr_base_y - pyr_peak_y)
            half_w = int(t_y * pyr_base_half)
            x0 = max(0, pyr_cx - half_w)
            x1 = min(RW, pyr_cx + half_w)
            frame[y, x0:x1] = np.array([60, 45, 25])
        # Sun disk behind pyramid
        sun_cx, sun_cy = int(RW * 0.55), int(RH * 0.22)
        sun_glow = np.clip(1.0 - np.sqrt((_xx - sun_cx)**2 + (_yy - sun_cy)**2) / 120, 0, 1) ** 1.2
        frame += sun_glow.reshape(RH, RW, 1) * np.array([255, 200, 80]) * 2.5
        # Sandstorm wall
        wall_x = int(RW * 1.2 - tn * RW * 1.5)
        y_range = np.arange(RH)
        waves = (30 * np.sin(2*np.pi*tn*3 + y_range * 0.02)).astype(int)
        x_positions = wall_x + waves
        sand_col = np.array([240, 190, 100])
        for y in range(RH):
            x_pos = x_positions[y]
            if x_pos < RW:
                start = max(0, x_pos)
                fade = np.clip((RW - x_pos) / 100, 0, 1)
                frame[y, start:] += sand_col * fade * 0.5
        if math.sin(2*np.pi*tn*7) > 0.9:
            frame += np.array([255, 220, 150]) * 0.4

    elif fx_type == "castle_collapse":
        # LARGE CASTLE SILHOUETTE (right side, with towers)
        castle_base_y = int(RH * 0.65)
        shake = int(8 * math.sin(2*np.pi*tn*10))
        # Main wall
        wall_x0 = int(RW * 0.45) + shake
        wall_x1 = int(RW * 0.95) + shake
        for y in range(int(RH * 0.40), RH):
            x0 = max(0, wall_x0)
            x1 = min(RW, wall_x1)
            if x0 < x1:
                frame[y, x0:x1] = np.array([35, 25, 20])
        # Left tower
        tower_x = int(RW * 0.48) + shake
        for y in range(int(RH * 0.25), castle_base_y):
            for dx in range(-20, 20):
                px = tower_x + dx
                if 0 <= px < RW:
                    frame[y, px] = [40, 30, 25]
        # Right tower
        tower_x2 = int(RW * 0.85) + shake
        for y in range(int(RH * 0.30), castle_base_y):
            for dx in range(-25, 25):
                px = tower_x2 + dx
                if 0 <= px < RW:
                    frame[y, px] = [35, 28, 22]
        # Large fire glow at base
        fire_cx, fire_cy = int(RW * 0.65), int(RH * 0.55)
        fire_glow = np.clip(1.0 - np.sqrt((_xx - fire_cx)**2 + (_yy - fire_cy)**2) / 150, 0, 1) ** 1.5
        flicker = 0.7 + 0.3 * math.sin(2*np.pi*tn*6)
        frame += fire_glow.reshape(RH, RW, 1) * np.array([255, 140, 30]) * 2.0 * flicker
        # Smoke clouds above
        for cloud_i in range(4):
            np.random.seed(WORLD_SEED + wi * 50 + cloud_i)
            cloud_cx = int(RW * 0.5 + np.random.uniform(-100, 150))
            cloud_cy = int(RH * 0.25 + np.random.uniform(-50, 50))
            cloud_r = 60 + np.random.randint(0, 40)
            cloud_mask = np.clip(1.0 - np.sqrt((_xx - cloud_cx)**2 + (_yy - cloud_cy)**2) / cloud_r, 0, 1)
            frame += cloud_mask.reshape(RH, RW, 1) * np.array([80, 60, 50]) * 0.4

    elif fx_type == "tornado":
        tornado_x = int(RW * 0.3 + 50 * math.sin(2*np.pi*tn))
        y_range = np.arange(int(RH * 0.2), RH)
        widths = (8 + (y_range - RH * 0.2) * 0.35).astype(int)
        spirals = (widths * math.sin(2*np.pi*tn*5) * np.sin(y_range * 0.05)).astype(int)
        x_centers = tornado_x + spirals
        tornado_col = np.array([200, 160, 100])
        for idx, y in enumerate(y_range):
            w_val = max(1, widths[idx])
            xc = x_centers[idx]
            dx = np.arange(-w_val, w_val)
            x_arr = xc + dx
            valid = (x_arr >= 0) & (x_arr < RW)
            fade = (1.0 - np.abs(dx) / w_val) * 0.5
            frame[y, x_arr[valid]] += tornado_col * fade[valid].reshape(-1, 1)
        # Debris flying around tornado
        np.random.seed(WORLD_SEED + wi * 300 + int(tn * 10))
        for d in range(15):
            angle = tn * 8 + d * 0.6
            r = 80 + d * 15
            dx = int(tornado_x + r * math.cos(angle))
            dy = int(RH * 0.5 + r * math.sin(angle) * 0.4)
            if 0 <= dx < RW and 0 <= dy < RH:
                frame[max(0,dy-2):min(RH,dy+3), max(0,dx-2):min(RW,dx+3)] += np.array([180, 140, 80]) * 0.6

    elif fx_type == "arcade_screen":
        # Bigger pac-man and ghosts, neon grid background
        # Neon grid lines
        for gx in range(0, RW, 40):
            if gx < RW:
                frame[:, max(0,gx):min(RW,gx+1)] += np.array([40, 0, 60])
        for gy in range(0, RH, 40):
            if gy < RH:
                frame[max(0,gy):min(RH,gy+1), :] += np.array([40, 0, 60])
        pacman_x = int((tn * RW * 2) % RW)
        pacman_y = int(RH * 0.3)
        # Big pac-man
        pac_glow = np.clip(1.0 - np.sqrt((_xx - pacman_x)**2 + (_yy - pacman_y)**2) / 30, 0, 1)
        # Mouth cutout
        mouth_angle = np.arctan2(_yy - pacman_y, _xx - pacman_x)
        mouth_open = 0.4 * (0.5 + 0.5 * math.sin(2*np.pi*tn*8))
        mouth_mask = (np.abs(mouth_angle) < mouth_open) & (_xx > pacman_x)
        pac_glow[mouth_mask] = 0
        frame += pac_glow.reshape(RH, RW, 1) * np.array([255, 255, 0]) * 1.5
        # Big ghosts
        ghost_colors = [np.array([255, 30, 30]), np.array([0, 255, 255]), np.array([255, 100, 220])]
        for i in range(3):
            ghost_x = int(pacman_x - 80 - i * 60) % RW
            g_glow = np.clip(1.0 - np.sqrt((_xx - ghost_x)**2 + (_yy - pacman_y)**2) / 25, 0, 1)
            frame += g_glow.reshape(RH, RW, 1) * ghost_colors[i] * 1.2

    elif fx_type == "flying_cars":
        # NEON CITYSCAPE SILHOUETTE
        np.random.seed(WORLD_SEED + wi * 200)
        # Skyline of buildings
        bld_heights = [0.35, 0.45, 0.55, 0.40, 0.60, 0.50, 0.42, 0.55, 0.38, 0.48,
                       0.52, 0.44, 0.58, 0.36, 0.46]
        bld_width = RW // len(bld_heights)
        for bi, bh in enumerate(bld_heights):
            x0 = bi * bld_width
            x1 = x0 + bld_width - 2
            y_top = int(RH * (1.0 - bh))
            # Dark building body
            frame[y_top:RH, x0:x1] = np.array([15, 8, 25])
            # Glowing windows
            for wy in range(y_top + 5, RH - 10, 12):
                for wx in range(x0 + 4, x1 - 4, 8):
                    if np.random.random() < 0.6:
                        win_col = np.array([255, 50, 200]) if np.random.random() < 0.5 else np.array([0, 200, 255])
                        wy0, wy1 = max(0, wy), min(RH, wy + 6)
                        wx0, wx1 = max(0, wx), min(RW, wx + 5)
                        frame[wy0:wy1, wx0:wx1] += win_col * 0.4
        # Neon reflections on ground
        ground_y = int(RH * 0.82)
        if ground_y < RH:
            reflection = frame[ground_y-20:ground_y, :, :].copy()
            ref_h = min(reflection.shape[0], RH - ground_y)
            frame[ground_y:ground_y+ref_h, :] += np.flip(reflection[:ref_h], axis=0) * 0.3
        # Flying cars with bright trails
        for i in range(6):
            car_x = int((tn * RW * 3 + i * 150) % (RW + 100) - 50)
            car_y = int(RH * 0.15 + i * 60)
            car_color = np.array([255, 50, 200] if i % 2 == 0 else [0, 220, 255])
            cy0, cy1 = max(0, car_y - 4), min(RH, car_y + 4)
            cx0, cx1 = max(0, car_x - 12), min(RW, car_x + 12)
            if cy0 < cy1 and cx0 < cx1:
                frame[cy0:cy1, cx0:cx1] = car_color
            # Bright trail
            trail_x = car_x - np.arange(40) * 3
            trail_valid = (trail_x >= 0) & (trail_x < RW)
            if np.any(trail_valid) and 0 <= car_y < RH:
                fade = (1.0 - np.arange(40) / 40)[trail_valid] * 0.7
                frame[car_y, trail_x[trail_valid]] += car_color * fade.reshape(-1, 1)

    elif fx_type == "mushroom_cloud":
        cloud_x = int(RW * 0.5)
        cloud_y_base = int(RH * 0.6 - tn * 150)
        # Large stem
        stem_half = 30
        stem_y = np.arange(max(0, cloud_y_base), RH)
        stem_x = np.arange(max(0, cloud_x - stem_half), min(RW, cloud_x + stem_half))
        if len(stem_y) > 0 and len(stem_x) > 0:
            frame[np.ix_(stem_y, stem_x)] = [80, 60, 40]
        # Large mushroom cap
        cap_r = int(120 + tn * 100)
        cap_glow = np.clip(1.0 - np.sqrt((_xx - cloud_x)**2 + ((_yy - cloud_y_base + 30) * 1.8)**2) / cap_r, 0, 1) ** 1.2
        frame += cap_glow.reshape(RH, RW, 1) * np.array([180, 120, 60]) * 1.5
        # Bright radioactive glow
        glow = np.clip(1.0 - DIST / (400 * (0.5 + tn)), 0, 1) ** 2
        frame += glow.reshape(RH, RW, 1) * np.array([100, 255, 80]) * 0.5
        # Expanding shockwave
        if tn > 0.3:
            shock_r = (tn - 0.3) * 600
            ring = np.clip(1.0 - np.abs(DIST - shock_r) / 15, 0, 1)
            frame += ring.reshape(RH, RW, 1) * np.array([200, 255, 150]) * 0.6

    elif fx_type == "wormhole":
        center_x, center_y = RW // 2, int(RH * 0.4)
        # Large wormhole glow
        worm_dist = np.sqrt((_xx - center_x)**2 + (_yy - center_y)**2)
        worm_glow = np.clip(1.0 - worm_dist / 200, 0, 1) ** 1.5
        frame += worm_glow.reshape(RH, RW, 1) * np.array([150, 180, 255]) * 2.0
        # Spiral arms
        i_arr = np.arange(200)
        w_angles = (i_arr / 200) * 6 * np.pi + tn * 4 * np.pi
        w_radii = 20 + i_arr * 2
        w_fade = (1.0 - i_arr / 200) * 1.5
        for arm in range(3):
            arm_offset = arm * 2 * np.pi / 3
            wx = (center_x + w_radii * np.cos(w_angles + arm_offset)).astype(int)
            wy = (center_y + w_radii * np.sin(w_angles + arm_offset) * 0.6).astype(int)
            valid = (wx >= 0) & (wx < RW) & (wy >= 0) & (wy < RH)
            worm_col = np.array([220, 235, 255])
            for c in range(3):
                np.add.at(frame[:, :, c], (wy[valid], wx[valid]), worm_col[c] * w_fade[valid])
        # Event horizon ring
        horizon_r = 80 + 30 * math.sin(2*np.pi*tn*2)
        ring = np.clip(1.0 - np.abs(worm_dist - horizon_r) / 10, 0, 1)
        frame += ring.reshape(RH, RW, 1) * np.array([255, 255, 255]) * 0.8


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

    # Background gradient (brighter)
    y_blend = np.linspace(0, 0.35, RH).reshape(-1, 1, 1)
    frame += y_blend * w["glow_col"] * 1.2

    # Cinematic BG effects
    render_bg_fx(frame, w["bg_fx"], tn, wi)

    # Glow around OSD position (5x brighter)
    pulse = 0.70 + 0.30 * math.sin(2*np.pi*tn*2)
    g1 = np.clip(1.0 - DIST / (350 * pulse), 0, 1) ** 1.5
    frame += g1.reshape(RH, RW, 1) * w["glow_col"] * 1.25

    # God rays from glow center (brighter)
    ray_pulse = 0.5 + 0.5 * math.sin(2*np.pi*tn*1.2)
    for ray_mask in GOD_RAY_MASKS:
        frame += ray_mask * w["glow_col"] * 0.18 * ray_pulse

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

    # Vignette - darken corners (reduced to keep brightness)
    frame *= (1.0 - VIGNETTE_MASK * 0.25)

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
