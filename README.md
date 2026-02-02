# Orange Sweater Dude v5.0

Cinematic animation pipeline that renders OSD across randomly selected historical/sci-fi eras with material transformations, particle effects, and camera work. Includes both a 2D NumPy/MoviePy pipeline and a 3D Blender scene.

## Eras

| Era | Material | Background FX |
|-----|----------|---------------|
| Jurassic | Molten lava | Meteor strike, shockwave |
| Ancient Egypt | Sand dissolving | Sandstorm, lightning |
| Medieval Siege | Pure fire | Castle collapsing |
| Wild West | Dust ghost | Tornado, debris |
| 1980s Arcade | Pixelated RGB split | Pac-Man, Tetris |
| Cyberpunk 2077 | Glitch corrupted | Flying cars, billboards |
| Nuclear Apocalypse | X-ray ghost | Mushroom cloud |
| Warp Speed | Star stretched | Wormhole opening |

4 eras are randomly selected per render (seeded).

## Effects

- Material shaders per era (lava flow, glitch scanlines, x-ray glow, etc.)
- Parallax depth layers on environment particles
- Cinematic lighting: rim light, god rays, ambient occlusion, vignette
- Smooth camera motion with easing (zoom drift, sway, shake)
- Multi-phase teleport: particle gather, burst explosion, shockwave, reform
- Color grading with S-curve and chromatic aberration
- Motion blur trails on particles

## Requirements

### 2D Pipeline
- Python 3
- NumPy
- Pillow
- MoviePy
- FFmpeg

### Blender Pipeline
- Blender 5.0+
- FFmpeg

## Usage

### 2D Pipeline

Place `osd_character.png` in the same directory, then:

```
python orange_sweater_dude.py
```

Outputs `osd_monday.mp4` (30s seamless loop, 1080x1920, 30fps).

### Blender 3D Pipeline

```
blender --background --python cyberpunk_city.py
```

Outputs `blender_test.mp4` (2s test render, 1080x1920, 30fps) and `cyberpunk_city.blend`.

## Blender Scene: Cyberpunk City

A Blade Runner-style street environment built with the Blender Python API (bpy).

**Geometry:**
- 16 buildings across left/right sides with rooftop extrusions
- 3 depth layers (foreground, midground, background)
- Wet reflective street with low-roughness ground plane

**Lighting:**
- 3-point lighting (warm key, cool fill, blue rim)
- 4 colored neon bounce lights at street level
- 8 neon signs with animated flicker (pink, cyan, orange, purple, green, yellow)
- 3 holographic billboards with emission + transparency

**Effects:**
- Volumetric fog/atmosphere between buildings
- Rain particle system (500 particles)
- Glowing windows (randomized warm/cool colors)
- Shallow depth of field (f/2.8, 35mm lens)
- Camera push-forward animation
- Filmic color management with high contrast
- Motion blur

**Character:**
- OSD imported as alpha image plane with mix shader (transparent + principled)
- Subtle breathing and floating animation
