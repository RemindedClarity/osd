# Orange Sweater Dude v5.0

Cinematic animation pipeline that renders OSD across randomly selected historical/sci-fi eras with material transformations, particle effects, and camera work. Outputs an 8-second seamless loop at 1080x1920.

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

- Python 3
- NumPy
- Pillow
- MoviePy
- FFmpeg

## Usage

Place `osd_character.png` in the same directory, then:

```
python orange_sweater_dude.py
```

Outputs `osd_monday.mp4`.
