#!/usr/bin/env python3
"""
Cyberpunk City Scene - Blender 5.0 bpy script
Renders OSD character in a Blade Runner-style street environment.
Run: blender --background --python cyberpunk_city.py
"""

import bpy
import bmesh
import math
import os
import random

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OSD_FILE = os.path.join(SCRIPT_DIR, "osd_character.png")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "blender_test.mp4")
OUTPUT_FRAMES = os.path.join(SCRIPT_DIR, "frames", "frame_")

RES_X, RES_Y = 1080, 1920
FPS = 30
DURATION_S = 2
TOTAL_FRAMES = FPS * DURATION_S

random.seed(42)


# ============================================================
# CLEANUP
# ============================================================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


# ============================================================
# MATERIALS
# ============================================================
def make_emission_mat(name, color, strength=5.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = strength
    links.new(emission.outputs[0], output.inputs[0])
    return mat


def make_building_mat(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (*color, 1.0)
    principled.inputs['Roughness'].default_value = 0.85
    principled.inputs['Metallic'].default_value = 0.1
    links.new(principled.outputs[0], output.inputs[0])
    return mat


def make_ground_mat():
    mat = bpy.data.materials.new("WetGround")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.02, 0.02, 0.025, 1.0)
    principled.inputs['Roughness'].default_value = 0.15
    principled.inputs['Metallic'].default_value = 0.3
    principled.inputs['Specular IOR Level'].default_value = 0.8
    links.new(principled.outputs[0], output.inputs[0])
    return mat


def make_window_mat(name, color, strength=2.0):
    """Glowing window material with slight randomization."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    mix = nodes.new('ShaderNodeMixShader')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.01, 0.01, 0.02, 1.0)
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = strength
    mix.inputs['Fac'].default_value = 0.7
    links.new(principled.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])
    return mat


def make_hologram_mat(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    # Alpha blend handled by node setup
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = 8.0
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = 0.5
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])
    return mat


# ============================================================
# GEOMETRY BUILDERS
# ============================================================
def create_building(name, x, y, width, depth, height, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, height / 2))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (width, depth, height)
    obj.data.materials.append(mat)

    # Add extrusions on top for rooftop detail
    if height > 8:
        for i in range(random.randint(1, 3)):
            ex_w = width * random.uniform(0.15, 0.35)
            ex_d = depth * random.uniform(0.15, 0.35)
            ex_h = random.uniform(1.5, 4.0)
            ex_x = x + random.uniform(-width * 0.3, width * 0.3)
            ex_y = y + random.uniform(-depth * 0.3, depth * 0.3)
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(ex_x, ex_y, height + ex_h / 2)
            )
            ext = bpy.context.active_object
            ext.name = f"{name}_roof_{i}"
            ext.scale = (ex_w, ex_d, ex_h)
            ext.data.materials.append(mat)

    return obj


def add_windows_to_building(bld_name, x, y, width, depth, height, side='front'):
    """Add glowing window planes to a building face."""
    win_colors = [
        (1.0, 0.8, 0.4),   # warm yellow
        (0.3, 0.7, 1.0),   # cool blue
        (1.0, 0.3, 0.6),   # pink
        (0.2, 1.0, 0.8),   # cyan
        (0.9, 0.5, 0.1),   # orange
    ]

    floors = int(height / 2.5)
    windows_per_floor = max(1, int(width / 2.0))

    for floor in range(floors):
        for wi in range(windows_per_floor):
            if random.random() < 0.35:  # 35% of windows are lit
                wz = 1.5 + floor * 2.5
                if wz > height - 1:
                    continue

                if side == 'front':
                    wx = x - width / 2 + (wi + 0.5) * (width / windows_per_floor)
                    wy = y - depth / 2 - 0.02
                    rot = (math.pi / 2, 0, 0)
                elif side == 'back':
                    wx = x - width / 2 + (wi + 0.5) * (width / windows_per_floor)
                    wy = y + depth / 2 + 0.02
                    rot = (math.pi / 2, 0, 0)
                elif side == 'left':
                    wx = x - width / 2 - 0.02
                    wy = y - depth / 2 + (wi + 0.5) * (depth / windows_per_floor)
                    rot = (math.pi / 2, 0, math.pi / 2)
                else:  # right
                    wx = x + width / 2 + 0.02
                    wy = y - depth / 2 + (wi + 0.5) * (depth / windows_per_floor)
                    rot = (math.pi / 2, 0, math.pi / 2)

                win_color = random.choice(win_colors)
                win_strength = random.uniform(1.0, 4.0)
                win_mat = make_window_mat(
                    f"win_{bld_name}_{floor}_{wi}",
                    win_color, win_strength
                )

                bpy.ops.mesh.primitive_plane_add(
                    size=1,
                    location=(wx, wy, wz),
                    rotation=rot
                )
                win = bpy.context.active_object
                win.name = f"win_{bld_name}_{floor}_{wi}"
                win.scale = (0.8, 0.8, 1.2)
                win.data.materials.append(win_mat)


def create_neon_sign(name, x, y, z, color, width=2.0, height=0.5, strength=15.0):
    """Create a neon light strip."""
    mat = make_emission_mat(f"neon_{name}", color, strength)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (width, 0.05, height)
    obj.data.materials.append(mat)
    return obj


def create_holographic_billboard(name, x, y, z, color, w=3.0, h=4.0):
    """Create a holographic billboard plane."""
    mat = make_hologram_mat(f"holo_{name}", color)
    bpy.ops.mesh.primitive_plane_add(
        size=1, location=(x, y, z),
        rotation=(math.pi / 2, 0, 0)
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (w, h, 1)
    obj.data.materials.append(mat)
    return obj


# ============================================================
# SCENE CONSTRUCTION
# ============================================================
def build_city():
    """Build a cyberpunk street with buildings on both sides."""

    # -- Ground plane (wet street) --
    ground_mat = make_ground_mat()
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Street"
    ground.data.materials.append(ground_mat)

    # Building materials (dark concrete tones)
    bld_colors = [
        (0.03, 0.03, 0.04),
        (0.04, 0.035, 0.03),
        (0.025, 0.03, 0.035),
        (0.05, 0.04, 0.04),
        (0.035, 0.03, 0.04),
    ]
    bld_mats = [make_building_mat(f"bld_mat_{i}", c) for i, c in enumerate(bld_colors)]

    # -- LEFT side buildings (x < 0) --
    left_buildings = [
        # (name, x, y, width, depth, height)
        ("L1", -6,  -5,  4, 5, 25),
        ("L2", -7,   5,  5, 6, 35),
        ("L3", -5,  15,  3, 4, 18),
        ("L4", -8,  25,  6, 5, 40),
        ("L5", -6,  38,  4, 5, 22),
        # Background left
        ("LB1", -14, -8, 6, 8, 50),
        ("LB2", -15, 10, 7, 7, 45),
        ("LB3", -13, 30, 5, 6, 55),
    ]

    for name, x, y, w, d, h in left_buildings:
        mat = random.choice(bld_mats)
        create_building(name, x, y, w, d, h, mat)
        # Windows on the street-facing side (right side of left buildings)
        add_windows_to_building(name, x, y, w, d, h, side='right')

    # -- RIGHT side buildings (x > 0) --
    right_buildings = [
        ("R1",  6,  -8,  4, 5, 30),
        ("R2",  7,   3,  5, 6, 20),
        ("R3",  5,  12,  3, 4, 28),
        ("R4",  8,  22,  6, 5, 35),
        ("R5",  6,  35,  4, 5, 15),
        # Background right
        ("RB1", 14, -5, 6, 8, 45),
        ("RB2", 15, 15, 7, 7, 60),
        ("RB3", 13, 32, 5, 6, 40),
    ]

    for name, x, y, w, d, h in right_buildings:
        mat = random.choice(bld_mats)
        create_building(name, x, y, w, d, h, mat)
        add_windows_to_building(name, x, y, w, d, h, side='left')

    # -- Neon signs along street --
    neon_configs = [
        ("neon1", -3.9, -3, 6,  (1.0, 0.1, 0.5), 3.0, 0.3, 20),  # pink
        ("neon2",  3.9,  1, 8,  (0.0, 0.8, 1.0), 2.5, 0.3, 18),  # cyan
        ("neon3", -3.9, 10, 5,  (1.0, 0.3, 0.0), 2.0, 0.4, 15),  # orange
        ("neon4",  3.9, 18, 7,  (0.5, 0.0, 1.0), 3.5, 0.3, 22),  # purple
        ("neon5", -3.9, 22, 9,  (0.0, 1.0, 0.4), 2.0, 0.3, 16),  # green
        ("neon6",  3.9, 30, 6,  (1.0, 1.0, 0.0), 2.5, 0.4, 14),  # yellow
        # Vertical neon strips
        ("neon_v1", -4.5, 8, 10, (1.0, 0.0, 0.8), 0.15, 6.0, 12),
        ("neon_v2",  4.5, 20, 12, (0.0, 0.6, 1.0), 0.15, 8.0, 10),
    ]

    neon_objects = []
    for name, x, y, z, color, w, h, strength in neon_configs:
        obj = create_neon_sign(name, x, y, z, color, w, h, strength)
        neon_objects.append(obj)

    # -- Holographic billboards --
    holo_configs = [
        ("holo1", -4.2, 6, 14,  (0.2, 0.8, 1.0), 4, 5),
        ("holo2",  4.2, 25, 12, (1.0, 0.3, 0.8), 3.5, 4),
        ("holo3", -4.5, 35, 16, (0.0, 1.0, 0.5), 3, 3.5),
    ]

    holo_objects = []
    for name, x, y, z, color, w, h in holo_configs:
        obj = create_holographic_billboard(name, x, y, z, color, w, h)
        holo_objects.append(obj)

    return neon_objects, holo_objects


# ============================================================
# OSD CHARACTER
# ============================================================
def add_osd_character():
    """Import OSD character as image plane with alpha."""
    if not os.path.exists(OSD_FILE):
        print(f"WARNING: {OSD_FILE} not found, skipping character")
        return None

    img = bpy.data.images.load(OSD_FILE)
    img.alpha_mode = 'STRAIGHT'
    print(f"  Image loaded: {img.name} ({img.size[0]}x{img.size[1]}, channels={img.channels})")

    # Create plane with correct aspect ratio
    aspect = img.size[0] / img.size[1] if img.size[1] > 0 else 1.0
    plane_height = 4.0
    plane_width = plane_height * aspect

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(0, 5, plane_height / 2 + 0.1),
        rotation=(math.pi / 2, 0, 0)
    )
    osd = bpy.context.active_object
    osd.name = "OSD_Character"
    osd.scale = (plane_width, plane_height, 1)

    # Ensure UVs exist and cover full image
    mesh = osd.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    # Material with image texture + alpha (mix shader for reliable EEVEE alpha)
    mat = bpy.data.materials.new("OSD_Mat")
    mat.use_nodes = True
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Mix shader: transparent where alpha=0, principled where alpha=1
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (400, 0)

    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (200, 100)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (200, -100)
    principled.inputs['Roughness'].default_value = 0.8

    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = img
    tex_node.location = (-200, 0)
    tex_node.extension = 'CLIP'

    # Texture -> Base Color
    links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])

    # Also emit the texture color so character is visible in dark scene
    links.new(tex_node.outputs['Color'], principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 0.8

    # Alpha drives mix factor: 0=transparent, 1=principled
    links.new(tex_node.outputs['Alpha'], mix.inputs['Fac'])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(principled.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])

    # Set material alpha blend mode for EEVEE
    try:
        mat.surface_render_method = 'BLENDED'
    except (AttributeError, TypeError):
        pass
    try:
        mat.blend_method = 'BLEND'
    except (AttributeError, TypeError):
        pass
    try:
        mat.show_transparent_back = True
    except (AttributeError, TypeError):
        pass

    osd.data.materials.append(mat)
    print(f"  OSD material assigned: mix shader (transparent + principled + emission)")

    # Breathing animation (subtle Y-scale pulse)
    osd.keyframe_insert(data_path="scale", frame=1)
    osd.scale = (plane_width, plane_height * 1.01, 1)
    osd.keyframe_insert(data_path="scale", frame=TOTAL_FRAMES // 2)
    osd.scale = (plane_width, plane_height, 1)
    osd.keyframe_insert(data_path="scale", frame=TOTAL_FRAMES)

    # Subtle floating motion (Z oscillation)
    osd.location.z = plane_height / 2 + 0.1
    osd.keyframe_insert(data_path="location", frame=1)
    osd.location.z = plane_height / 2 + 0.25
    osd.keyframe_insert(data_path="location", frame=TOTAL_FRAMES // 2)
    osd.location.z = plane_height / 2 + 0.1
    osd.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)

    return osd


# ============================================================
# LIGHTING
# ============================================================
def setup_lighting():
    """3-point lighting + neon ambient."""

    # Key light (warm, slightly above and to the right)
    bpy.ops.object.light_add(type='AREA', location=(4, 2, 10))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 300
    key.data.color = (1.0, 0.85, 0.7)
    key.data.size = 4
    key.rotation_euler = (math.radians(60), 0, math.radians(-30))

    # Fill light (cool, from the left, softer)
    bpy.ops.object.light_add(type='AREA', location=(-5, 0, 6))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 100
    fill.data.color = (0.5, 0.7, 1.0)
    fill.data.size = 6
    fill.rotation_euler = (math.radians(50), 0, math.radians(40))

    # Rim light (behind and above, strong backlight)
    bpy.ops.object.light_add(type='AREA', location=(0, 12, 8))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 500
    rim.data.color = (0.3, 0.6, 1.0)
    rim.data.size = 3
    rim.rotation_euler = (math.radians(120), 0, 0)

    # Neon bounce lights (colored point lights at street level)
    bounce_configs = [
        ((-3, -2, 3), (1.0, 0.1, 0.5), 80),   # pink bounce
        ((3, 5, 3),   (0.0, 0.8, 1.0), 60),    # cyan bounce
        ((-2, 15, 4), (1.0, 0.3, 0.0), 50),    # orange bounce
        ((2, 25, 3),  (0.5, 0.0, 1.0), 70),    # purple bounce
    ]

    for loc, color, energy in bounce_configs:
        bpy.ops.object.light_add(type='POINT', location=loc)
        light = bpy.context.active_object
        light.name = f"Bounce_{color}"
        light.data.energy = energy
        light.data.color = color
        light.data.shadow_soft_size = 2.0


# ============================================================
# CAMERA
# ============================================================
def setup_camera():
    """Animated camera pushing forward through the city."""
    bpy.ops.object.camera_add(location=(0, -8, 3.5))
    cam = bpy.context.active_object
    cam.name = "MainCamera"
    bpy.context.scene.camera = cam

    cam.data.lens = 35
    cam.data.sensor_width = 36

    # Depth of field
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 13.0
    cam.data.dof.aperture_fstop = 2.8

    # Start position
    cam.location = (0, -8, 3.5)
    cam.rotation_euler = (math.radians(80), 0, 0)
    cam.keyframe_insert(data_path="location", frame=1)
    cam.keyframe_insert(data_path="rotation_euler", frame=1)

    # End position (pushed forward + slight drift)
    cam.location = (0.3, -4, 3.3)
    cam.rotation_euler = (math.radians(82), 0, math.radians(1))
    cam.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)
    cam.keyframe_insert(data_path="rotation_euler", frame=TOTAL_FRAMES)

    return cam


# ============================================================
# RAIN PARTICLES
# ============================================================
def setup_rain():
    """Rain particle system."""
    # Emitter plane above the scene
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 15, 30))
    emitter = bpy.context.active_object
    emitter.name = "RainEmitter"

    # Hide emitter from render
    emitter.hide_render = True

    # Add particle system
    mod = emitter.modifiers.new("Rain", type='PARTICLE_SYSTEM')
    ps = mod.particle_system.settings

    ps.count = 500
    ps.frame_start = 1
    ps.frame_end = TOTAL_FRAMES
    ps.lifetime = TOTAL_FRAMES + 10
    ps.emit_from = 'FACE'

    # Rain physics
    ps.physics_type = 'NEWTON'
    ps.mass = 0.01
    ps.normal_factor = 0
    ps.factor_random = 0.2

    # Gravity will pull them down (default scene gravity)
    ps.effector_weights.gravity = 1.0

    # Render as objects - create a thin cylinder for raindrop
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.01, depth=0.3,
        location=(100, 100, 100)  # off-screen
    )
    drop = bpy.context.active_object
    drop.name = "Raindrop"

    drop_mat = make_emission_mat("RaindropMat", (0.6, 0.7, 0.9), 0.5)
    drop.data.materials.append(drop_mat)

    ps.render_type = 'OBJECT'
    ps.instance_object = drop
    ps.particle_size = 1.0

    drop.hide_render = True
    drop.hide_viewport = True

    return emitter


# ============================================================
# VOLUMETRIC FOG
# ============================================================
def setup_atmosphere():
    """Add volumetric fog to the world."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputWorld')

    # Dark sky background
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.005, 0.005, 0.015, 1.0)
    bg.inputs['Strength'].default_value = 0.5

    # Volume scatter for fog
    vol_scatter = nodes.new('ShaderNodeVolumeScatter')
    vol_scatter.inputs['Color'].default_value = (0.4, 0.5, 0.7, 1.0)
    vol_scatter.inputs['Density'].default_value = 0.015
    vol_scatter.inputs['Anisotropy'].default_value = 0.3

    links.new(bg.outputs[0], output.inputs['Surface'])
    links.new(vol_scatter.outputs[0], output.inputs['Volume'])


# ============================================================
# NEON FLICKER ANIMATION
# ============================================================
def animate_neon_flicker(neon_objects):
    """Animate neon sign emission strength for flicker effect."""
    for obj in neon_objects:
        if not obj.data.materials:
            continue
        mat = obj.data.materials[0]
        if not mat.use_nodes:
            continue

        # Find emission node
        emission = None
        for node in mat.node_tree.nodes:
            if node.type == 'EMISSION':
                emission = node
                break
        if emission is None:
            continue

        base_strength = emission.inputs['Strength'].default_value

        # Keyframe flicker pattern
        for f in range(1, TOTAL_FRAMES + 1):
            # Pseudo-random flicker
            flicker = 0.7 + 0.3 * math.sin(f * 0.8 + hash(obj.name) % 100)
            if random.random() < 0.05:  # occasional dip
                flicker *= 0.3
            emission.inputs['Strength'].default_value = base_strength * flicker
            emission.inputs['Strength'].keyframe_insert(
                data_path="default_value", frame=f
            )


def animate_holograms(holo_objects):
    """Animate holographic billboard opacity."""
    for obj in holo_objects:
        if not obj.data.materials:
            continue
        mat = obj.data.materials[0]
        if not mat.use_nodes:
            continue

        mix = None
        for node in mat.node_tree.nodes:
            if node.type == 'MIX_SHADER':
                mix = node
                break
        if mix is None:
            continue

        for f in range(1, TOTAL_FRAMES + 1):
            pulse = 0.4 + 0.2 * math.sin(f * 0.3 + hash(obj.name) % 50)
            if random.random() < 0.03:
                pulse = 0.1  # glitch
            mix.inputs['Fac'].default_value = pulse
            mix.inputs['Fac'].keyframe_insert(
                data_path="default_value", frame=f
            )


# ============================================================
# RENDER SETTINGS
# ============================================================
def setup_render():
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES

    # Use EEVEE for fast rendering (seconds per frame vs minutes)
    scene.render.engine = 'BLENDER_EEVEE'
    print("Using EEVEE renderer (fast)")

    # EEVEE quality settings - use try/except for Blender 5.0 API differences
    eevee = scene.eevee
    for attr, val in [
        ('taa_render_samples', 64),
        ('use_volumetric_lights', True),
        ('use_volumetric_shadows', True),
        ('volumetric_tile_size', 8),
        ('volumetric_end', 100.0),
        ('use_ssr', True),
        ('use_ssr_refraction', True),
        ('ssr_quality', 0.5),
        ('use_bloom', True),
        ('bloom_threshold', 0.8),
        ('bloom_intensity', 0.5),
        ('bloom_radius', 6.0),
        ('shadow_cube_size', '1024'),
        ('shadow_cascade_size', '2048'),
    ]:
        try:
            setattr(eevee, attr, val)
        except (AttributeError, TypeError):
            pass  # Skip unsupported properties

    # Resolution
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS

    # Output
    scene.render.filepath = OUTPUT_FRAMES
    scene.render.image_settings.file_format = 'PNG'

    # Motion blur
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5

    # Film / color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  CYBERPUNK CITY - Blender 5.0 Scene Builder")
    print("=" * 60)

    print("  Clearing scene...")
    clear_scene()

    print("  Building city geometry...")
    neon_objects, holo_objects = build_city()

    print("  Adding OSD character...")
    osd = add_osd_character()

    print("  Setting up 3-point lighting...")
    setup_lighting()

    print("  Setting up camera...")
    cam = setup_camera()

    print("  Adding rain particles...")
    setup_rain()

    print("  Adding volumetric fog...")
    setup_atmosphere()

    print("  Animating neon flicker...")
    animate_neon_flicker(neon_objects)
    animate_holograms(holo_objects)

    print("  Configuring render settings...")
    setup_render()

    # Save .blend file
    blend_path = os.path.join(SCRIPT_DIR, "cyberpunk_city.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"  Saved: {blend_path}")

    # Create frames directory
    frames_dir = os.path.join(SCRIPT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Render animation
    print(f"\n  Rendering {TOTAL_FRAMES} frames at {RES_X}x{RES_Y}...")
    print("  This will take a while...\n")
    bpy.ops.render.render(animation=True)

    # Encode to MP4 with ffmpeg
    print("\n  Encoding to MP4...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        OUTPUT_FILE
    ], check=True, capture_output=True)

    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 60)
    print("  DONE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
