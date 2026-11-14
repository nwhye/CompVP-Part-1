import bpy
import math
import os
from mathutils import Vector

FLAME_SCALE = 4.0


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (0, -2.5, 0)
    cam.rotation_euler = (math.radians(90), 0, 0)

    # main light
    light_data = bpy.data.lights.new("HarshLight", type='POINT')
    light_data.energy = 500
    light_data.shadow_soft_size = 0.02
    light = bpy.data.objects.new("HarshLight", light_data)
    scene.collection.objects.link(light)

    # lil ambient light
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.02, 1)
    scene.world = world

    # Render settings
    engines = {e.identifier for e in
               bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    # material
    mat = bpy.data.materials.new("MatteFace")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.9

    return light, mat


def batch_render_faces(mesh_dir, output_dir):
    light_obj, mat = setup_scene()

    light_positions = [
        (1.5, -1.0, 1.2),    # side key
        (-1.5, -1.0, 1.2),   # opposite side key
        #(0.0, -1.0, 2.0),    # top
        #(0.0, -1.5, -1.5),   # under-lighting
    ]

    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    obj_files = sorted(f for f in os.listdir(mesh_dir) if f.endswith('.obj'))

    for file_name in obj_files:
        subject_id = os.path.splitext(file_name)[0]

        bpy.ops.wm.obj_import(
            filepath=os.path.join(mesh_dir, file_name),
            forward_axis='NEGATIVE_Z', up_axis='Y',
        )
        obj = bpy.context.selected_objects[0]

        for p in obj.data.polygons:
            p.use_smooth = True
        obj.data.materials.append(mat)

        obj.scale = (FLAME_SCALE,) * 3
        bpy.context.view_layer.update()
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        obj.location -= sum(corners, Vector()) / 8

        for i, pos in enumerate(light_positions):
            light_obj.location = pos
            bpy.context.scene.render.filepath = os.path.join(
                output_dir, "original", f"{subject_id}_light_{i}.png")
            bpy.ops.render.render(write_still=True)

        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        if bpy.data.filepath:
            return os.path.dirname(bpy.data.filepath)
        return os.getcwd()


if __name__ == "__main__":
    BASE = get_script_dir()
    MESH_DIRECTORY = os.path.join(BASE, "generated_objs")
    OUTPUT_DIRECTORY = os.path.join(BASE, "out")
    batch_render_faces(MESH_DIRECTORY, OUTPUT_DIRECTORY)