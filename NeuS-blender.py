# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import bpy
import bpy_extras
import mathutils
from mathutils import Matrix
from mathutils import Vector
import numpy as np


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix( # ここで一応blenderからopencvへの変換が担保されているはず
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam) # ここが問題？
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--views', type=int, default=50, required=False)
    parser.add_argument('--width', type=int, default=800, required=False)
    parser.add_argument('--height', type=int, default=800, required=False)
    parser.add_argument('--format', type=str, default='PNG', required=False)
    parser.add_argument('--random_views', action='store_true')
    parser.add_argument('--upper_views', action='store_true')
    parser.add_argument('--radius', type=float, default=4.0, required=False)
    parser.add_argument('--panorama', action='store_true')
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--hdr', action='store_true')

    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    DEBUG = args.debug
    VIEWS = args.views
    WIDTH_RESOLUTION = args.width
    HEIGHT_RESOLUTION = args.height

    RESULTS_PATH = args.output_dir
    DEPTH_SCALE = 1.4
    COLOR_DEPTH = 8
    FORMAT = args.format
    RANDOM_VIEWS = args.random_views
    UPPER_VIEWS = args.upper_views

    # fp = bpy.path.abspath(f"//{RESULTS_PATH}")
    # fp_image = bpy.path.abspath(f"//{RESULTS_PATH}") + "/image"
    # fp_mask = bpy.path.abspath(f"//{RESULTS_PATH}") + "/coarse_mask"
    fp = RESULTS_PATH
    fp_png= RESULTS_PATH+"/image"
    fp_hdr= RESULTS_PATH+"/hdr"
    fp_panorama = RESULTS_PATH+"/panorama"
    fp_mask = RESULTS_PATH+"/coarse_mask"

    if not os.path.exists(fp):
        os.makedirs(fp)
    if not os.path.exists(fp_png):
        os.makedirs(fp_png)
    if not os.path.exists(fp_hdr):
        os.makedirs(fp_hdr)
    if not os.path.exists(fp_panorama):
        os.makedirs(fp_panorama)
    if not os.path.exists(fp_mask):
        os.makedirs(fp_mask)

    def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True


    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    #bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.image_settings.file_format = str(FORMAT)
    bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

    if not DEBUG:
        # Create input render layer node.
        render_layers = tree.nodes.new('CompositorNodeRLayers')

        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        if FORMAT == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        else:
            # Remap as other types can not represent the full range of depth.
            map = tree.nodes.new(type="CompositorNodeMapValue")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.offset = [-0.7]
            map.size = [DEPTH_SCALE]
            map.use_min = True
            map.min = [0]
            links.new(render_layers.outputs['Depth'], map.inputs[0])

            links.new(map.outputs[0], depth_file_output.inputs[0])

        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    # bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.film_transparent = False

    '''
    Added
    '''
    # set environment texture
    environment_texture_path = "./exr/bismarckturm_4k.exr"
    bpy.context.scene.world.node_tree.nodes['Environment Texture'].image = bpy.data.images.load(environment_texture_path)

    # Create collection for objects not to render with background


    objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

    def parent_obj_to_camera(b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scn = bpy.context.scene
        scn.collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty
        # scn.objects.active = b_empty
        return b_empty



    from math import radians

    scene = bpy.context.scene
    scene.render.resolution_x = WIDTH_RESOLUTION
    scene.render.resolution_y = HEIGHT_RESOLUTION
    scene.render.resolution_percentage = 100

    # radius = 4.0
    # theta = 30.0 * np.pi / 180.0
    # phi = 0.0 * np.pi / 180.0
    # cx = radius * np.sin(theta) * np.cos(phi)
    # cy = radius * np.sin(theta) * np.sin(phi)
    # cz = radius * np.cos(theta)

    cam = scene.objects['Camera']
    # cam.location = (cx, cy, cz)
    # r = 4.0
    r = args.radius
    theta = np.pi / 6
    cam.location = (0.0, r, r * np.sin(np.pi - theta))

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    if args.panorama:
        cam.data.type = 'PANO'
        cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'

    stepsize = 360.0 / VIEWS
    rotation_mode = 'XYZ'

    if not DEBUG:
        for output_node in [depth_file_output, normal_file_output]:
            output_node.base_path = ''

    out_data['frames'] = []

    npz_records = {}


    if RANDOM_VIEWS:
        for i in range(0, VIEWS):
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
                # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

            if DEBUG:
                break
            else:
                if args.panorama:
                    cam.data.type = 'PANO'
                    cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'
                    scene.render.filepath = fp_panorama + '/{0:03d}'.format(int(i))
                    scene.render.image_settings.file_format = 'HDR'
                    bpy.ops.render.render(write_still=True)  # render still
                if args.png:
                    cam.data.type = 'PERSP'
                    scene.render.filepath = fp_png + '/{0:03d}'.format(int(i))
                    scene.render.image_settings.file_format = 'PNG'
                    bpy.ops.render.render(write_still=True)  # render still
                if args.hdr:
                    cam.data.type = 'PERSP'
                    scene.render.filepath = fp_hdr + '/{0:03d}'.format(int(i))
                    scene.render.image_settings.file_format = 'HDR'
                    bpy.ops.render.render(write_still=True)  # render still
                # print("cancel redering")

                # render mask image
                # set configuration to mask image version
                cam.data.type = 'PERSP'
                scene.render.image_settings.file_format = 'PNG'  # set output format to .png
                scene.render.image_settings.color_mode = 'RGBA'
                bpy.context.scene.render.film_transparent = True
                bpy.context.scene.view_settings.exposure = -10.0
                scene.render.filepath = fp_mask + '/{0:03d}'.format(int(i))
                bpy.ops.render.render(write_still=True)  # render still

                # set configuration to original version
                bpy.context.scene.render.film_transparent = False
                bpy.context.scene.view_settings.exposure = 1.0

            # frame_data = {
            #     'file_path': scene.render.filepath,
            #     # 'rotation': radians(stepsize),
            #     # 'transform_matrix': listify_matrix(cam.matrix_world)
            # }
            # out_data['frames'].append(frame_data)


            P, K, Rt = get_3x4_P_matrix_from_blender(cam)

            key_P = "world_mat_{}".format(i)
            key_K = "K_{}".format(i)
            key_Rt = "Rt_{}".format(i)
            key_matrix_world = "matrix_world_{}".format(i)
            npz_records[key_P] = np.array(listify_matrix(P))
            npz_records[key_K] = np.array(listify_matrix(K))
            npz_records[key_Rt] = np.array(listify_matrix(Rt))
            npz_records[key_matrix_world] = np.array(listify_matrix(cam.matrix_world))

            if RANDOM_VIEWS:
                if UPPER_VIEWS:
                    rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                    b_empty.rotation_euler = rot
                else:
                    b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        # print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))

        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)
        # 30 <= theta <= 150
        # 0 <= phi <= 360

        theta_start = np.pi / 6.0
        theta_end = np.pi * 5.0 / 6.0
        phi_start = 0.0
        phi_end = np.pi * 2.0
        r = args.radius

        theta_chunks = 10 # 15 degrees increments
        phi_chunks = 24 # 15 degrees increments
        i = 0
        for theta_step in range(0, theta_chunks):
            for phi_step in range(0, phi_chunks):
                print("Rotation theta:{}/{}, phi:{}/{}".format(theta_step, theta_chunks, phi_step, phi_chunks))

                theta = theta_start + (theta_end - theta_start) * theta_step / theta_chunks
                phi = phi_start + (phi_end - phi_start) * phi_step / phi_chunks
                cam.location = (r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta))
                if DEBUG:
                    break
                else:
                    if args.panorama:
                        cam.data.type = 'PANO'
                        cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'
                        scene.render.filepath = fp_panorama + '/{0:03d}'.format(int(i))
                        scene.render.image_settings.file_format = 'HDR'
                        bpy.ops.render.render(write_still=True)  # render still
                    if args.png:
                        cam.data.type = 'PERSP'
                        scene.render.filepath = fp_png + '/{0:03d}'.format(int(i))
                        scene.render.image_settings.file_format = 'PNG'
                        bpy.ops.render.render(write_still=True)  # render still
                    if args.hdr:
                        cam.data.type = 'PERSP'
                        scene.render.filepath = fp_hdr + '/{0:03d}'.format(int(i))
                        scene.render.image_settings.file_format = 'HDR'
                        bpy.ops.render.render(write_still=True)  # render still

                    # print("cancel redering")

                    # render mask image
                    # set configuration to mask image version
                    cam.data.type = 'PERSP'
                    scene.render.image_settings.file_format = 'PNG'  # set output format to .png
                    scene.render.image_settings.color_mode = 'RGBA'
                    bpy.context.scene.render.film_transparent = True
                    bpy.context.scene.view_settings.exposure = -10.0
                    scene.render.filepath = fp_mask + '/{0:03d}'.format(int(i))

                    bpy.ops.render.render(write_still=True)  # render still

                    # set configuration to original version
                    bpy.context.scene.render.film_transparent = False
                    bpy.context.scene.view_settings.exposure = 1.0

                # frame_data = {
                #     'file_path': scene.render.filepath,
                #     # 'rotation': radians(stepsize),
                #     # 'transform_matrix': listify_matrix(cam.matrix_world)
                # }
                # out_data['frames'].append(frame_data)

                P, K, Rt = get_3x4_P_matrix_from_blender(cam)

                key_P = "world_mat_{}".format(i)
                key_K = "K_{}".format(i)
                key_Rt = "Rt_{}".format(i)
                key_matrix_world = "matrix_world_{}".format(i)
                npz_records[key_P] = np.array(listify_matrix(P))
                npz_records[key_K] = np.array(listify_matrix(K))
                npz_records[key_Rt] = np.array(listify_matrix(Rt))
                npz_records[key_matrix_world] = np.array(listify_matrix(cam.matrix_world))

                i += 1


    if not DEBUG:
        # with open(fp + '/' + 'transforms.json', 'w') as out_file:
        #     json.dump(out_data, out_file, indent=4)
        np.savez(os.path.join(fp, 'cameras'), **npz_records)
