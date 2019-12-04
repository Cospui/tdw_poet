import numpy as np
import random
import math
from scipy.spatial import distance
from tdw.output_data import IsOnNavMesh
from PIL import Image
import io
import os
from tdw.controller import Controller
import tdw.librarian as librarian


class TDWUtils:
    """
    Utility functions for controllers.

    Usage: `from tdw.tdw_utils import TDWUtils`
    """

    VECTOR3_ZERO = {"x": 0, "y": 0, "z": 0}

    @staticmethod
    def vector3_to_array(vector3):
        """
        Convert a Vector3 object to a numpy array.

        :param vector3: The Vector3 object.

        :return A numpy array.
        """

        return np.array([vector3["x"], vector3["y"], vector3["z"]])

    @staticmethod
    def array_to_vector3(arr):
        """
        Convert a numpy array to a Vector3.

        :param arr: The numpy array.

        :return A Vector3.
        """

        return {"x": arr[0], "y": arr[1], "z": arr[2]}

    @staticmethod
    def get_random_point_in_circle(center, radius):
        """
        Get a random point in a circle, defined by a center and radius.

        :param center: The center of the circle.
        :param radius: The radius of the circle.

        :return A numpy array.
        """

        alpha = 2 * math.pi * random.random()
        r = radius * math.sqrt(random.random())
        x = r * math.cos(alpha) + center[0]
        z = r * math.sin(alpha) + center[2]

        return np.array([x, 0, z])

    @staticmethod
    def get_magnitude(vector3):
        """
        Get the magnitude of a Vector3.

        :param vector3: The Vector3 object.

        :return The vector magnitude.
        """

        return np.linalg.norm(TDWUtils.vector3_to_array(vector3))

    @staticmethod
    def extend_line(p0, p1, d, clamp_y=True):
        """
        Extend the line defined by p0 to p1 by distance d. Clamps the y value to 0.

        :param p0: The origin.
        :param p1: The second point.
        :param d: The distance of which the line is to be extended.
        :param clamp_y: Clamp the y value to 0.

        :return: The position at distance d.
        """

        if clamp_y:
            p0[1] = 0
            p1[1] = 0

        # Get the distance between the two points.
        d0 = distance.euclidean(p0, p1)
        # Get the total distance.
        d_total = d0 + d

        return p1 + ((p1 - p0) * d_total)

    @staticmethod
    def get_distance(vector3_0, vector3_1):
        """
        Calculate the distance between two Vector3 objects.

        :param vector3_0: The first Vector3.
        :param vector3_1: The second Vector3.

        :return The distance.
        """

        return distance.euclidean(TDWUtils.vector3_to_array(vector3_0), TDWUtils.vector3_to_array(vector3_1))

    @staticmethod
    def get_box(width, length):
        """
        Create a box GridPoint objects of the given dimensions.

        :param width: The width of the box.
        :param length: The length of the box.

        :return The box.
        """

        box = []
        for x in range(width):
            for y in range(length):
                if x == 0 or x == width - 1 or y == 0 or y == length - 1:
                    box.append({"x": x, "y": y})
        return box

    @staticmethod
    def get_vector3(x, y, z):
        """
        Returns a dictionary: {"x": x, "y", y, "z": z}
        :param x: The x value.
        :param y: The y value.
        :param z: The z value.

        :return: A Vector3.
        """

        return {"x": x, "y": y, "z": z}

    @staticmethod
    def create_empty_room(width, length, num_envs=1):
        """
        Create an empty room.

        :param width: The width of the room.
        :param length: The length of the room.
        :param num_envs: The number of rooms.

        :return: The command.
        """

        return {"$type": "create_exterior_walls", "walls": TDWUtils.get_box(width, length), "num_envs": num_envs}

    @staticmethod
    def save_images(images, frame, output_directory="dist", filename=None, resize_to=None):
        """
        Save each image in the Images object.

        :param images: The Images object. Contains each capture pass plus metadata.
        :param frame: The current frame number.
        :param output_directory: The directory to write images to.
        :param filename: The filename of each image.
        :param resize_to: Specify a (width, height) tuple to resize the images to. This is slower than saving as-is.
        """

        frame = Controller.get_frame(frame)

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        for i in range(images.get_num_passes()):
            if not filename:
                fi = str(frame) + "_" + images.get_avatar_id() + "_" + str(images.get_env_id()) +\
                     images.get_pass_mask(i) + "." + images.get_extension(i)
            else:
                fi = filename + "." + images.get_extension(i)

            if resize_to:
                Image.open(io.BytesIO(images.get_image(i))).resize((resize_to[0], resize_to[1]), Image.LANCZOS)\
                    .save(os.path.join(output_directory, fi))
            else:
                with open(os.path.join(output_directory, fi), "wb") as f:
                    f.write(images.get_image(i))

    @staticmethod
    def get_random_position_on_nav_mesh(c, width, length, x_e=0, z_e=0, bake=True, rng=random.uniform):
        """
        Returns a random position on a NavMesh.

        :param c: The controller.
        :param width: The width of the environment.
        :param length: The length of the environment.
        :param bake: If true, send bake_nav_mesh.
        :param rng: Random number generator.
        :param x_e: The x position of the environment.
        :param z_e: The z position of the environment.

        :return The coordinates as a tuple (x, y, z)
        """

        if bake:
            c.communicate({'$type': 'bake_nav_mesh'})

        # Try to find a valid position on the NavMesh.
        is_on = False
        x, y, z = (0, 0, 0)
        while not is_on:
            # Get a random position.
            x = rng(-width / 2, width / 2) + x_e
            z = rng(-length / 2, length / 2) + z_e
            resp = c.communicate(
                {'$type': 'send_is_on_nav_mesh',
                 'position': {'x': x, 'y': 0, 'z': z},
                 'max_distance': 4.0
                 })
            answer = IsOnNavMesh(resp[0])
            is_on = answer.get_is_on()
            x, y, z = answer.get_position()
        return x, y, z

    @staticmethod
    def set_visual_material(substructure, object_id, material):
        """
        Returns a list of commands to set ALL visual materials on an object to a single material.

        :param substructure: The metadata substructure of the object.
        :param object_id: The ID of the object in the scene.
        :param material: The name of the new material.

        :return The list of commands.
        """

        commands = []
        for s in substructure:
            for i in range(len(s["materials"])):
                commands.append({"$type": "set_visual_material",
                                 "id": object_id,
                                 "new_material_name": material,
                                 "object_name": s["name"],
                                 "old_material_index": i})
        return commands

    @staticmethod
    def get_depth_values(image, screen_size, far_plane, near_plane):
        """
        Get the depth values of each pixel in a _depth image pass.

        :param image: The image pass as a numpy array.
        :param screen_size: The screen size
        :param far_plane: The far clipping plane of the camera.
        :param near_plane: The near clipping plane of the camera.
        """

        # Convert the image to a 2D image array.
        image = np.array(Image.open(io.BytesIO(image)))

        depth = (image[:, :, 0] * screen_size * screen_size + image[:, :, 1] * screen_size + image[:, :, 2]) / \
                (screen_size * screen_size * screen_size) * (far_plane + near_plane)
        return depth

    @staticmethod
    def create_avatar(avatar_type="A_Img_Caps_Kinematic", avatar_id="a", position=None, look_at=None):
        """
        This is a wrapper for `create_avatar` and, optionally, `teleport_avatar_to` and `look_at_position`.
        Returns a list of commands.

        :param avatar_type: The type of avatar.
        :param avatar_id: The avatar ID.
        :param position: The position of the avatar. If this is None, the avatar won't teleport.
        :param look_at: If this isn't None, the avatar will look at this position.
        """

        # Create the avatar.
        commands = [{"$type": "create_avatar",
                     "type": avatar_type,
                     "id": avatar_id}]

        # Teleport the avatar.
        if position:
            commands.append({"$type": "teleport_avatar_to",
                             "avatar_id": avatar_id,
                             "position": position})
        if look_at:
            commands.append({"$type": "look_at_position",
                             "avatar_id": avatar_id,
                             "position": look_at})
        return commands
