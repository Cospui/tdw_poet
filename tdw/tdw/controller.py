import zmq
import json
import os


class Controller(object):
    """
    Base class for all controllers.

    Usage: `from tdw.controller import Controller`
    """

    def __init__(self, port=1071):
        """
        Create the network socket and bind the socket to the port.

        :param port: The port number.
        """

        context = zmq.Context()

        self.socket = context.socket(zmq.REP)
        self.socket.bind('tcp://*:' + str(port))

        self.socket.recv()

    def communicate(self, commands):
        """
        Send commands and receive output data in response.

        :param commands: A list of JSON commands.
        :return The output data from the build.
        """

        if not isinstance(commands, list):
            commands = [commands]

        self.socket.send_multipart([json.dumps(commands).encode('utf-8')])

        return self.socket.recv_multipart()

    def start(self, scene="ProcGenScene"):
        """
        Init TDW.

        :param scene: The scene to load.
        """

        self.communicate([{"$type": "load_scene", "scene_name": scene}])

    def load_streamed_scene(self, scene="tdw_room_2018"):
        """
        Load a streamed scene.
        :param scene: The name of the streamed scene.
        """

        self.communicate([{"$type": "load_streamed_scene", "scene_name": scene}])

    def add_object(self, model_name, position={"x": 0, "y": 0, "z": 0}, rotation={"x": 0, "y": 0, "z": 0}, env_id=0):
        """
        Add a model from the model library to the scene.

        :param model_name: The name of the model.
        :param position: The position of the model.
        :param rotation: The starting rotation of the model, in Euler angles.
        :param env_id: The environment ID.
        :return The ID of the new object.
        """

        object_id = Controller.get_unique_id()
        self.communicate({"$type": "add_object",
                          "env_id": env_id,
                          "model_name": model_name,
                          "position": position,
                          "rotation": rotation,
                          "id": object_id})
        return object_id

    @staticmethod
    def get_unique_id():
        """
        Generate a unique integer. Useful when creating objects.

        :return The new unique ID.
        """

        return int.from_bytes(os.urandom(3), byteorder='big')

    @staticmethod
    def get_frame(frame):
        """
        Converts the frame byte array to an integer.

        :param frame: The frame as bytes.
        :return: The frame as an integer.
        """

        return int.from_bytes(frame, byteorder='big')
