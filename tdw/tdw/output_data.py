from tdw.FBOutput import Vector3, Quaternion, PassMask, Color
from tdw.FBOutput import Environments as Envs
from tdw.FBOutput import Transforms as Trans
from tdw.FBOutput import Rigidbodies as Rigis
from tdw.FBOutput import Bounds as Bouns
from tdw.FBOutput import Images as Imags
from tdw.FBOutput import AvatarKinematic as AvKi
from tdw.FBOutput import AvatarNonKinematic as AvNoKi
from tdw.FBOutput import AvatarSimpleBody as AvSi
from tdw.FBOutput import AvatarStickyMitten as AvSM
from tdw.FBOutput import SegmentationColors as Segs
from tdw.FBOutput import AvatarSegmentationColor as AvSC
from tdw.FBOutput import AvatarStickyMittenSegmentationColors as AvSMSC
from tdw.FBOutput import IsOnNavMesh as IsNM
from tdw.FBOutput import IdPassGrayscale as IdGS
from tdw.FBOutput import Collision as Col
from tdw.FBOutput import ImageSensors as ImSe
from tdw.FBOutput import CameraMatrices as CaMa
from tdw.FBOutput import IdPassSegmentationColors as IdSC
from tdw.FBOutput import ArrivedAtNavMeshDestination as Arri
from tdw.FBOutput import FlexParticles as Flex
from tdw.FBOutput import Video as Vid


class OutputDataUndefinedError(Exception):
    pass


class OutputData(object):
    def __init__(self, b):
        self.bytes = bytearray(b)
        self.data = self.get_data()

    def get_data(self):
        raise OutputDataUndefinedError("Undefined!")

    @staticmethod
    def get_data_type_id(b):
        """
        Returns the ID of the serialized object.
        :param b: A byte array.
        """

        return b[4:8].decode('utf-8')

    @staticmethod
    def _get_vector3(constructor):
        """
        Returns x, y, and z values of a Vector3, given a constructor.

        :param constructor: A constructor that accepts 1 parameter of type Vector3.
        """

        return OutputData._get_xyz(constructor(Vector3.Vector3()))

    @staticmethod
    def _get_xyz(vector3):
        """
        returns the x, y, and z values of a Vector3, given the Vector3 object.

        :param vector3: The Vector3 object.
        """

        return vector3.X(), vector3.Y(), vector3.Z()

    @staticmethod
    def _get_quaternion(constructor):
        """
        Returns x, y, z, and w values of a Quaternion, given a constructor.

        :param constructor: A constructor that accepts 1 parameter of type Quaternion.
        """

        return OutputData._get_xyzw(constructor(Quaternion.Quaternion()))

    @staticmethod
    def _get_xyzw(quaternion):
        """
        returns the x, y, and z values of a Quaternion, given the Quaternion object.

        :param quaternion: The Quaternion object.
        """

        return quaternion.X(), quaternion.Y(), quaternion.Z(), quaternion.W()

    @staticmethod
    def _get_color(constructor):
        """
        Returns the r, g, and b values of a Color, given a constructor.

        :param constructor: A constructor that accepts 1 parameter of type Color.
        :return:
        """

        return OutputData._get_rgb(constructor(Color.Color()))

    @staticmethod
    def _get_rgb(color):
        """
        returns the r, g, and b values of a Color, given the Color object.

        :param color: The Color object.
        """
        return color.R(), color.G(), color.B()


class Environments(OutputData):
    def get_data(self):
        return Envs.Environments.GetRootAsEnvironments(self.bytes, 0)

    def get_center(self, index):
        return OutputData._get_vector3(self.data.Envs(index).Center)

    def get_bounds(self, index):
        return OutputData._get_vector3(self.data.Envs(index).Bounds)

    def get_id(self, index):
        return self.data.Envs(index).Id()

    def get_num(self):
        return self.data.EnvsLength()


class Transforms(OutputData):
    def get_data(self):
        return Trans.Transforms.GetRootAsTransforms(self.bytes, 0)

    def get_num(self):
        return self.data.ObjectsLength()

    def get_id(self, index):
        return self.data.Objects(index).Id()

    def get_position(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Position)

    def get_forward(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Forward)

    def get_rotation(self, index):
        return OutputData._get_quaternion(self.data.Objects(index).Rotation)


class Rigidbodies(OutputData):
    def get_data(self):
        return Rigis.Rigidbodies.GetRootAsRigidbodies(self.bytes, 0)

    def get_num(self):
        return self.data.ObjectsLength()

    def get_id(self, index):
        return self.data.Objects(index).Id()

    def get_velocity(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Velocity)

    def get_angular_velocity(self, index):
        return OutputData._get_vector3(self.data.Objects(index).AngularVelocity)

    def get_mass(self, index):
        return self.data.Objects(index).Mass()

    def get_sleeping(self, index):
        return self.data.Objects(index).Sleeping()


class Bounds(OutputData):
    def get_data(self):
        return Bouns.Bounds.GetRootAsBounds(self.bytes, 0)

    def get_num(self):
        return self.data.ObjectsLength()

    def get_id(self, index):
        return self.data.Objects(index).Id()

    def get_front(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Front)

    def get_back(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Back)

    def get_left(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Left)

    def get_right(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Right)

    def get_top(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Top)

    def get_bottom(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Bottom)

    def get_center(self, index):
        return OutputData._get_vector3(self.data.Objects(index).Center)


class Images(OutputData):
    PASS_MASKS = {PassMask.PassMask._img: "_img",
                  PassMask.PassMask._id: "_id",
                  PassMask.PassMask._category: "_category",
                  PassMask.PassMask._mask: "_mask",
                  PassMask.PassMask._depth: "_depth",
                  PassMask.PassMask._normals: "_normals",
                  PassMask.PassMask._flow: "_flow"
                  }

    def get_data(self):
        return Imags.Images.GetRootAsImages(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_sensor_name(self):
        return self.data.SensorName().decode('utf-8')

    def get_num_passes(self):
        return self.data.PassesLength()

    def get_pass_mask(self, index):
        return Images.PASS_MASKS[self.data.Passes(index).PassMask()]

    def get_image(self, index):
        return self.data.Passes(index).ImageAsNumpy()

    def get_extension(self, index):
        return "png" if self.data.Passes(index).Extension() == 1 else "jpg"

    def get_env_id(self):
        return self.data.EnvId()


class AvatarKinematic(OutputData):
    def get_data(self):
        return AvKi.AvatarKinematic.GetRootAsAvatarKinematic(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.Id().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_position(self):
        return OutputData._get_xyz(self.data.Position())

    def get_rotation(self):
        return OutputData._get_xyzw(self.data.Rotation())

    def get_forward(self):
        return OutputData._get_xyz(self.data.Forward())


class AvatarNonKinematic(AvatarKinematic):
    def get_data(self):
        return AvNoKi.AvatarNonKinematic.GetRootAsAvatarNonKinematic(self.bytes, 0)

    def get_velocity(self):
        return OutputData._get_xyz(self.data.Velocity())

    def get_angular_velocity(self):
        return OutputData._get_xyz(self.data.AngularVelocity())

    def get_mass(self):
        return self.data.Mass()

    def get_sleeping(self):
        return self.data.Sleeping()


class AvatarSimpleBody(AvatarNonKinematic):
    def get_data(self):
        return AvSi.AvatarSimpleBody.GetRootAsAvatarSimpleBody(self.bytes, 0)

    def get_visible_body(self):
        return self.data.VisibleBody().decode('utf-8')


class AvatarStickyMitten(AvatarNonKinematic):
    def get_data(self):
        return AvSM.AvatarStickyMitten.GetRootAsAvatarStickyMitten(self.bytes, 0)

    def get_num_body_parts(self):
        return self.data.BodyPartsLength()

    def get_num_rigidbody_parts(self):
        return self.data.RigidbodyPartsLength()

    def get_body_part_position(self, index):
        return OutputData._get_vector3(self.data.BodyParts(index).Position)

    def get_body_part_rotation(self, index):
        return OutputData._get_quaternion(self.data.BodyParts(index).Rotation)

    def get_body_part_forward(self, index):
        return OutputData._get_vector3(self.data.BodyParts(index).Forward)

    def get_body_part_id(self, index):
        return self.data.BodyParts(index).Id()

    def get_rigidbody_part_velocity(self, index):
        return OutputData._get_vector3(self.data.RigidbodyParts(index).Velocity)

    def get_rigidbody_part_angular_velocity(self, index):
        return OutputData._get_vector3(self.data.RigidbodyParts(index).AngularVelocity)

    def get_rigidbody_part_mass(self, index):
        return self.data.RigidbodyParts(index).Mass()

    def get_rigidbody_part_sleeping(self, index):
        return self.data.RigidbodyParts(index).Sleeping()

    def get_rigidbody_part_id(self, index):
        return self.data.RigidbodyParts(index).Id()


class SegmentationColors(OutputData):
    def get_data(self):
        return Segs.SegmentationColors.GetRootAsSegmentationColors(self.bytes, 0)

    def get_num(self):
        return self.data.ObjectsLength()

    def get_object_id(self, index):
        return self.data.Objects(index).Id()

    def get_object_color(self, index):
        return OutputData._get_rgb(self.data.Objects(index).SegmentationColor())

    def get_object_name(self, index):
        return self.data.Objects(index).Name().decode('utf-8')


class AvatarSegmentationColor(OutputData):
    def get_data(self):
        return AvSC.AvatarSegmentationColor.GetRootAsAvatarSegmentationColor(self.bytes, 0)

    def get_id(self):
        return self.data.Id().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_segmentation_color(self):
        return OutputData._get_rgb(self.data.SegmentationColor())


class AvatarStickyMittenSegmentationColors(OutputData):
    def get_data(self):
        return AvSMSC.AvatarStickyMittenSegmentationColors.GetRootAsAvatarStickyMittenSegmentationColors(self.bytes, 0)

    def get_id(self):
        return self.data.Id().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_num_body_parts(self):
        return self.data.BodyPartsLength()

    def get_body_part_id(self, index):
        return self.data.BodyParts(index).Id()

    def get_body_part_segmentation_color(self, index):
        return OutputData._get_rgb(self.data.BodyParts(index).SegmentationColor())

    def get_body_part_name(self, index):
        return self.data.BodyParts(index).Name().decode('utf-8')


class IsOnNavMesh(OutputData):
    def get_data(self):
        return IsNM.IsOnNavMesh.GetRootAsIsOnNavMesh(self.bytes, 0)

    def get_position(self):
        return OutputData._get_xyz(self.data.Position())

    def get_is_on(self):
        return self.data.IsOn()


class IdPassGrayscale(OutputData):
    def get_data(self):
        return IdGS.IdPassGrayscale.GetRootAsIdPassGrayscale(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_sensor_name(self):
        return self.data.SensorName().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_grayscale(self):
        return self.data.Grayscale()


class Collision(OutputData):
    def get_data(self):
        return Col.Collision.GetRootAsCollision(self.bytes, 0)

    def get_collider_id(self):
        return self.data.ColliderId()

    def get_collidee_id(self):
        return self.data.CollideeId()

    def get_relative_velocity(self):
        return OutputData._get_xyz(self.data.RelativeVelocity())

    def get_state(self):
        state = self.data.State()
        if state == 1:
            return "enter"
        elif state == 2:
            return "stay"
        else:
            return "exit"

    def get_num_contacts(self):
        return self.data.ContactsLength()

    def get_contact_normal(self, index):
        return OutputData._get_vector3(self.data.Contacts(index).Normal)

    def get_contact_point(self, index):
        return OutputData._get_vector3(self.data.Contacts(index).Point)


class ImageSensors(OutputData):
    def get_data(self):
        return ImSe.ImageSensors.GetRootAsImageSensors(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_num_sensors(self):
        return self.data.SensorsLength()

    def get_sensor_name(self, index):
        return self.data.Sensors(index).Name().decode('utf-8')

    def get_sensor_on(self, index):
        return self.data.Sensors(index).IsOn()

    def get_sensor_rotation(self, index):
        return OutputData._get_xyzw(self.data.Sensors(index).Rotation())


class CameraMatrices(OutputData):
    def get_data(self):
        return CaMa.CameraMatrices.GetRootAsCameraMatrices(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_sensor_name(self):
        return self.data.SensorName().decode('utf-8')

    def get_projection_matrix(self):
        return self.data.ProjectionMatrixAsNumpy()

    def get_camera_matrix(self):
        return self.data.CameraMatrixAsNumpy()


class IdPassSegmentationColors(OutputData):
    def get_data(self):
        return IdSC.IdPassSegmentationColors.GetRootAsIdPassSegmentationColors(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_sensor_name(self):
        return self.data.SensorName().decode('utf-8')

    def get_num_segmentation_colors(self):
        return self.data.SegmentationColorsLength()

    def get_segmentation_color(self, index):
        return OutputData._get_rgb(self.data.SegmentationColors(index))


class ArrivedAtNavMeshDestination(OutputData):
    def get_data(self):
        return Arri.ArrivedAtNavMeshDestination.GetRootAsArrivedAtNavMeshDestination(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()


class FlexParticles(OutputData):
    def get_data(self):
        return Flex.FlexParticles.GetRootAsFlexParticles(self.bytes, 0)

    def get_num_objects(self):
        return self.data.ObjectsLength()

    def get_particles(self, index):
        particles = []
        for i in range(self.data.Objects(index).ParticlesLength()):
            particles.append(OutputData._get_xyzw(self.data.Objects(index).Particles(i)))
        return particles

    def get_velocities(self, index):
        velocities = []
        for i in range(self.data.Objects(index).VelocitiesLength()):
            velocities.append(OutputData._get_xyz(self.data.Objects(index).Velocities(i)))
        return velocities

    def get_id(self, index):
        return self.data.Objects(index).Id()


class Video(OutputData):
    def get_data(self):
        return Vid.Video.GetRootAsVideo(self.bytes, 0)

    def get_avatar_id(self):
        return self.data.AvatarId().decode('utf-8')

    def get_env_id(self):
        return self.data.EnvId()

    def get_video(self):
        return self.data.VideoDataAsNumpy()
