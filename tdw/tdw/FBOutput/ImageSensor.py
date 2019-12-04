# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import tdw.flatbuffers

class ImageSensor(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsImageSensor(cls, buf, offset):
        n = tdw.flatbuffers.encode.Get(tdw.flatbuffers.packer.uoffset, buf, offset)
        x = ImageSensor()
        x.Init(buf, n + offset)
        return x

    # ImageSensor
    def Init(self, buf, pos):
        self._tab = tdw.flatbuffers.table.Table(buf, pos)

    # ImageSensor
    def Name(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ImageSensor
    def IsOn(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(tdw.flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # ImageSensor
    def Rotation(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = o + self._tab.Pos
            from .Quaternion import Quaternion
            obj = Quaternion()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def ImageSensorStart(builder): builder.StartObject(3)
def ImageSensorAddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, tdw.flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def ImageSensorAddIsOn(builder, isOn): builder.PrependBoolSlot(1, isOn, 0)
def ImageSensorAddRotation(builder, rotation): builder.PrependStructSlot(2, tdw.flatbuffers.number_types.UOffsetTFlags.py_type(rotation), 0)
def ImageSensorEnd(builder): return builder.EndObject()
