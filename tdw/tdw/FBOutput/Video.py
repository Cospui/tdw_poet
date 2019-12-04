# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import tdw.flatbuffers

class Video(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsVideo(cls, buf, offset):
        n = tdw.flatbuffers.encode.Get(tdw.flatbuffers.packer.uoffset, buf, offset)
        x = Video()
        x.Init(buf, n + offset)
        return x

    # Video
    def Init(self, buf, pos):
        self._tab = tdw.flatbuffers.table.Table(buf, pos)

    # Video
    def Id(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Video
    def EnvId(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(tdw.flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Video
    def VideoData(self, j):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(tdw.flatbuffers.number_types.Uint8Flags, a + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # Video
    def VideoDataAsNumpy(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(tdw.flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Video
    def VideoDataLength(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def VideoStart(builder): builder.StartObject(3)
def VideoAddId(builder, id): builder.PrependUOffsetTRelativeSlot(0, tdw.flatbuffers.number_types.UOffsetTFlags.py_type(id), 0)
def VideoAddEnvId(builder, envId): builder.PrependInt32Slot(1, envId, 0)
def VideoAddVideoData(builder, videoData): builder.PrependUOffsetTRelativeSlot(2, tdw.flatbuffers.number_types.UOffsetTFlags.py_type(videoData), 0)
def VideoStartVideoDataVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def VideoEnd(builder): return builder.EndObject()
