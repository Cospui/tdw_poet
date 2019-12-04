# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import tdw.flatbuffers

class Quaternion(object):
    __slots__ = ['_tab']

    # Quaternion
    def Init(self, buf, pos):
        self._tab = tdw.flatbuffers.table.Table(buf, pos)

    # Quaternion
    def X(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(0))
    # Quaternion
    def Y(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(4))
    # Quaternion
    def Z(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(8))
    # Quaternion
    def W(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(12))

def CreateQuaternion(builder, x, y, z, w):
    builder.Prep(4, 16)
    builder.PrependFloat32(w)
    builder.PrependFloat32(z)
    builder.PrependFloat32(y)
    builder.PrependFloat32(x)
    return builder.Offset()
