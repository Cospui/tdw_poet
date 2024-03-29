# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import tdw.flatbuffers

class Vector4(object):
    __slots__ = ['_tab']

    # Vector4
    def Init(self, buf, pos):
        self._tab = tdw.flatbuffers.table.Table(buf, pos)

    # Vector4
    def X(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(0))
    # Vector4
    def Y(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(4))
    # Vector4
    def Z(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(8))
    # Vector4
    def W(self): return self._tab.Get(tdw.flatbuffers.number_types.Float32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(12))

def CreateVector4(builder, x, y, z, w):
    builder.Prep(4, 16)
    builder.PrependFloat32(w)
    builder.PrependFloat32(z)
    builder.PrependFloat32(y)
    builder.PrependFloat32(x)
    return builder.Offset()
