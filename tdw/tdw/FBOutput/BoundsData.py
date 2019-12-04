# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import tdw.flatbuffers

class BoundsData(object):
    __slots__ = ['_tab']

    # BoundsData
    def Init(self, buf, pos):
        self._tab = tdw.flatbuffers.table.Table(buf, pos)

    # BoundsData
    def Id(self): return self._tab.Get(tdw.flatbuffers.number_types.Int32Flags, self._tab.Pos + tdw.flatbuffers.number_types.UOffsetTFlags.py_type(0))
    # BoundsData
    def Front(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 4)
        return obj

    # BoundsData
    def Back(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 16)
        return obj

    # BoundsData
    def Right(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 28)
        return obj

    # BoundsData
    def Left(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 40)
        return obj

    # BoundsData
    def Top(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 52)
        return obj

    # BoundsData
    def Bottom(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 64)
        return obj

    # BoundsData
    def Center(self, obj):
        obj.Init(self._tab.Bytes, self._tab.Pos + 76)
        return obj


def CreateBoundsData(builder, id, front_x, front_y, front_z, back_x, back_y, back_z, right_x, right_y, right_z, left_x, left_y, left_z, top_x, top_y, top_z, bottom_x, bottom_y, bottom_z, center_x, center_y, center_z):
    builder.Prep(4, 88)
    builder.Prep(4, 12)
    builder.PrependFloat32(center_z)
    builder.PrependFloat32(center_y)
    builder.PrependFloat32(center_x)
    builder.Prep(4, 12)
    builder.PrependFloat32(bottom_z)
    builder.PrependFloat32(bottom_y)
    builder.PrependFloat32(bottom_x)
    builder.Prep(4, 12)
    builder.PrependFloat32(top_z)
    builder.PrependFloat32(top_y)
    builder.PrependFloat32(top_x)
    builder.Prep(4, 12)
    builder.PrependFloat32(left_z)
    builder.PrependFloat32(left_y)
    builder.PrependFloat32(left_x)
    builder.Prep(4, 12)
    builder.PrependFloat32(right_z)
    builder.PrependFloat32(right_y)
    builder.PrependFloat32(right_x)
    builder.Prep(4, 12)
    builder.PrependFloat32(back_z)
    builder.PrependFloat32(back_y)
    builder.PrependFloat32(back_x)
    builder.Prep(4, 12)
    builder.PrependFloat32(front_z)
    builder.PrependFloat32(front_y)
    builder.PrependFloat32(front_x)
    builder.PrependInt32(id)
    return builder.Offset()
