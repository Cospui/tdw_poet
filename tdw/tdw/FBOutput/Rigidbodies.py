# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import tdw.flatbuffers

class Rigidbodies(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRigidbodies(cls, buf, offset):
        n = tdw.flatbuffers.encode.Get(tdw.flatbuffers.packer.uoffset, buf, offset)
        x = Rigidbodies()
        x.Init(buf, n + offset)
        return x

    # Rigidbodies
    def Init(self, buf, pos):
        self._tab = tdw.flatbuffers.table.Table(buf, pos)

    # Rigidbodies
    def Objects(self, j):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += tdw.flatbuffers.number_types.UOffsetTFlags.py_type(j) * 36
            from .RigidbodyData import RigidbodyData
            obj = RigidbodyData()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Rigidbodies
    def ObjectsLength(self):
        o = tdw.flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def RigidbodiesStart(builder): builder.StartObject(1)
def RigidbodiesAddObjects(builder, objects): builder.PrependUOffsetTRelativeSlot(0, tdw.flatbuffers.number_types.UOffsetTFlags.py_type(objects), 0)
def RigidbodiesStartObjectsVector(builder, numElems): return builder.StartVector(36, numElems, 4)
def RigidbodiesEnd(builder): return builder.EndObject()
