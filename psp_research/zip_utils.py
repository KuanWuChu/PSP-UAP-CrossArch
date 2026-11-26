"""ZIP64 local-header size decoding (APPNOTE extra-field 0x0001)."""
import struct


def local_sizes(compressed_size, uncompressed_size, extra):
    if compressed_size != 0xffffffff and uncompressed_size != 0xffffffff:
        return compressed_size, uncompressed_size
    position, zip64 = 0, None
    while position < len(extra):
        if len(extra) - position < 4:
            raise ValueError('truncated ZIP extra field')
        kind, length = struct.unpack_from('<HH', extra, position)
        position += 4
        if position + length > len(extra):
            raise ValueError('truncated ZIP extra payload')
        if kind == 1:
            if zip64 is not None:
                raise ValueError('duplicate ZIP64 size field')
            zip64 = extra[position:position+length]
        position += length
    required = 8 * ((compressed_size == 0xffffffff) + (uncompressed_size == 0xffffffff))
    if zip64 is None or len(zip64) < required:
        raise ValueError('missing ZIP64 sizes')
    position = 0
    if uncompressed_size == 0xffffffff:
        uncompressed_size = struct.unpack_from('<Q', zip64, position)[0]
        position += 8
    if compressed_size == 0xffffffff:
        compressed_size = struct.unpack_from('<Q', zip64, position)[0]
    return compressed_size, uncompressed_size
