
__all__ = ["encode_norm_lat", "encode_norm_long", "decode_norm_long", "decode_norm_lat"]

def encode_norm_lat(coord):
    return coord/100

def decode_norm_lat(coord):
    return coord*100

def encode_norm_long(coord):
    return coord/200

def decode_norm_long(coord):
    return coord*200