from typing import List

from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist


def popcnt(bitlist: Bitlist) -> uint64:
    bits = uint64(0)
    for bit in bitlist:
        if bit:
            bits += 1
    return bits


def indices_inside_committee(bitlist: Bitlist) -> List[uint64]:
    indexes = list()
    for i, bit in enumerate(bitlist):
        if bit:
            indexes.append(uint64(i))
    return indexes
