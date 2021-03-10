from dataclasses import dataclass, field
from typing import List, Dict

from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist

from eth2spec.phase0 import spec


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


def encode_merkleable_dict(thedict: dict) -> dict:
    result = dict()
    for key in thedict.keys():
        result[key.encode_bytes()] = thedict[key].encode_bytes()
    return result


def decode_merkleable_dict(thedict: dict, key_class, value_class) -> dict:
    result = dict()
    for key in thedict.keys():
        result[key_class.decode_bytes(key)] = value_class.decode_bytes(thedict[key])
    return result


@dataclass(eq=True, frozen=True)
class PickleableLatestMessage(object):
    epoch: bytes
    root: bytes

    @classmethod
    def from_latest_message(cls, latest_message: spec.LatestMessage):
        return cls(
            epoch=latest_message.epoch.encode_bytes(),
            root=bytes(latest_message.root)
        )

    def into_latest_message(self):
        return spec.LatestMessage(
            epoch=spec.Epoch.decode_bytes(self.epoch),
            root=spec.Root.decode_bytes(self.root)
        )


@dataclass
class PickleableStore(object):
    time: bytes
    genesis_time: bytes
    justified_checkpoint: bytes
    finalized_checkpoint: bytes
    best_justified_checkpoint: bytes
    blocks: Dict[bytes, bytes] = field(default_factory=dict)
    block_states: Dict[bytes, bytes] = field(default_factory=dict)
    checkpoint_states: Dict[bytes, bytes] = field(default_factory=dict)
    latest_messages: Dict[bytes, PickleableLatestMessage] = field(default_factory=dict)

    @classmethod
    def from_store(cls, store: spec.Store):
        latest_messages = dict()
        for key in store.latest_messages.keys():
            latest_messages[key.encode_bytes()] = PickleableLatestMessage.from_latest_message(store.latest_messages[key])
        return cls(
            time=store.time.encode_bytes(),
            genesis_time=store.genesis_time.encode_bytes(),
            justified_checkpoint=store.justified_checkpoint.encode_bytes(),
            finalized_checkpoint=store.finalized_checkpoint.encode_bytes(),
            best_justified_checkpoint=store.best_justified_checkpoint.encode_bytes(),
            blocks=encode_merkleable_dict(store.blocks),
            block_states=encode_merkleable_dict(store.block_states),
            checkpoint_states=encode_merkleable_dict(store.checkpoint_states),
            latest_messages=latest_messages
        )

    def into_store(self) -> spec.Store:
        latest_messages = dict()
        for key in self.latest_messages.keys():
            latest_messages[spec.ValidatorIndex.decode_bytes(key)] = self.latest_messages[key].into_latest_message()
        return spec.Store(
            time=uint64.decode_bytes(self.time),
            genesis_time=uint64.decode_bytes(self.genesis_time),
            justified_checkpoint=spec.Checkpoint.decode_bytes(self.justified_checkpoint),
            finalized_checkpoint=spec.Checkpoint.decode_bytes(self.finalized_checkpoint),
            best_justified_checkpoint=spec.Checkpoint.decode_bytes(self.best_justified_checkpoint),
            blocks=decode_merkleable_dict(self.blocks, spec.Root, spec.BeaconBlock),
            block_states=decode_merkleable_dict(self.block_states, spec.Root, spec.BeaconState),
            checkpoint_states=decode_merkleable_dict(self.checkpoint_states, spec.Checkpoint, spec.BeaconState),
            latest_messages=latest_messages
        )
