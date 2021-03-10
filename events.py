from dataclasses import dataclass, field
from typing import Union

from remerkleable.basic import uint64

from eth2spec.phase0 import spec


MESSAGE_TYPE = Union[spec.Attestation, spec.SignedAggregateAndProof, spec.SignedBeaconBlock]
GLOBAL_COUNTER = [uint64(0), ]


@dataclass
class Event:
    time: uint64
    counter: uint64 = field(init=False)

    def __post_init__(self):
        GLOBAL_COUNTER[0] = GLOBAL_COUNTER[0] + 1
        self.counter = GLOBAL_COUNTER[0].copy()

    def __lt__(self, other):
        return self.counter < other.counter if self.time == other.time else self.time < other.time


@dataclass
class NextSlotEvent(Event):
    slot: spec.Slot


@dataclass
class LatestVoteOpportunity(Event):
    slot: spec.Slot


@dataclass
class AggregateOpportunity(Event):
    slot: spec.Slot


@dataclass
class MessageEvent(Event):
    message: MESSAGE_TYPE
    fromidx: spec.ValidatorIndex
    toidx: spec.ValidatorIndex


@dataclass
class SimulationEndEvent(Event):
    pass
