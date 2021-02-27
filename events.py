from dataclasses import dataclass
from typing import Any, Optional, Union

from remerkleable.basic import uint64

from eth2spec.phase0 import spec


MESSAGE_TYPE = Union[spec.Attestation, spec.SignedAggregateAndProof, spec.SignedBeaconBlock]


@dataclass
class Event:
    time: uint64

    def __lt__(self, other):
        return id(self.time) < id(other.time) if self.time == other.time else self.time < other.time


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
    fromidx: int
    toidx: Optional[int]


@dataclass
class SimulationEndEvent(Event):
    pass
