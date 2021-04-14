from dataclasses import dataclass, field
from typing import Union, Optional

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
class ValidatorInitializationEvent(Event):
    empty_list: bool
    startidx: Optional[int]
    endidx: Optional[int]
    keydir: Optional[str]


@dataclass
class NextSlotEvent(Event):
    slot: int


@dataclass
class LatestVoteOpportunity(Event):
    slot: int


@dataclass
class AggregateOpportunity(Event):
    slot: int


@dataclass
class TargetedEvent(Event):
    toidx: Optional[int]


@dataclass
class ProduceStatisticsEvent(TargetedEvent):
    pass


@dataclass
class ProduceGraphEvent(TargetedEvent):
    show: bool


@dataclass
class MessageEvent(Event):
    message: bytes
    message_type: str
    fromidx: int
    toidx: Optional[int]


@dataclass
class RequestDeposit(Event):
    stake: int


@dataclass
class SimulationEndEvent(Event):
    message: Optional[str] = None
