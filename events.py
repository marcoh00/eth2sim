from dataclasses import dataclass, field
from typing import Union, Optional

from remerkleable.basic import uint64

from eth2spec.phase0 import spec
from random import randint


MESSAGE_TYPE = Union[spec.Attestation, spec.SignedAggregateAndProof, spec.SignedBeaconBlock]

"""
For keeping determinism, every Event has gained a field `priority`.
Certain messages do need to have certain priorities:

Messages which directly lead to the creation of other messages do need to have a higher priority than
those who don't:

NextSlot/AttestationOpportunity/AggregationOpportunity > Block > Attestation/AggregatedAttestation
"""


@dataclass
class Event:
    time: uint64
    priority: int

    def __lt__(self, other):
        return self.priority < other.priority if self.time == other.time else self.time < other.time


@dataclass
class ValidatorInitializationEvent(Event):
    pass


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
    custom_latency: Optional[int] = field(default=None)
    delayed: Optional[int] = field(default=None)
    marker: Optional[int] = field(default_factory=lambda: randint(0, 4294967296))


@dataclass
class RequestDeposit(Event):
    stake: int


@dataclass
class SimulationEndEvent(Event):
    message: Optional[str] = None
