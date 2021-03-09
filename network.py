from typing import Optional, Any, Sequence

import numpy as np

from remerkleable.basic import uint64

from eth2spec.phase0 import spec
from events import MessageEvent, MESSAGE_TYPE


class Network(object):
    def __init__(self, simulator, rand):
        self.simulator = simulator
        self.random = np.random.RandomState(seed=rand)

    def latency(self) -> uint64:
        # return uint64(int(max(0, self.random.normal(1, 1))))
        return uint64(0)

    def send(self, fromidx: spec.ValidatorIndex, toidx: Optional[Sequence[spec.ValidatorIndex]], message: MESSAGE_TYPE):
        time = self.simulator.simulator_time + self.latency()
        self.simulator.events.put(MessageEvent(time, message, fromidx, toidx))
