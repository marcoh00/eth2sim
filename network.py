from typing import Optional, Any

import numpy as np

from remerkleable.basic import uint64

from events import MessageEvent, MESSAGE_TYPE


class Network(object):
    def __init__(self, simulator, rand):
        self.simulator = simulator
        self.random = np.random.RandomState(seed=rand)

    def latency(self) -> uint64:
        return uint64(int(max(0, self.random.normal(1, 1))))

    def send(self, message: MESSAGE_TYPE, fromidx: int, toidx: Optional[int]):
        time = self.simulator.simulator_time + self.latency()
        self.simulator.events.put(MessageEvent(time, message, fromidx, toidx))
