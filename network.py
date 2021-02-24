from typing import Optional, Any

import numpy as np

from remerkleable.basic import uint64

from events import MessageEvent, MESSAGE_TYPE


class Network(object):
    def __init__(self, simulator, rand):
        self.simulator = simulator
        self.random = np.random.RandomState(seed=rand)

    def latency(self):
        return int(max(0, np.random.normal(1, 1)))

    def send(self, message: MESSAGE_TYPE, fromidx: int, toidx: Optional[int]):
        simulator_time = self.simulator.normalized_simulator_time()
        latency = self.latency()
        latency_u64 = uint64(latency)
        time = simulator_time + latency_u64
        self.simulator.events.put((time, MessageEvent(simulator_time, message, fromidx, toidx)))
