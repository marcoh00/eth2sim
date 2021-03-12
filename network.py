import numpy as np

from remerkleable.basic import uint64

from eth2spec.phase0 import spec
from events import MessageEvent, MESSAGE_TYPE

USE_RANDOM_LATENCY = True


class Network(object):
    def __init__(self, simulator, rand):
        self.simulator = simulator
        self.random = np.random.RandomState(seed=rand)

    def __latency(self) -> uint64:
        random_latency = uint64(int(max(0, self.random.normal(1, 1))))
        return random_latency if USE_RANDOM_LATENCY else uint64(0)

    # noinspection PyUnusedLocal
    def latency(self, fromidx: spec.ValidatorIndex, toidx: spec.ValidatorIndex):
        return self.__latency()

    def send(self, fromidx: spec.ValidatorIndex, toidx: spec.ValidatorIndex, message: MESSAGE_TYPE):
        time = self.simulator.simulator_time + self.latency(fromidx, toidx)
        self.simulator.events.put(MessageEvent(time, message, fromidx, toidx))

    def delay(self, message: MessageEvent):
        message.time = message.time + self.latency(message.fromidx, message.toidx)
