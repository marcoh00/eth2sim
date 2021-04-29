from remerkleable.basic import uint64

import hashlib
from events import MessageEvent, MESSAGE_TYPE

USE_RANDOM_LATENCY = True
LATENCY_MAP_GT = ((0, 1), (200, 2), (230, 3), (245, 4), (250, 5), (253, 8))


class Network(object):
    def __init__(self, simulator, rand):
        self.simulator = simulator
        self.random = rand
        #self.random = np.random.RandomState(seed=rand)

    # noinspection PyUnusedLocal
    def latency(self, time: uint64, fromidx: int, toidx: int):
        hashgen = hashlib.sha256(usedforsecurity=False)
        hashgen.update(f"{str(self.random)}-{str(time)}-{str(fromidx)}-{str(toidx)}".encode('utf-8'))
        # 1 byte = 0-255
        rndvalue = int(hashgen.digest()[0])
        latency = 0
        for maptuple in LATENCY_MAP_GT:
            if rndvalue > maptuple[0]:
                latency = maptuple[1]
        if fromidx == toidx:
            latency = 0
        # print(f'LATENCY [{time},{fromidx},{toidx}] {latency}')
        return latency

    def delay(self, message: MessageEvent):
        message.time = message.time + self.latency(message.time, message.fromidx, message.toidx)
