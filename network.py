from typing import Tuple, Optional, Union, Dict

from remerkleable.basic import uint64

import hashlib
from events import MessageEvent, MESSAGE_TYPE

USE_RANDOM_LATENCY = True
LATENCY_MAP_BLOCK = ((0, 0), (26, 1), (102, 2), (230, 3))
LATENCY_MAP_AGG_ATTESTATION = (
    (0, 0),
    (51, 1),
    (128, 2),
    (179, 5),
    (204, 7),
    (230, 10),
    (242, 11),
)
LATENCY_MAP_ATTESTATION = (
    (0, 0),
    (128, 1),
    (179, 2),
    (230, 4)
)
LATENCY_MAP_GT = LATENCY_MAP_AGG_ATTESTATION


class Network(object):
    def __init__(
        self,
        simulator,
        rand,
        custom_latency_map: Optional[
            Union[Tuple[Tuple[int]], Dict[str, Tuple[Tuple[int]]]]
        ] = None,
        latency_modifier=None,
    ):
        self.simulator = simulator
        self.random = rand
        if type(custom_latency_map) == dict:
            dictkeys = tuple(custom_latency_map.keys())
            if "" in custom_latency_map:
                self.default_latency_map = custom_latency_map[""]
                print(
                    f"[NETWORK] Using custom latency maps with a custom fallback for: {dictkeys}"
                )
            else:
                self.default_latency_map = LATENCY_MAP_GT
                print(
                    f"[NETWORK] Using custom latency maps with default fallback for: {dictkeys}"
                )
            self.additional_maps = custom_latency_map
        elif type(custom_latency_map) == tuple:
            self.default_latency_map = custom_latency_map
            self.additional_maps = dict()
            print("[NETWORK] Using a single custom latency map")
        else:
            self.default_latency_map = LATENCY_MAP_GT
            self.additional_maps = {
                "Attestation": LATENCY_MAP_ATTESTATION,
                "SignedAggregateAndProof": LATENCY_MAP_AGG_ATTESTATION,
                "SignedBeaconBlock": LATENCY_MAP_BLOCK,
            }
            print(f"[NETWORK] Using default latency maps")
        self.latency_modifier = latency_modifier
        if self.latency_modifier is None:
            self.latency_modifier = lambda x: x

    # noinspection PyUnusedLocal
    def latency(self, time: uint64, fromidx: int, toidx: int, msgtype=None):
        hashgen = hashlib.sha256(usedforsecurity=False)
        hashgen.update(
            f"{str(self.random)}-{str(time)}-{str(fromidx)}-{str(toidx)}".encode(
                "utf-8"
            )
        )
        # 1 byte = 0-255
        rndvalue = int(hashgen.digest()[0])
        latency = 0

        latency_map = (
            self.default_latency_map
            if msgtype is None or msgtype not in self.additional_maps
            else self.additional_maps[msgtype]
        )
        for maptuple in latency_map:
            if rndvalue > maptuple[0]:
                latency = self.latency_modifier(maptuple[1])
        if fromidx == toidx:
            latency = 0
        # print(f'LATENCY [{time},{fromidx},{toidx}] {latency} [[{latency_map}]]')
        return latency

    def delay(self, message: MessageEvent):
        if message.custom_latency is not None:
            latency = message.custom_latency
            print(
                f"Message from Client {message.fromidx} to Client {message.toidx} with custom latency {message.custom_latency} will be delivered on {message.time + latency}"
            )
        else:
            latency = self.latency(
                message.time, message.fromidx, message.toidx, message.message_type
            )
        message.time += latency
        message.delayed = latency
