from multiprocessing import Queue, JoinableQueue
from typing import Optional, List

from remerkleable.basic import uint64
from remerkleable.byte_arrays import ByteVector

from beaconclient import BeaconClient
from eth2spec.phase0 import spec
from events import SimulationEndEvent
from simulator import Simulator, IndexedBeaconClient
from validator import Validator


class Builder:
    def __init__(self, parent_builder=None):
        self.parent_builder = parent_builder

    def build(self, callback=True, counter=0):
        if callback and self.parent_builder is not None:
            self.parent_builder.register(self)
            return self.parent_builder
        else:
            return self.build_impl(counter)

    def build_impl(self, counter):
        raise NotImplementedError

    def register(self, child_builder):
        raise NotImplementedError


class ValidatorBuilder(Builder):
    keydir: Optional[str]

    def __init__(self, parent_builder=None):
        super().__init__(parent_builder)
        self.keydir = "cryptokeys"
        self.startbalance = spec.MAX_EFFECTIVE_BALANCE

    def build_impl(self, counter):
        return Validator(
            counter=counter,
            startbalance=self.startbalance,
            keydir=self.keydir
        )

    def register(self, child_builder):
        pass

    def set_keydir(self, keydir):
        self.keydir = keydir

    def set_startbalance(self, balance):
        self.startbalance = balance


class BeaconClientBuilder(Builder):
    configpath: str
    configname: str

    debug: bool
    debugfile: Optional[str]
    profile: bool

    validators_count: int
    validator_builders: List[ValidatorBuilder]

    neccessary_info_set: bool
    validator_start_at: int
    recv_queue: JoinableQueue
    send_queue: Queue

    def __init__(self, configpath, configname, parent_builder=None,):
        super(BeaconClientBuilder, self).__init__(parent_builder)
        self.validator_builders = []
        self.configpath = configpath
        self.configname = configname
        self.debug = False
        self.debugfile = None
        self.profile = False

        self.neccessary_info_set = False
        self.validator_start_at = 0
        self.simulator_to_client_queue = JoinableQueue()
        self.client_to_simulator_queue = Queue()

    def build_impl(self, counter):
        if not self.neccessary_info_set:
            raise ValueError('Need to specify queues and validator start index')
        validators = list()
        for validator_builder in self.validator_builders:
            validator = validator_builder.build(False, self.validator_start_at + len(validators))
            validators.append(validator)
        return BeaconClient(
            counter=counter,
            simulator_to_client_queue=self.simulator_to_client_queue,
            client_to_simulator_queue=self.client_to_simulator_queue,
            configpath=self.configpath,
            configname=self.configname,
            validators=validators,
            debug=self.debug
        )

    def register(self, child_builder):
        for _ in range(self.validators_count):
            self.validator_builders.append(child_builder)

    def set_debug(self, flag=False):
        self.debug = flag
        return self

    def set_profile(self, flag=False):
        self.profile = flag
        return self

    def validators(self, count):
        self.validators_count = count
        return ValidatorBuilder(parent_builder=self)

    def neccessary_information(self, validator_start_at, client_to_simulator_queue):
        self.neccessary_info_set = True
        self.validator_start_at = validator_start_at
        self.client_to_simulator_queue = client_to_simulator_queue


class SimulationBuilder(Builder):
    configpath: str
    configname: str
    rand: ByteVector
    current_child_count: int
    end_time: uint64

    beacon_client_builders: List[BeaconClientBuilder]

    def __init__(self, configpath, configname, rand, parent_builder=None):
        super(SimulationBuilder, self).__init__(parent_builder)
        self.beacon_client_builders = []
        self.configpath = configpath
        self.configname = configname
        self.rand = rand
        self.end_time = uint64((8 * spec.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH) + 24)

    def beacon_client(self, count):
        self.current_child_count = count
        return BeaconClientBuilder(self.configpath, self.configname, parent_builder=self)

    def build_impl(self, counter):
        client_to_simulator_queue = Queue()
        # Build client set
        client_counter = 0
        validator_counter = 0
        clients = list()
        for client_builder in self.beacon_client_builders:
            simulator_to_client_queue = JoinableQueue()
            client_builder.neccessary_information(validator_counter, client_to_simulator_queue)
            client = client_builder.build(callback=False, counter=client_counter)
            indexed_client = IndexedBeaconClient(
                queue=simulator_to_client_queue,
                beacon_client=client
            )
            indexed_client.beacon_client.simulator_to_client_queue = simulator_to_client_queue
            clients.append(indexed_client)
            client_counter += 1
            validator_counter += len(indexed_client.beacon_client.validators)
        simulator = Simulator(self.rand)
        simulator.queue = client_to_simulator_queue
        simulator.clients = clients
        simulator.events.put(SimulationEndEvent(simulator.genesis_time + self.end_time))
        return simulator

    def set_end_time(self, end_time):
        self.end_time = end_time

    def register(self, child_builder):
        for _ in range(self.current_child_count):
            self.beacon_client_builders.append(child_builder)


"""
SimulationBuilder(configpath, configname)
    .beacon_client(4)
        .debug(True)
        .profile(True)
        .validators(7)
            .keydir(keydir)
            .build()
        .build()
    .build()
"""