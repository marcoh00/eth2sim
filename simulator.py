import datetime
import queue
from dataclasses import dataclass
from multiprocessing import Queue, JoinableQueue
from pathlib import Path
from typing import Optional, List, Sequence, Iterable, Union, Tuple, Dict, Callable

from remerkleable.basic import uint64
from remerkleable.byte_arrays import ByteVector

import eth2spec.phase0.spec as spec
from beaconclient import BeaconClient, BeaconClientBuilder
from builder import Builder
from eth2spec.phase0 import spec
from eth2spec.test.helpers.deposits import build_deposit, build_deposit_data
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, SimulationEndEvent, MessageEvent, \
    Event, MESSAGE_TYPE, ProduceGraphEvent, ProduceStatisticsEvent, TargetedEvent, ValidatorInitializationEvent
from helpers import initialize_beacon_state_from_mocked_eth1, queue_element_or_none
from network import Network


@dataclass
class IndexedBeaconClient(object):
    queue: JoinableQueue
    beacon_client: BeaconClient


class Simulator:

    genesis_time: uint64
    genesis_state: Optional[spec.BeaconState]
    genesis_block: Optional[spec.BeaconBlock]
    simulator_time: uint64
    simulator_prio: int
    slot: spec.Slot
    clients: List[IndexedBeaconClient]
    network: Network
    events: queue.Queue[Event]
    past_events: List[Event]
    random: ByteVector
    queue: Queue
    should_quit: bool

    def __init__(self,
                 rand: ByteVector,
                 custom_latency_map: Optional[Union[Tuple[Tuple[int]], Dict[str, Tuple[Tuple[int]]]]] = None,
                 latency_modifier: Optional[Callable[[int], int]] = None):
        self.genesis_time = spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY
        self.simulator_time = self.genesis_time.copy()
        self.simulator_prio = 2**64 - 1
        self.slot = spec.Slot(0)
        self.clients = []
        self.network = Network(self, int.from_bytes(rand[0:4], "little"), custom_latency_map, latency_modifier)
        self.events = queue.PriorityQueue()
        self.past_events = list()
        self.random = rand
        self.queue = Queue()
        self.should_quit = False

    def normalized_simulator_time(self):
        return self.simulator_time - self.genesis_time

    def next_slot_event(self):
        self.events.put(
            NextSlotEvent(
                self.genesis_time + (self.slot * spec.SECONDS_PER_SLOT) + spec.SECONDS_PER_SLOT,
                0,
                self.slot + 1)
        )

    def next_latest_vote_opportunity(self):
        self.events.put(
             LatestVoteOpportunity(
                 self.genesis_time + (self.slot * spec.SECONDS_PER_SLOT) + (spec.SECONDS_PER_SLOT // 3),
                 0,
                 self.slot)
        )

    def next_aggregate_opportunity(self):
        self.events.put(
             AggregateOpportunity(
                 self.genesis_time + (self.slot * spec.SECONDS_PER_SLOT) + (2 * (spec.SECONDS_PER_SLOT // 3)),
                 0,
                 self.slot)
        )

    def read_genesis(self, filename=''):
        print('[SIMULATOR] Read Genesis State and Block from file')
        state_file = Path(f"state_{filename}.ssz")
        block_file = Path(f"block_{filename}.ssz")
        with open(state_file, 'rb') as fp:
            encoded_state = fp.read()
            self.genesis_state = spec.BeaconState.decode_bytes(encoded_state)
        with open(block_file, 'rb') as fp:
            encoded_block = fp.read()
            self.genesis_block = spec.BeaconBlock.decode_bytes(encoded_block)

    def generate_genesis(self, filename=None, mocked=False):
        deposit_data = []
        deposits = []

        print(f'[SIMULATOR] Generate Genesis state ({sum(len(client.beacon_client.validators) for client in self.clients)} validators)')
        validatorno = 0
        start_time = datetime.datetime.now()
        eth1_timestamp = spec.MIN_GENESIS_TIME
        for client in self.clients:
            for validator in client.beacon_client.validators:
                if validatorno % 1000 == 0:
                    print(f"[SIMULATOR][Deposit] Create deposit #{validatorno}")
                if mocked:
                    deposit_data = build_deposit_data(
                        spec=spec,
                        pubkey=validator.pubkey,
                        privkey=validator.privkey,
                        amount=validator.startbalance,
                        withdrawal_credentials=spec.BLS_WITHDRAWAL_PREFIX + spec.hash(validator.pubkey)[1:],
                        signed=True
                    )
                    deposit = spec.Deposit(
                        proof=list(bytes.fromhex('0000000000000000000000000000000000000000000000000000000000000000') for _ in range(0, spec.DEPOSIT_CONTRACT_TREE_DEPTH + 1)),
                        data=deposit_data
                    )
                else:
                    deposit, root, deposit_data = build_deposit(
                        spec=spec,
                        deposit_data_list=deposit_data,
                        pubkey=validator.pubkey,
                        privkey=validator.privkey,
                        amount=validator.startbalance,
                        withdrawal_credentials=spec.BLS_WITHDRAWAL_PREFIX + spec.hash(validator.pubkey)[1:],
                        signed=True
                )
                deposits.append(deposit)
                validatorno += 1

        if mocked:
            self.genesis_state = initialize_beacon_state_from_mocked_eth1(spec, self.random, eth1_timestamp, deposits)
        else:
            self.genesis_state = spec.initialize_beacon_state_from_eth1(self.random, eth1_timestamp, deposits)
        assert spec.is_valid_genesis_state(self.genesis_state)
        self.genesis_block = spec.BeaconBlock(state_root=spec.hash_tree_root(self.genesis_state))
        print('[SIMULATOR] Genesis generation successful')

        if filename is not None:
            print('[SIMULATOR] Export Genesis State and Block to file')
            mocked = "mocked_" if mocked else ""
            state_file = f"state_{mocked}{filename}.ssz"
            block_file = f"block_{mocked}{filename}.ssz"
            with open(state_file, 'wb') as fp:
                fp.write(self.genesis_state.encode_bytes())
            with open(block_file, 'wb') as fp:
                fp.write(self.genesis_block.encode_bytes())

    @staticmethod
    def __obtain_validator_range(client: BeaconClient) -> Union[Tuple[int, int], None]:
        no_validators = len(client.validators)
        if no_validators == 0:
            return None
        elif no_validators == 1:
            return client.validators[0].counter, client.validators[0].counter
        else:
            return client.validators[0].counter, client.validators[-1].counter

    def initialize_clients(self):
        encoded_state = self.genesis_state.encode_bytes()
        encoded_block = self.genesis_block.encode_bytes()
        print('[SIMULATOR] state_root={} block_root={}'.format(
            spec.hash_tree_root(self.genesis_state), spec.hash_tree_root(self.genesis_block)
        ))
        print('[SIMULATOR] Start Validator processes')
        for client in self.clients:
            # Delete all validators inside the client and tell it to initialize them again
            # This is a lot faster than serializing the existing validator objects and sending them
            # over to the newly created process
            client.beacon_client.validators = []
            client.beacon_client.start()

            print(f'[SIMULATOR] Beacon Client {client.beacon_client.counter} started')
            client.queue.put(ValidatorInitializationEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                priority=0
            ))
            client.queue.put(MessageEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                priority=10,
                message=encoded_state,
                message_type='BeaconState',
                fromidx=0,
                toidx=None
            ))
            client.queue.put(MessageEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                priority=20,
                message=encoded_block,
                message_type='BeaconBlock',
                fromidx=0,
                toidx=None
            ))
        for client in self.clients:
            client.queue.join()

        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()
        print('[SIMULATOR] Initialization complete')

    def handle_next_slot_event(self, event: NextSlotEvent):
        if event.slot % spec.SLOTS_PER_EPOCH == 0:
            print(f"---------- EPOCH {spec.compute_epoch_at_slot(spec.Slot(event.slot))} ----------")
        print(f"!!! SLOT {event.slot} !!!")
        self.slot += 1
        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()
        self.__distribute_event(event)

    def __distribute_end_event(self, event: SimulationEndEvent):
        if event.message:
            print(f'---------- !!!!! {event.message} !!!!! ----------')
        self.should_quit = True
        self.__distribute_event(event)

    def __distribute_event(self, event: Event, receiver: Optional[int] = None):
        if receiver is not None:
            self.clients[receiver].queue.put(event)
            return
        for validator in self.clients:
            validator.queue.put(event)

    def __distribute_targeted_event(self, event: TargetedEvent):
        if event.toidx is None:
            raise ValueError('Event must have a receiver at this point')
        else:
            self.__distribute_event(event, event.toidx)

    def __recv_message_event(self, event: MessageEvent):
        if event.message_type == 'SignedBeaconBlock':
            print(f"[{int(datetime.datetime.now().timestamp())}] Block Message {id(event)} by Beacon Client {event.fromidx}")
        if event.toidx is not None:
            self.network.delay(event)
            self.events.put(event)
        else:
            for index in range(len(self.clients)):
                event_with_receiver = MessageEvent(
                    time=event.time,
                    priority=event.priority,
                    message=event.message,
                    message_type=event.message_type,
                    fromidx=event.fromidx,
                    marker=event.marker,
                    toidx=index
                )
                self.network.delay(event_with_receiver)
                self.events.put(event_with_receiver)

    def __recv_end_event(self, event: SimulationEndEvent):
        for validator in self.clients:
            validator.queue.put(event)
            validator.beacon_client.join()
        if event.message:
            print(f'---------- !!!!! {event.message} !!!!! ----------')
        self.should_quit = True

    def start_simulation(self):
        send_actions = {
            NextSlotEvent: self.handle_next_slot_event,
            LatestVoteOpportunity: self.__distribute_event,
            AggregateOpportunity: self.__distribute_event,
            MessageEvent: self.__distribute_targeted_event,
            ProduceStatisticsEvent: self.__distribute_targeted_event,
            ProduceGraphEvent: self.__distribute_targeted_event,
            SimulationEndEvent: self.__distribute_end_event
        }
        recv_actions = {
            MessageEvent: self.__recv_message_event,
            SimulationEndEvent: self.__recv_end_event
        }

        self.start_time = datetime.datetime.now()
        print(f"[{simtime(self.start_time)}] Start Simulation")

        while not self.should_quit:
            top = self.events.get()
            self.simulator_time = top.time
            self.simulator_prio = top.priority
            print(f'[{simtime(self.start_time)}] Current time: {self.simulator_time} '
                  f'({sec_to_time(self.simulator_time - self.genesis_time)} since genesis)')
            current_time_and_prio_events = list(self.__collect_events_upto_current_time_and_prio())
            current_time_and_prio_events.append(top)
            while len(current_time_and_prio_events) >= 1:
                for event in current_time_and_prio_events:
                    # noinspection PyTypeChecker
                    send_actions[type(event)](event)
                if not self.should_quit:
                    for client in self.clients:
                        client.queue.join()
                    recv_event = queue_element_or_none(self.queue)
                    while recv_event is not None:
                        # noinspection PyArgumentList
                        recv_actions[type(recv_event)](recv_event)
                        if recv_event.time < self.simulator_time:
                            print(f'[{simtime(self.start_time)}][WARNING] Shall distribute event for the past! {recv_event}')
                        recv_event = queue_element_or_none(self.queue)
                current_time_and_prio_events = tuple(self.__collect_events_upto_current_time_and_prio())
        print(f'[{simtime(self.start_time)}] [SIMULATOR] FINISH SIMULATION AT TIMESTAMP {self.simulator_time} '
              f'({sec_to_time(self.simulator_time - self.genesis_time)} since genesis)')

    def __collect_events_upto_current_time_and_prio(self, progress_time=False) -> Iterable[Event]:
        element: Optional[Event] = queue_element_or_none(self.events)
        while element is not None:
            if element.time == self.simulator_time and element.priority == self.simulator_prio:
                yield element
            elif element.time == self.simulator_time and element.priority < self.simulator_prio:
                print(f"[WARNING] Priority downgrade ({self.simulator_prio} -> {element.priority})")
                print(f'Current: {element}')
                print(f'Previous: {self.element_cache}')
                self.simulator_prio = element.priority
                yield element
            elif element.time < self.simulator_time:
                print(f'[WARNING] element.time is before simulator_time! {str(element)}')
                # TODO FIX THIS!!!
                element.time = self.simulator_time
                yield element
            else:
                self.events.put(element)
                if progress_time:
                    self.simulator_time = element.time
                    self.simulator_prio = element.priority
                return
            self.element_cache = element
            element = queue_element_or_none(self.events)


def sec_to_time(seconds: int) -> str:
    minutes = seconds // 60
    hours = minutes // 60

    seconds -= minutes * 60
    minutes -= hours * 60
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def simtime(start_time: datetime.datetime) -> str:
    delta = datetime.datetime.now() - start_time
    return sec_to_time(delta.seconds)


class SimulationBuilder(Builder):
    configpath: str
    configname: str
    rand: ByteVector
    current_child_count: int
    end_time: uint64
    statproducers: List[Tuple[int, int]]
    graphproducers: List[Tuple[int, int, bool]]
    custom_latency_map: Optional[Union[Tuple[Tuple[int]], Dict[str, Tuple[Tuple[int]]]]] = None
    latency_modifier: Callable[[int], int]

    beacon_client_builders: List[BeaconClientBuilder]

    def __init__(self, configpath, configname, rand, parent_builder=None):
        super(SimulationBuilder, self).__init__(parent_builder)
        self.beacon_client_builders = []
        self.configpath = configpath
        self.configname = configname
        self.rand = rand
        self.end_time = uint64((8 * spec.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH) + 24)
        self.statproducers = []
        self.graphproducers = []
        self.custom_latency_map = None
        self.latency_modifier = lambda x: x

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
        simulator = Simulator(self.rand, self.custom_latency_map, self.latency_modifier)
        simulator.queue = client_to_simulator_queue
        simulator.clients = clients
        simulator.events.put(SimulationEndEvent(simulator.genesis_time + self.end_time, 0))
        for client, time in self.statproducers:
            simulator.events.put(ProduceStatisticsEvent(simulator.genesis_time + time, 100, client))
        for client, time, show in self.graphproducers:
            simulator.events.put(ProduceGraphEvent(simulator.genesis_time + time, 100, client, show))
        return simulator

    def set_end_time(self, end_time):
        self.end_time = end_time
        return self

    def set_custom_latency_map(self, latency_map, modifier=None):
        self.custom_latency_map = latency_map
        if modifier is not None:
            self.latency_modifier = modifier
        return self

    def add_statistics_output(self, client, time):
        self.statproducers.append((client, time))
        return self

    def add_graph_output(self, client, time, show):
        self.graphproducers.append((client, time, show))
        return self

    def register(self, child_builder):
        for _ in range(self.current_child_count):
            self.beacon_client_builders.append(child_builder)
