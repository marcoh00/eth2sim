import queue
from dataclasses import dataclass
from multiprocessing import Queue, JoinableQueue
from pathlib import Path
from typing import Optional, List, Sequence, Iterable, Union, Tuple

from remerkleable.basic import uint64
from remerkleable.byte_arrays import ByteVector

import eth2spec.phase0.spec as spec
from beaconclient import BeaconClient
from eth2spec.test.helpers.deposits import build_deposit
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, SimulationEndEvent, MessageEvent, \
    Event, MESSAGE_TYPE, ProduceGraphEvent, ProduceStatisticsEvent, TargetedEvent, ValidatorInitializationEvent
from helpers import queue_element_or_none
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
    slot: spec.Slot
    clients: List[IndexedBeaconClient]
    network: Network
    events: queue.Queue[Event]
    past_events: List[Event]
    random: ByteVector
    queue: Queue
    should_quit: bool

    def __init__(self, rand: ByteVector):
        self.genesis_time = spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY
        self.simulator_time = self.genesis_time.copy()
        self.slot = spec.Slot(0)
        self.clients = []
        self.network = Network(self, int.from_bytes(rand[0:4], "little"))
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
                self.slot + 1)
        )

    def next_latest_vote_opportunity(self):
        self.events.put(
             LatestVoteOpportunity(
                 self.genesis_time + (self.slot * spec.SECONDS_PER_SLOT) + (spec.SECONDS_PER_SLOT // 3),
                 self.slot)
        )

    def next_aggregate_opportunity(self):
        self.events.put(
             AggregateOpportunity(
                 self.genesis_time + (self.slot * spec.SECONDS_PER_SLOT) + (2 * (spec.SECONDS_PER_SLOT // 3)),
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

    def generate_genesis(self, filename=None):
        deposit_data = []
        deposits = []

        print('[SIMULATOR] Generate Genesis state')
        eth1_timestamp = spec.MIN_GENESIS_TIME
        for client in self.clients:
            for validator in client.beacon_client.validators:
                print(f'[VALIDATOR {validator.counter}] Deposit {validator.startbalance} Gwei')
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

        self.genesis_state = spec.initialize_beacon_state_from_eth1(self.random, eth1_timestamp, deposits)
        assert spec.is_valid_genesis_state(self.genesis_state)
        self.genesis_block = spec.BeaconBlock(state_root=spec.hash_tree_root(self.genesis_state))
        print('[SIMULATOR] Genesis generation successful')

        if filename is not None:
            print('[SIMULATOR] Export Genesis State and Block to file')
            state_file = f"state_{filename}.ssz"
            block_file = f"block_{filename}.ssz"
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

            # TODO somehow combine this with the builder. This wouldn't work for different keydirs for example.
            # Different startbalances wouldn't be accessible either after this
            # (although these are currently never accessed after genesis anymore)
            client_validator_counters = self.__obtain_validator_range(client.beacon_client)
            keydir = 'cryptokeys'
            client.beacon_client.validators = []

            client.beacon_client.start()

            print(f'[SIMULATOR] Beacon Client {client.beacon_client.counter} started')
            client.queue.put(ValidatorInitializationEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                empty_list=client_validator_counters is None,
                startidx=client_validator_counters[0] if client_validator_counters is not None else None,
                endidx=client_validator_counters[1] if client_validator_counters is not None else None,
                keydir=keydir
            ))
            client.queue.put(MessageEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                message=encoded_state,
                message_type='BeaconState',
                fromidx=0,
                toidx=None
            ))
            client.queue.put(MessageEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
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
        if event.toidx is not None:
            self.network.delay(event)
            self.events.put(event)
        else:
            for index in range(len(self.clients)):
                event_with_receiver = MessageEvent(
                    time=event.time,
                    message=event.message,
                    message_type=event.message_type,
                    fromidx=event.fromidx,
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

        while not self.should_quit:
            top = self.events.get()
            self.simulator_time = top.time
            print(f'Current time: {self.simulator_time}')
            current_time_events = list(self.__collect_events_upto_current_time())
            current_time_events.append(top)
            while len(current_time_events) >= 1:
                for event in current_time_events:
                    # noinspection PyTypeChecker
                    send_actions[type(event)](event)
                if not self.should_quit:
                    for validator in self.clients:
                        validator.queue.join()
                    recv_event = queue_element_or_none(self.queue)
                    while recv_event is not None:
                        # noinspection PyArgumentList
                        recv_actions[type(recv_event)](recv_event)
                        if recv_event.time < self.simulator_time:
                            print(f'[WARNING] Shall distribute event for the past! {recv_event}')
                        recv_event = queue_element_or_none(self.queue)
                current_time_events = tuple(self.__collect_events_upto_current_time())

    def __collect_events_upto_current_time(self, progress_time=False) -> Iterable[Event]:
        element = queue_element_or_none(self.events)
        while element is not None:
            if element.time == self.simulator_time:
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
                return
            element = queue_element_or_none(self.events)

    def __handle_validator_message_event(self,
                                         fromidx: spec.ValidatorIndex,
                                         toidx: Optional[Sequence[spec.ValidatorIndex]],
                                         message: MESSAGE_TYPE):
        if toidx:
            for validatoridx in toidx:
                self.network.send(fromidx, validatoridx, message)
        else:
            for validator in self.clients:
                self.network.send(fromidx, spec.ValidatorIndex(validator.index), message)
