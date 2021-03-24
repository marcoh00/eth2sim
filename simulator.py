import argparse
import multiprocessing
import queue
import random
from datetime import datetime
from multiprocessing import Queue, JoinableQueue
from pathlib import Path
from typing import Optional, List, Sequence, Tuple, Iterable

from dataclasses import dataclass
from py_ecc.bls12_381 import curve_order
from remerkleable.basic import uint64
from remerkleable.byte_arrays import ByteVector

import eth2spec.phase0.spec as spec
from eth2spec.config import config_util
from eth2spec.test.helpers.deposits import build_deposit
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, SimulationEndEvent, MessageEvent, \
    Event, MESSAGE_TYPE, RequestDeposit
from helpers import queue_element_or_none
from network import Network
from pathvalidation import valid_writable_path
from validator import Validator
from importlib import reload


@dataclass
class IndexedValidator(object):
    queue: JoinableQueue
    validator: Validator


class Simulator:

    genesis_time: uint64
    simulator_time: uint64
    slot: spec.Slot
    validators: List[IndexedValidator]
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
        self.validators = []
        self.network = Network(self, int.from_bytes(rand[0:4], "little"))
        self.events = queue.PriorityQueue()
        self.past_events = list()
        self.random = rand
        self.queue = Queue()
        self.should_quit = False

    def normalized_simulator_time(self):
        return self.simulator_time - self.genesis_time

    def add_validator(self, configpath: str, configname: str, keys: Optional[str]):
        print(f'[CREATE] Validator {len(self.validators)}')
        debug = False
        if len(self.validators) % 1 == 0:
            debug = True
        validator_queue = JoinableQueue()
        self.validators.append(
            IndexedValidator(validator_queue, Validator(len(self.validators), validator_queue, self.queue, configpath, configname, keys, debug))
        )

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

    def generate_genesis(self, eth1_blockhash):
        deposit_data = []
        deposits = []

        print('Generate Genesis state')
        for validator in self.validators:
            deposit, root, deposit_data = build_deposit(
                spec=spec,
                deposit_data_list=deposit_data,
                pubkey=validator.validator.pubkey,
                privkey=validator.validator.privkey,
                amount=spec.MAX_EFFECTIVE_BALANCE,
                withdrawal_credentials=spec.BLS_WITHDRAWAL_PREFIX + spec.hash(validator.validator.pubkey)[1:],
                signed=True
            )
            deposits.append(deposit)
        eth1_timestamp = spec.MIN_GENESIS_TIME

        genesis_state = spec.initialize_beacon_state_from_eth1(eth1_blockhash, eth1_timestamp, deposits)
        assert spec.is_valid_genesis_state(genesis_state)
        genesis_block = spec.BeaconBlock(state_root=spec.hash_tree_root(genesis_state))

        encoded_state = genesis_state.encode_bytes()
        encoded_block = genesis_block.encode_bytes()

        print('Start Validator processes')
        color_step = min((0xFFFFFF - 0x7F7F7F) // len(self.validators), 0x27F)
        for validator in self.validators:
            validator.validator.colorstep = color_step
            validator.validator.start()
            validator.queue.put(MessageEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                message=encoded_state,
                message_type='BeaconState',
                fromidx=0,
                toidx=None
            ))
            validator.queue.put(MessageEvent(
                time=uint64(spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY),
                message=encoded_block,
                message_type='BeaconBlock',
                fromidx=0,
                toidx=None
            ))
            # vqueue.put(SimulationEndEvent(time=uint64(0)))
        for validator in self.validators:
            validator.queue.join()
            # validator.join()

        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()
        print('[GENESIS] Alright')

    def handle_next_slot_event(self, event: NextSlotEvent):
        if event.slot % spec.SLOTS_PER_EPOCH == 0:
            print("---------- EPOCH BOUNDARY ----------")
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
            self.validators[receiver].queue.put(event)
            return
        for validator in self.validators:
            validator.queue.put(event)

    def __distribute_message_event(self, event: MessageEvent):
        if event.toidx is None:
            raise ValueError('Event must have a receiver at this point')
        else:
            self.__distribute_event(event, event.toidx)

    def __recv_message_event(self, event: MessageEvent):
        if event.toidx is not None:
            self.network.delay(event)
            self.events.put(event)
        else:
            for index in range(len(self.validators)):
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
        for validator in self.validators:
            validator.queue.put(event)
            validator.validator.join()
        if event.message:
            print(f'---------- !!!!! {event.message} !!!!! ----------')
        self.should_quit = True

    def start_simulation(self):
        send_actions = {
            NextSlotEvent: self.handle_next_slot_event,
            LatestVoteOpportunity: self.__distribute_event,
            AggregateOpportunity: self.__distribute_event,
            MessageEvent: self.__distribute_message_event,
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
                    # print('WAIT FOR VALIDATORS TO FINISH TASKS')
                    for validator in self.validators:
                        validator.queue.join()
                        # print(f"{validator.validator.counter} IS FINISHED")
                    recv_event = queue_element_or_none(self.queue)
                    while recv_event is not None:
                        # noinspection PyArgumentList
                        recv_actions[type(recv_event)](recv_event)
                        # print(f"Current time: {self.simulator_time} / Event time: {recv_event.time} / Event: {type(recv_event).__name__}")
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
            for validator in self.validators:
                self.network.send(fromidx, spec.ValidatorIndex(validator.index), message)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configpath', type=str, required=False, default='../../configs')
    parser.add_argument('--configname', type=str, required=False, default='minimal')
    parser.add_argument('--cryptokeys', type=valid_writable_path, required=False, default='./cryptokeys')
    # parser.add_argument('--state', type=valid_writable_path, required=False, default='./state')
    parser.add_argument('--eth1blockhash', type=bytes.fromhex, required=False, default=random.randbytes(32).hex())
    args = parser.parse_args()
    config_util.prepare_config(args.configpath, args.configname)
    # noinspection PyTypeChecker
    reload(spec)
    simulator = Simulator(args.eth1blockhash)
    spec.bls.bls_active = False

    print('Ethereum 2.0 Beacon Chain Simulator')
    print(f'Beacon Chain Configuration: {spec.CONFIG_NAME}')
    print(f'Eth1BlockHash for Genesis Block: {args.eth1blockhash.hex()}')
    print(f'Cryptographic Keys: {args.cryptokeys}')

    for i in range(64):
        simulator.add_validator(args.configpath, args.configname, args.cryptokeys)
    simulator.generate_genesis(args.eth1blockhash)
    simulator.events.put(SimulationEndEvent(simulator.genesis_time + uint64((3 * spec.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH) + 24)))
    simulator.start_simulation()


if __name__ == '__main__':
    start = datetime.now()
    try:
        multiprocessing.set_start_method("spawn")
        test()
        end = datetime.now()
    except KeyboardInterrupt:
        end = datetime.now()
    print(f"{end-start}")
