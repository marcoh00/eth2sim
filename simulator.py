import argparse
import multiprocessing
import queue
import random
import sys
from datetime import datetime
from multiprocessing.spawn import freeze_support
from pathlib import Path
from typing import Optional, List, Sequence
from py_ecc.bls import G2ProofOfPossession as Bls
from py_ecc.bls12_381 import curve_order
from remerkleable.basic import uint64
from remerkleable.byte_arrays import ByteVector

import eth2spec.phase0.spec as spec
from eth2spec.config import config_util
from eth2spec.test.helpers.deposits import build_deposit
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, SimulationEndEvent, MessageEvent, \
    Event, MESSAGE_TYPE
from network import Network
from pathvalidation import valid_writable_path
from validator import Validator, encode_offload_slot_arguments, offload_slot_state_transition, \
    decode_offload_slot_arguments, decode_offload_slot_results, encode_offload_block_arguments, \
    offload_block_processing, decode_offload_block_results
from importlib import reload


class Simulator:

    genesis_time: uint64
    simulator_time: uint64
    slot: spec.Slot
    validators: List[Validator]
    network: Network
    events: queue.Queue[Event]
    past_events: List[Event]
    random: ByteVector

    def __init__(self, rand: ByteVector, pool: multiprocessing.Pool):
        self.genesis_time = spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY
        self.simulator_time = self.genesis_time.copy()
        self.slot = spec.Slot(0)
        self.validators = []
        self.network = Network(self, int.from_bytes(rand[0:4], "little"))
        self.events = queue.PriorityQueue()
        self.past_events = list()
        self.random = rand
        self.pool = pool

    def normalized_simulator_time(self):
        return self.simulator_time - self.genesis_time

    def add_validator(self, keys: Optional[Path]):
        pubkey = None
        privkey = None
        pubkey_file = None
        privkey_file = None

        if keys:
            pubkey_file = (keys / f'{len(self.validators)}.pubkey')
            privkey_file = (keys / f'{len(self.validators)}.privkey')
            if pubkey_file.is_file() and privkey_file.is_file():
                print('Read keys from file')
                pubkey = pubkey_file.read_bytes()
                privkey = int(privkey_file.read_text())
        if not privkey or not pubkey:
            print('Generate and save keys')
            privkey = random.randint(1, curve_order)
            pubkey = Bls.SkToPk(privkey)
            if keys:
                assert isinstance(pubkey_file, Path)
                pubkey_file.write_bytes(pubkey)
                privkey_file.write_text(str(privkey))
        print(f'[CREATE] Validator {len(self.validators)}')
        self.validators.append(Validator(len(self.validators), privkey=privkey, pubkey=pubkey))

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
        for validator in self.validators:
            print(f'[DEPOSIT] Validator {validator.index}')
            deposit, root, deposit_data = build_deposit(
                spec=spec,
                deposit_data_list=deposit_data,
                pubkey=validator.pubkey,
                privkey=validator.privkey,
                amount=spec.MAX_EFFECTIVE_BALANCE,
                withdrawal_credentials=spec.BLS_WITHDRAWAL_PREFIX + spec.hash(validator.pubkey)[1:],
                signed=True
            )
            deposits.append(deposit)
        eth1_timestamp = spec.MIN_GENESIS_TIME

        genesis_state = spec.initialize_beacon_state_from_eth1(eth1_blockhash, eth1_timestamp, deposits)
        assert spec.is_valid_genesis_state(genesis_state)
        genesis_block = spec.BeaconBlock(state_root=spec.hash_tree_root(genesis_state))

        for validator in self.validators:
            validator.state = genesis_state.copy()
            validator.index = max(
                genesis_validator[0]
                for genesis_validator in enumerate(genesis_state.validators)
                if genesis_validator[1].pubkey == validator.pubkey
            )
            validator.store = spec.get_forkchoice_store(validator.state, genesis_block)
            print(f'[STATE] Validator {validator.index}')

        for validator in self.validators:
            optional_attestation = validator.attest()
            if optional_attestation is not None:
                self.__handle_validator_message_event(*optional_attestation)
        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()
        print('[GENESIS] Alright')

    def handle_next_slot_event(self, event: NextSlotEvent):
        is_epoch_boundary = False
        if event.slot % spec.SLOTS_PER_EPOCH == 0:
            print("---------- EPOCH BOUNDARY ----------")
            is_epoch_boundary = True

        print(f"!!! SLOT {event.slot} !!!")
        self.slot += 1
        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()

        if not is_epoch_boundary:
            for validator in self.validators:
                optional_block = validator.handle_next_slot_event(event)
                if optional_block is not None:
                    self.__handle_validator_message_event(*optional_block)
                    print(optional_block[2])
        else:
            print('Pre-slot update')
            for validator in self.validators:
                validator.pre_slot_update(event)
            print('Serialize states...')
            states = (encode_offload_slot_arguments(validator, event) for validator in self.validators)
            print('Calc...')
            new_states = self.pool.map(offload_slot_state_transition, states)
            print('Got new states')
            for validator, newstate in zip(self.validators, new_states):
                head_state, store, proposer = decode_offload_slot_results(newstate)
                validator.store = store
                validator.proposer_current_slot = proposer
                optional_block = validator.post_slot_state_update(event, head_state)
                if optional_block is not None:
                    self.__handle_validator_message_event(*optional_block)
                    print(optional_block[2])

    def handle_latest_vote_opportunity(self, event: LatestVoteOpportunity):
        for validator in self.validators:
            optional_vote = validator.handle_latest_voting_opportunity(event)
            if optional_vote is not None:
                self.__handle_validator_message_event(*optional_vote)

    def handle_aggregate_opportunity(self, event: AggregateOpportunity):
        for validator in self.validators:
            optional_aggregate = validator.handle_aggregate_opportunity(event)
            if optional_aggregate is not None:
                self.__handle_validator_message_event(*optional_aggregate)

    def handle_message_event(self, event: MessageEvent):
        self.validators[event.toidx].handle_message_event(event)

    def handle_parallel_block_processing(self, cross_epoch_boundary_messages: Sequence[MessageEvent]):
        if len(cross_epoch_boundary_messages) > 0:
            validator_indices = tuple(message.toidx for message in cross_epoch_boundary_messages)
            print(f'Block serialize ({len(cross_epoch_boundary_messages)})')
            states = (encode_offload_block_arguments(message.message,
                                                     self.validators[message.toidx].state,
                                                     self.validators[message.toidx].store
                                                     ) for message in cross_epoch_boundary_messages)
            print('Block calc')
            results = self.pool.map(offload_block_processing, states)
            print('Block apply')
            for message, encoded_result in zip(cross_epoch_boundary_messages, results):
                state, store = decode_offload_block_results(encoded_result)
                self.validators[message.toidx].state = state
                self.validators[message.toidx].store = store
                for attestation in message.message.message.body.attestations:
                    self.validators[message.toidx].attestation_cache.add_attestation(attestation)

    def handle_bulk_block_messages(self, bulk_block_messages: Sequence[MessageEvent]):
        cross_epoch_boundary_messages = list()
        same_epoch_messages = list()
        for message in bulk_block_messages:
            assert isinstance(message.message, spec.SignedBeaconBlock)
            validator = self.validators[message.toidx]
            validator_epoch = spec.get_current_epoch(validator.state)
            block_epoch = spec.compute_epoch_at_slot(message.message.message.slot)
            if block_epoch > validator_epoch:
                cross_epoch_boundary_messages.append(message)
            else:
                same_epoch_messages.append(message)

        for message in same_epoch_messages:
            self.handle_message_event(message)
        self.handle_parallel_block_processing(cross_epoch_boundary_messages)

    def start_simulation(self):
        last_time = uint64(0)
        bulk_block_messages = list()
        actions = {
            NextSlotEvent: self.handle_next_slot_event,
            LatestVoteOpportunity: self.handle_latest_vote_opportunity,
            AggregateOpportunity: self.handle_aggregate_opportunity,
            MessageEvent: self.handle_message_event,
            Event: lambda event: None
        }
        while True:
            next_action = self.events.get()
            self.past_events.append(next_action)

            if next_action.time > last_time:
                print(f'At time: {next_action.time}')
                last_time = next_action.time
                self.handle_bulk_block_messages(bulk_block_messages)
                bulk_block_messages = list()

            if isinstance(next_action, SimulationEndEvent):
                break
            assert self.simulator_time <= next_action.time
            self.simulator_time = next_action.time

            if isinstance(next_action, MessageEvent) and isinstance(next_action.message, spec.SignedBeaconBlock):
                bulk_block_messages.append(next_action)
                continue

            actions[type(next_action)](next_action)

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


def multiprocessing_initializer(configpath: str, configname: str):
    config_util.prepare_config(configpath, configname)
    # noinspection PyTypeChecker
    reload(spec)
    spec.bls.bls_active = False


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configpath', type=str, required=False, default='../../configs')
    parser.add_argument('--configname', type=str, required=False, default='minimal')
    parser.add_argument('--cryptokeys', type=valid_writable_path, required=False, default='./cryptokeys')
    # parser.add_argument('--state', type=valid_writable_path, required=False, default='./state')
    parser.add_argument('--eth1blockhash', type=bytes.fromhex, required=False, default=random.randbytes(32).hex())
    args = parser.parse_args()
    print(f'Initialize Simulator with {multiprocessing.cpu_count()} helper processes')
    pool = multiprocessing.Pool(multiprocessing.cpu_count(),
                                initializer=multiprocessing_initializer,
                                initargs=(args.configpath, args.configname)
                                )
    config_util.prepare_config(args.configpath, args.configname)
    # noinspection PyTypeChecker
    reload(spec)
    simulator = Simulator(args.eth1blockhash, pool)
    spec.bls.bls_active = False

    print('Ethereum 2.0 Beacon Chain Simulator')
    print(f'Beacon Chain Configuration: {spec.CONFIG_NAME}')
    print(f'Eth1BlockHash for Genesis Block: {args.eth1blockhash.hex()}')
    print(f'Cryptographic Keys: {args.cryptokeys}')
    # print(f'State Backup: {args.state}')

    for i in range(64):
        simulator.add_validator(args.cryptokeys)
    simulator.generate_genesis(args.eth1blockhash)
    simulator.events.put(SimulationEndEvent(simulator.genesis_time + uint64(500)))
    simulator.start_simulation()


if __name__ == '__main__':
    freeze_support()
    start = datetime.now()
    try:
        test()
        end = datetime.now()
    except KeyboardInterrupt:
        end = datetime.now()
    print(f"Simulated {end-start}")