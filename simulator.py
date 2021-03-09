import argparse
import queue
import random
from pathlib import Path
from typing import Optional, Sequence, List
from py_ecc.bls import G2ProofOfPossession as Bls
from py_ecc.bls12_381 import curve_order
from eth_typing import BLSSignature, BLSPubkey
from remerkleable.basic import uint64
from remerkleable.byte_arrays import ByteVector

import eth2spec.phase0.spec as spec
import eth2spec.test.utils as utils
from eth2spec.config import config_util
from eth2spec.test.helpers.deposits import build_deposit
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, SimulationEndEvent, MessageEvent, Event
from network import Network
from pathvalidation import valid_writable_path
from validator import Validator
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

    def __init__(self, rand: ByteVector):
        self.genesis_time = spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY
        self.simulator_time = self.genesis_time.copy()
        self.slot = spec.Slot(0)
        self.validators = []
        self.network = Network(self, int.from_bytes(rand[0:4], "little"))
        self.events = queue.PriorityQueue()
        self.past_events = list()
        self.random = rand

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
                self.network.send(*optional_attestation)
        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()
        print('[GENESIS] Alright')

    def handle_next_slot_event(self, event: NextSlotEvent):
        self.slot += 1
        self.next_latest_vote_opportunity()
        self.next_aggregate_opportunity()
        self.next_slot_event()
        for validator in self.validators:
            optional_block = validator.handle_next_slot_event(event)
            if optional_block is not None:
                self.network.send(*optional_block)
        print(event)

    def handle_latest_vote_opportunity(self, event: LatestVoteOpportunity):
        for validator in self.validators:
            optional_vote = validator.handle_latest_voting_opportunity(event)
            if optional_vote is not None:
                self.network.send(*optional_vote)
        print(event)

    def handle_aggregate_opportunity(self, event: AggregateOpportunity):
        for validator in self.validators:
            optional_aggregate = validator.handle_aggregate_opportunity(event)
            if optional_aggregate is not None:
                self.network.send(*optional_aggregate)
        print(event)

    def handle_message_event(self, event: MessageEvent):
        if event.toidx:
            for toidx in event.toidx:
                self.validators[toidx].handle_message_event(event)
        else:
            for validator in self.validators:
                validator.handle_message_event(event)
        print(event)

    def start_simulation(self):
        actions = {
            NextSlotEvent: self.handle_next_slot_event,
            LatestVoteOpportunity: self.handle_latest_vote_opportunity,
            AggregateOpportunity: self.handle_aggregate_opportunity,
            MessageEvent: self.handle_message_event
        }
        while True:
            next_action = self.events.get()
            self.past_events.append(next_action)
            print(f'At time: {next_action.time}')
            if isinstance(next_action, SimulationEndEvent):
                break
            assert self.simulator_time <= next_action.time
            self.simulator_time = next_action.time
            actions[type(next_action)](next_action)



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
    SIMULATOR = Simulator(args.eth1blockhash)
    spec.bls.bls_active = False

    print('Ethereum 2.0 Beacon Chain Simulator')
    print(f'Beacon Chain Configuration: {spec.CONFIG_NAME}')
    print(f'Eth1BlockHash for Genesis Block: {args.eth1blockhash.hex()}')
    print(f'Cryptographic Keys: {args.cryptokeys}')
    # print(f'State Backup: {args.state}')

    for i in range(64):
        SIMULATOR.add_validator(args.cryptokeys)
    SIMULATOR.generate_genesis(args.eth1blockhash)
    SIMULATOR.events.put(SimulationEndEvent(SIMULATOR.genesis_time + uint64(1000)))
    SIMULATOR.start_simulation()


test()

