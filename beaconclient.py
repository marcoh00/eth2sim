import cProfile
import json
import pathlib
import pprint
import sys
import traceback
from importlib import reload
from multiprocessing import Queue, JoinableQueue
from multiprocessing.context import Process
import time
from typing import Tuple, Optional, List, Dict, Sequence, Set

from dataclasses import dataclass

import dataclasses
from graphviz import Digraph
from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist
from remerkleable.complex import Container

import eth2spec.phase0.spec as spec
from cache import AttestationCache, BlockCache
from colors import COLORS
from eth2spec.config import config_util
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, MessageEvent, Event, SimulationEndEvent, \
    ProduceStatisticsEvent, ProduceGraphEvent, ValidatorInitializationEvent
from helpers import queue_element_or_none
from validator import Validator, ValidatorBuilder


class BeaconClient(Process):
    counter: int
    simulator_to_client_queue: JoinableQueue
    client_to_simulator_queue: Queue
    state: spec.BeaconState
    store: spec.Store
    head_root: spec.Root

    validators: List[Validator]

    #                                                          Which, How many'th, total committee size
    committee: Dict[spec.Slot, Dict[spec.ValidatorIndex, Tuple[spec.CommitteeIndex, int, int]]]
    committee_count: Dict[spec.Epoch, int]

    current_slot: spec.Slot
    slot_last_attested: Optional[spec.Slot]
    last_attestation_data: Dict[spec.CommitteeIndex, spec.AttestationData]
    current_time: uint64
    proposer_current_slot: Optional[spec.ValidatorIndex]

    attestation_cache: AttestationCache
    block_cache: BlockCache
    slashings: Dict[str, List]

    should_quit: bool

    def __init__(self,
                 counter: int,
                 simulator_to_client_queue: JoinableQueue,
                 client_to_simulator_queue: Queue,
                 configpath: str,
                 configname: str,
                 validator_builders: Sequence[ValidatorBuilder],
                 validator_first_counter: int,
                 debug=False,
                 profile=False):
        super().__init__()
        self.counter = counter
        self.simulator_to_client_queue = simulator_to_client_queue
        self.client_to_simulator_queue = client_to_simulator_queue
        self.validator_builders = validator_builders
        self.validator_first_counter = validator_first_counter
        self.validators = []
        self.current_slot = spec.Slot(0)
        self.slot_last_attested = None
        self.committee = dict()
        self.committee_count = dict()
        self.last_attestation_data = dict()
        self.current_time = uint64(0)
        self.attestation_cache = AttestationCache()
        self.should_quit = False
        self.event_cache = None
        self.specconf = (configpath, configname)
        self.debug = debug
        self.debugfile = None
        self.profile = profile
        self.slashings = {'proposer': [], 'attester': []}
        self.starttime = int(time.time())
        self.proposer_current_slot = None
        self.build_validators()

    def __debug(self, obj, typ: str):
        if not self.debug:
            return
        if not self.debugfile:
            self.debugfile = open(f'log_{self.counter}_{int(time.time())}.json', 'w')
        if isinstance(obj, Container):
            logobj = str(obj)
        elif isinstance(obj, dict):
            logobj = dict()
            for key, value in obj.items():
                logobj[key] = str(value)
        elif isinstance(obj, int):
            logobj = obj
        else:
            logobj = str(obj)
        json.dump(
            {'level': typ,
             'index': self.counter,
             'timestamp': self.current_time,
             'slot': self.current_slot,
             'message': logobj}, self.debugfile)
        self.debugfile.write('\n')
        if 'Error' in typ:
            self.debugfile.flush()

    def run(self):
        config_util.prepare_config(self.specconf[0], self.specconf[1])
        # noinspection PyTypeChecker
        reload(spec)
        spec.bls.bls_active = False
        if self.profile:
            pr = cProfile.Profile()
            pr.enable()
        while not self.should_quit:
            item = self.simulator_to_client_queue.get()
            self.__handle_event(item)
            self.simulator_to_client_queue.task_done()
        if self.profile:
            pr.disable()
            pr.dump_stats(f"profile_{self.counter}_{time.time()}.prof")
        if self.debug:
            print(f'[BEACON CLIENT {self.counter}] Write Log file')
            self.debugfile.close()
        print(f"[BEACON CLIENT {self.counter}] Terminate process")

    def __handle_event(self, event: Event):
        if event.time < self.current_time:
            self.client_to_simulator_queue.put(
                SimulationEndEvent(time=self.current_time,
                                   message=f"Own time {self.current_time} is later than event time {event.time}. {event}/{self.event_cache}")
            )
        if event.time > self.current_time and self.debug and self.debugfile:
            self.debugfile.flush()
        self.event_cache = event
        self.current_time = event.time
        actions = {
            MessageEvent: self.__handle_message_event,
            ValidatorInitializationEvent: self.__handle_validator_initialization,
            SimulationEndEvent: self.__handle_simulation_end,
            LatestVoteOpportunity: self.handle_latest_voting_opportunity,
            AggregateOpportunity: self.handle_aggregate_opportunity,
            NextSlotEvent: self.handle_next_slot_event,
            ProduceStatisticsEvent: self.handle_statistics_event,
            ProduceGraphEvent: self.produce_graph_event
        }
        # noinspection PyArgumentList,PyTypeChecker
        actions[type(event)](event)

    def __handle_message_event(self, event: MessageEvent):
        decoder = {
            'Attestation': spec.Attestation,
            'SignedAggregateAndProof': spec.SignedAggregateAndProof,
            'SignedBeaconBlock': spec.SignedBeaconBlock,
            'BeaconBlock': spec.BeaconBlock,
            'BeaconState': spec.BeaconState
        }
        actions = {
            'Attestation': self.handle_attestation,
            'SignedAggregateAndProof': self.handle_aggregate,
            'SignedBeaconBlock': self.handle_block,
            'BeaconBlock': self.handle_genesis_block,
            'BeaconState': self.handle_genesis_state
        }
        self.__debug({
            'time': event.time,
            'prio': event.priority,
            'type': event.message_type,
            'marker': event.marker,
            'delayed': event.delayed
        }, 'MessageRecv')
        payload = decoder[event.message_type].decode_bytes(event.message)
        # noinspection PyArgumentList
        actions[event.message_type](payload)

    def __handle_validator_initialization(self, event: ValidatorInitializationEvent):
        self.build_validators()

    def handle_statistics_event(self, event: ProduceStatisticsEvent):
        stats = statistics(self)
        asdict = dataclasses.asdict(stats)
        if event.print_event:
            pprint.pp(asdict)
        self.__debug(str(stats), 'Statistics')
        filename = f'stats_{self.counter}_{self.starttime}.json'
        if not pathlib.Path(f'stats_{self.counter}_{self.starttime}.json').is_file():
            with open(filename, 'w', encoding='utf-8') as fp:
                fp.write('[')
        with open(f'stats_{self.counter}_{self.starttime}.json', 'a') as fp:
            fp.write(',\n')
            json.dump(asdict, fp, indent=2)

    def produce_graph_event(self, event: ProduceGraphEvent):
        self.graph(event.show)

    def handle_genesis_state(self, state: spec.BeaconState):
        print(f'[BEACON CLIENT {self.counter}] Initialize state from Genesis State')
        self.state = state
        for validator in self.validators:
            validator.index_from_state(state)

    def handle_genesis_block(self, block: spec.BeaconBlock):
        print(f'[BEACON CLIENT {self.counter}] Initialize Fork Choice and BlockCache with Genesis Block')
        self.store = spec.get_forkchoice_store(self.state, block)
        self.block_cache = BlockCache(block)
        self.head_root = spec.hash_tree_root(block)
        print(f'[BEACON CLIENT {self.counter}] Initialize committee cache for first epoch')
        self.update_committee(spec.Epoch(0))

    def __handle_simulation_end(self, event: SimulationEndEvent):
        self.__debug(event, 'SimulationEnd')
        statsfile = f'stats_{self.counter}_{self.starttime}.json'
        if self.debug and pathlib.Path(statsfile).is_file():
            with open(statsfile, 'a', encoding='utf-8') as fp:
                fp.write(']')
        print(f'[BEACON CLIENT {self.counter}] Received SimulationEndEvent')
        time.sleep(3)
        # Mark all remaining tasks as done
        next_task = queue_element_or_none(self.simulator_to_client_queue)
        while next_task is not None:
            self.simulator_to_client_queue.task_done()
            next_task = queue_element_or_none(self.simulator_to_client_queue)
        self.should_quit = True
        if self.debug:
            self.graph(show=False)

    def update_committee(self, epoch: spec.Epoch):
        start_slot = spec.compute_start_slot_at_epoch(epoch)
        committee_count_per_slot = spec.get_committee_count_per_slot(self.state, epoch)
        self.committee_count[epoch] = committee_count_per_slot
        for slot in (spec.Slot(s) for s in range(start_slot, start_slot + spec.SLOTS_PER_EPOCH)):
            if slot not in self.committee:
                self.committee[slot] = dict()
            for committee_index in (spec.CommitteeIndex(c) for c in range(committee_count_per_slot)):
                committee = spec.get_beacon_committee(self.state, spec.Slot(slot), spec.CommitteeIndex(committee_index))
                for validator_index in enumerate(committee):
                    self.committee[slot][validator_index[1]] = (committee_index, validator_index[0], len(committee))
            self.__debug({'slot': int(slot), 'committee': str(self.committee[slot])}, 'CommitteeUpdate')

    def propose_block(self, validator: Validator, head_state: spec.BeaconState, slashme=False):
        print(f"[BEACON CLIENT {self.counter}] Create block for Validator {self.proposer_current_slot}")
        min_slot_to_include = spec.compute_start_slot_at_epoch(
            spec.compute_epoch_at_slot(self.current_slot)
        ) if self.current_slot > spec.SLOTS_PER_EPOCH else spec.Slot(0)
        max_slot_to_include = self.current_slot - 1

        # If the attestation's target epoch is the current epoch,
        # the source checkpoint must match with the proposer's
        #
        # ep(target) == ep(now) => cp_justified == cp_source
        # <=> ep(target) != ep(now) v cp_justified == cp_source
        candidate_attestations = tuple(
            attestation
            for attestation in self.attestation_cache.attestations_not_seen_in_block(
                min_slot_to_include,
                max_slot_to_include
            )
            if attestation.data.target.epoch != spec.get_current_epoch(head_state)
            or head_state.current_justified_checkpoint == attestation.data.source
        )
        block = spec.BeaconBlock(
            slot=head_state.slot,
            proposer_index=validator.index,
            parent_root=self.head_root,
            state_root=spec.Bytes32(bytes(0 for _ in range(0, 32)))
        )
        randao_reveal = spec.get_epoch_signature(head_state, block, validator.privkey)

        # Copy eth1_data from previous block as we do not connect
        # to a (simulated) eth1 chain to process new deposits
        eth1_data = head_state.eth1_data

        # Include slashable blocks and attestations if the corresponding validators are deemed slashable
        proposer_slashings: List[spec.ProposerSlashing, spec.MAX_PROPOSER_SLASHINGS] = list(
            slashing for slashing in self.block_cache.search_slashings(head_state)
            if spec.is_slashable_validator(
                head_state.validators[slashing.signed_header_1.message.proposer_index],
                spec.get_current_epoch(head_state)
            )
        )
        attester_slashings: List[spec.AttesterSlashing, spec.MAX_ATTESTER_SLASHINGS] = list(
            slashing for slashing in self.attestation_cache.search_slashings(head_state)
            if any(
                spec.is_slashable_validator(head_state.validators[vindex], spec.get_current_epoch(head_state))
                for vindex in set(slashing.attestation_1.attesting_indices)
                    .intersection(set(slashing.attestation_2.attesting_indices)))
        )

        # Truncate the lists if they contain more elements than allowed by spec
        if len(proposer_slashings) > spec.MAX_PROPOSER_SLASHINGS:
            proposer_slashings = proposer_slashings[0:spec.MAX_PROPOSER_SLASHINGS]
        if len(attester_slashings) > spec.MAX_ATTESTER_SLASHINGS:
            attester_slashings = attester_slashings[0:spec.MAX_ATTESTER_SLASHINGS]
        if len(candidate_attestations) > spec.MAX_ATTESTATIONS:
            candidate_attestations = candidate_attestations[0:spec.MAX_ATTESTATIONS]

        # Provoke slashing by forming an alternative block which doesn't contain the last 2 attestations
        if slashme:
            candidate_attestations = candidate_attestations[0:len(candidate_attestations) - 2]

        # We do not support deposits and voluntary exists
        deposits: List[spec.Deposit, spec.MAX_DEPOSITS] = list()
        voluntary_exits: List[spec.VoluntaryExit, spec.MAX_VOLUNTARY_EXITS] = list()

        body = spec.BeaconBlockBody(
            randao_reveal=randao_reveal,
            eth1_data=eth1_data,
            graffiti=spec.Bytes32(bytes(255 for _ in range(0, 32))),
            proposer_slashings=proposer_slashings,
            attester_slashings=attester_slashings,
            attestations=candidate_attestations,
            deposits=deposits,
            voluntary_exits=voluntary_exits
        )
        block.body = body
        try:
            new_state_root = spec.compute_new_state_root(self.state, block)
        except AssertionError:
            validator_status = [str(validator) for validator in self.state.validators if validator.slashed]
            print(validator_status)
            print(block)
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            self.__debug({'text': text, 'block': str(block)}, 'FindNewStateRootError')
            self.client_to_simulator_queue.put(SimulationEndEvent(time=self.current_time, priority=0, message=text))
            return
        block.state_root = new_state_root
        signed_block = spec.SignedBeaconBlock(
            message=block,
            signature=spec.get_block_signature(self.state, block, validator.privkey)
        )
        encoded_signed_block = signed_block.encode_bytes()

        message = MessageEvent(
            time=self.current_time,
            priority=20,
            message=encoded_signed_block,
            message_type='SignedBeaconBlock',
            fromidx=self.counter,
            toidx=None
        )
        self.client_to_simulator_queue.put(message)
        if self.debug:
            self.graph(show=False)
        self.__debug(message.marker, 'BlockMessageSent')
        self.__debug(signed_block, 'ProposeBlock')

    def handle_next_slot_event(self, message: NextSlotEvent):
        self.current_slot = spec.Slot(message.slot)
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)

        spec.on_tick(self.store, message.time)
        if message.slot % spec.SLOTS_PER_EPOCH == 0:
            self.update_committee(current_epoch)

        # Keep two epochs of past attestations
        min_slot_to_keep = spec.compute_start_slot_at_epoch(
            spec.compute_epoch_at_slot(self.current_slot) - 1
        ) if self.current_slot > spec.SLOTS_PER_EPOCH else spec.Slot(0)
        self.attestation_cache.cleanup_time_cache(min_slot_to_keep)

        # Handle up to last slot's attestations
        for attestation in self.attestation_cache.attestations_not_known_to_forkchoice(
                spec.Slot(0),
                max(self.current_slot - 1, 0)):
            try:
                if attestation.data.beacon_block_root in self.store.blocks:
                    spec.on_attestation(self.store, attestation)
                    self.attestation_cache.accept_attestation(attestation, forkchoice=True)
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]
                print(f'[BEACON CLIENT {self.counter}] Could not validate attestation')
                self.__debug({'line': line, 'text': text, 'attestation': str(attestation)}, 'ValidateAttestationOnSlotBoundaryError')

        # Advance state for checking
        self.__compute_head()
        head_state = self.state.copy()
        if head_state.slot < message.slot:
            spec.process_slots(head_state, spec.Slot(message.slot))
        self.proposer_current_slot = spec.get_beacon_proposer_index(head_state)
        
        if self.proposer_current_slot in self.indexed_validators and not self.slashed(self.proposer_current_slot):
            # Propose block if needed
            self.propose_block(self.indexed_validators[self.proposer_current_slot], head_state)
        self.handle_statistics_event(
            ProduceStatisticsEvent(time=self.current_time, priority=0, toidx=None, print_event=False)
        )
        self.post_next_slot_event(head_state, self.indexed_validators)
    
    def post_next_slot_event(self, head_state, indexed_validators):
        pass

    def handle_latest_voting_opportunity(self, message: LatestVoteOpportunity):
        if self.slot_last_attested is None or self.slot_last_attested < message.slot:
            self.attest()

    def handle_aggregate_opportunity(self, message: AggregateOpportunity):
        if self.slot_last_attested != message.slot:
            return
        attesting_validators = self.__attesting_validators_at_current_slot()
        if len(attesting_validators) < 1:
            return
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        slot = spec.Slot(message.slot)
        attestations_to_aggregate = dict()

        # Attestations are aggregated per committee
        for committee in (spec.CommitteeIndex(c) for c in range(self.committee_count[current_epoch])):
            # Aggregate all known unaggregated attestations which represent
            # the same vote this beacon client and its validators attested for
            attestations_to_aggregate[committee] = [
                attestationcache.attestation
                for attestationcache
                in self.attestation_cache.cache_by_time.get(slot, dict()).get(committee, list())
                if committee in self.last_attestation_data and attestationcache.attestation.data == self.last_attestation_data[committee]
                and len(attestationcache.attesting_indices) == 1
            ]
        for (validator_index, validator) in attesting_validators.items():
            # For each validator, check if it is supposed to aggregate
            validator_committee = self.committee[self.current_slot][validator_index][0]
            slot_signature = spec.get_slot_signature(self.state, spec.Slot(message.slot), validator.privkey)
            if not spec.is_aggregator(self.state,
                                      spec.Slot(message.slot),
                                      validator_committee,
                                      slot_signature):
                continue

            aggregation_bits = list(0 for _ in range(self.committee[self.current_slot][validator_index][2]))

            # Set all bits representing the validators inside the aggregated attestation
            for attestation in attestations_to_aggregate[validator_committee]:
                for idx, bit in enumerate(attestation.aggregation_bits):
                    if bit:
                        aggregation_bits[idx] = 1
            # noinspection PyArgumentList
            aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*aggregation_bits)

            attestation = spec.Attestation(
                aggregation_bits=aggregation_bits,
                data=self.last_attestation_data[validator_committee],
                signature=spec.get_aggregate_signature(attestations_to_aggregate[validator_committee])
            )
            aggregate_and_proof = spec.get_aggregate_and_proof(
                self.state, validator_index, attestation, validator.privkey
            )
            aggregate_and_proof_signature = spec.get_aggregate_and_proof_signature(
                self.state, aggregate_and_proof, validator.privkey
            )
            signed_aggregate_and_proof = spec.SignedAggregateAndProof(
                message=aggregate_and_proof,
                signature=aggregate_and_proof_signature
            )
            encoded_aggregate_and_proof = signed_aggregate_and_proof.encode_bytes()
            send_message = MessageEvent(
                time=self.current_time,
                priority=30,
                message=encoded_aggregate_and_proof,
                message_type='SignedAggregateAndProof',
                fromidx=self.counter,
                toidx=None
            )
            self.client_to_simulator_queue.put(send_message)
            self.__debug(send_message.marker, 'AggregateAndProofMessageSent')
            self.__debug(signed_aggregate_and_proof, 'AggregateAndProofSend')

    def handle_attestation(self, attestation: spec.Attestation):
        self.__debug(attestation, 'AttestationRecv')
        self.attestation_cache.add_attestation(attestation, self.state)

    def handle_aggregate(self, aggregate: spec.SignedAggregateAndProof):
        # TODO check signature
        self.handle_attestation(aggregate.message.aggregate)

    def handle_block(self, block: spec.SignedBeaconBlock):
        # Try to process an old chain first
        old_chain = self.block_cache.longest_outstanding_chain(self.store)
        if len(old_chain) > 0:
            self.__debug(len(old_chain), 'OrphanedBlocksToHandle')
        for cblock in old_chain:
            spec.on_block(self.store, cblock)
            self.block_cache.accept_block(cblock)
            self.__process_block_contents(cblock)

        # Process the new block now
        self.__debug({
            'root': spec.hash_tree_root(block.message),
            'parent': block.message.parent_root,
            'proposer': block.message.proposer_index
        }, 'BlockRecv')
        self.block_cache.add_block(block)
        try:
            chain = self.block_cache.chain_for_block(block, self.store)
        except KeyError as e:
            print(f'[BEACON CLIENT {self.counter}] Could not validate block: {e} - parts of chain are missing')
            chain = []
        self.__debug(len(chain), 'BlocksToHandle')
        for cblock in chain:
            try:
                spec.on_block(self.store, cblock)
                self.block_cache.accept_block(cblock)
                self.__process_block_contents(cblock)
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]
                print(f'[BEACON CLIENT {self.counter}] Could not validate block (assert line {line}, stmt {text}! {str(block.message)}!')

    def __process_block_contents(self, block: spec.SignedBeaconBlock):
        for aslashing in block.message.body.attester_slashings:
            self.__debug(aslashing, 'AttesterSlashing')
            self.slashings['attester'].append(str(aslashing))
        for pslashing in block.message.body.proposer_slashings:
            self.__debug(pslashing, 'ProposerSlashing')
            self.slashings['proposer'].append(str(pslashing))
        for attestation in block.message.body.attestations:
            self.attestation_cache.add_attestation(attestation, self.state, seen_in_block=True)

    def handle_message_event(self, message: MessageEvent):
        actions = {
            'Attestation': self.handle_attestation,
            'SignedAggregateAndProof': self.handle_aggregate,
            'SignedBeaconBlock': self.handle_block
        }
        decoder = {
            'Attestation': spec.Attestation.decode_bytes,
            'SignedAggregateAndProof': spec.SignedAggregateAndProof.decode_bytes,
            'SignedBeaconBlock': spec.SignedBeaconBlock.decode_bytes
        }
        # noinspection PyArgumentList
        payload = decoder[message.message_type](message.message)
        # noinspection PyArgumentList
        actions[message.message_type](payload)

    def attest(self):
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        if self.current_slot not in self.committee or current_epoch not in self.committee_count:
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
        attesting_validators = self.__attesting_validators_at_current_slot()
        if len(attesting_validators) < 1:
            return

        self.__compute_head()
        head_block = self.store.blocks[self.head_root]
        head_state = self.state.copy()
        if head_state.slot < self.current_slot:
            # If the current head state's slot is lower than the current slot by time,
            # advance the state to the current slot.
            # Internally, the last seen block is copied and used as the block for every slot
            # up until the current slot.
            spec.process_slots(head_state, self.current_slot)

        # As specified inside validator.md spec / Attesting / Note:
        start_slot = spec.compute_start_slot_at_epoch(spec.get_current_epoch(head_state))
        epoch_boundary_block_root = spec.hash_tree_root(head_block) \
            if start_slot == head_state.slot \
            else spec.get_block_root(head_state, spec.get_current_epoch(head_state))

        # Create Attestation for every managed Validator who is supposed to attest
        for (validator_index, validator) in attesting_validators.items():
            # validator_committee:
            # 0: CommiteeIndex of the validator's committee
            # 1: Index of the validator inside the committee
            # 2: Size of the committee
            validator_committee = self.committee[self.current_slot][validator_index][0]
            attestation_data = self.produce_attestation_data(validator_index, validator_committee, epoch_boundary_block_root, head_block, head_state)
            if attestation_data is None:
                continue

            aggregation_bits_list = list(
                0 for _ in range(self.committee[self.current_slot][validator_index][2])
            )
            aggregation_bits_list[self.committee[self.current_slot][validator_index][1]] = 1
            # noinspection PyArgumentList
            aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*aggregation_bits_list)

            attestation = spec.Attestation(
                data=attestation_data,
                aggregation_bits=aggregation_bits,
                signature=spec.get_attestation_signature(self.state, attestation_data, validator.privkey)
            )

            self.attestation_cache.add_attestation(attestation, self.state)
            self.slot_last_attested = self.current_slot.copy()
            self.last_attestation_data[validator_committee] = attestation_data

            encoded_attestation = attestation.encode_bytes()
            message = MessageEvent(
                time=self.current_time,
                priority=30,
                message=encoded_attestation,
                message_type='Attestation',
                fromidx=self.counter,
                toidx=None
            )
            self.client_to_simulator_queue.put(message)
            self.__debug(message.marker, 'AttestationMessageSent')
            self.__debug(attestation, 'AttestationSend')
    
    def produce_attestation_data(self, validator_index, validator_committee, epoch_boundary_block_root, head_block, head_state) -> Optional[spec.AttestationData]:
        return spec.AttestationData(
                slot=self.current_slot,
                index=validator_committee,
                beacon_block_root=spec.hash_tree_root(head_block),
                source=head_state.current_justified_checkpoint,
                target=spec.Checkpoint(epoch=spec.get_current_epoch(head_state), root=epoch_boundary_block_root)
            )

    def __compute_head(self):
        self.head_root = spec.get_head(self.store)
        self.state = self.store.block_states[self.head_root]

    def slashed(self, validator: Optional[spec.ValidatorIndex] = None) -> bool:
        if self.state.validators[validator].slashed:
            print(f'[BEACON CLIENT {self.counter}] Excluding Validator {validator} beacuse of slashing')
        return self.state.validators[validator].slashed

    def __attesting_validators_at_current_slot(self) -> Dict[spec.ValidatorIndex, Validator]:
        attesting_indices = set(validator.index for validator in self.validators) \
            .intersection(set(self.committee[self.current_slot].keys()))
        attesting_validators: Dict[spec.ValidatorIndex, Validator] = {validator.index: validator
                                                                      for validator in self.validators
                                                                      if validator.index in attesting_indices
                                                                      and not self.slashed(validator.index)}
        return attesting_validators

    def __validators_by_index(self) -> Dict[spec.ValidatorIndex, Validator]:
        return {validator.index: validator
                for validator in self.validators}

    @staticmethod
    def __validator_color(validator: spec.ValidatorIndex):
        return COLORS[validator % len(COLORS)]

    def graph(self, show=True):
        g = Digraph('G', filename=f'graph_{int(time.time())}_{self.counter}.gv')
        blocks_by_slot_and_epoch: Dict[spec.Epoch, Dict[spec.Slot, List[Tuple[spec.Root, spec.BeaconBlock]]]] = {}
        for root, block in self.store.blocks.items():
            epoch = spec.compute_epoch_at_slot(block.slot)
            slot = block.slot
            if epoch not in blocks_by_slot_and_epoch:
                blocks_by_slot_and_epoch[epoch] = {}
            if slot not in blocks_by_slot_and_epoch[epoch]:
                blocks_by_slot_and_epoch[epoch][slot] = []
            blocks_by_slot_and_epoch[epoch][slot].append((root, block))

        justified_epoch, justified_root = self.store.justified_checkpoint.epoch, self.store.justified_checkpoint.root
        finalized_epoch, finalized_root = self.store.finalized_checkpoint.epoch, self.store.finalized_checkpoint.root
        head_root = self.head_root
        for epoch in blocks_by_slot_and_epoch.keys():
            with g.subgraph(name=f'cluster_epoch_{int(epoch)}') as e:
                e.attr(label=f'Epoch {int(epoch)}')
                epoch_style = 'solid'
                epoch_color = 'grey'
                if epoch <= justified_epoch:
                    epoch_style = 'solid'
                    epoch_color = 'black'
                if epoch <= finalized_epoch:
                    epoch_style = 'bold'
                    epoch_color = 'black'
                e.attr(color=epoch_color, style=epoch_style)
                for slot in blocks_by_slot_and_epoch[epoch].keys():
                    with e.subgraph(name=f'cluster_slot_{int(slot)}') as s:
                        s.attr(label=f'Slot {int(slot)}', style='dashed')
                        # print(f'At slot: {int(slot)}')
                        for root, block in blocks_by_slot_and_epoch[epoch][slot]:
                            color = self.__validator_color(block.proposer_index)
                            shape = 'box'
                            if root == head_root:
                                shape = 'octagon'
                            if root == justified_root:
                                shape = 'doubleoctagon'
                            if root == finalized_root:
                                shape = 'tripleoctagon'
                            s.node(str(root), shape=shape, color=color, style='filled')
        for epoch in blocks_by_slot_and_epoch.keys():
            for slot in blocks_by_slot_and_epoch[epoch].keys():
                for root, block in blocks_by_slot_and_epoch[epoch][slot]:
                    g.edge(str(root), str(block.parent_root))
        for validator, message in self.store.latest_messages.items():
            color = self.__validator_color(validator)
            g.node(str(validator), shape='circle', color=color, style='filled')
            g.edge(str(validator), str(message.root))
        # print(blocks_by_slot_and_epoch)

        g.save()
        if show:
            g.view()

    def build_validators(self):
        self.validators = list()
        for builder in self.validator_builders:
            for _ in range(0, builder.validators_count):
                self.validators.append(builder.build(False, self.validator_first_counter + len(self.validators)))
        self.indexed_validators = {validator.index: validator for validator in self.validators}


# Statistics

@dataclass
class Statistics:
    client_index: int
    current_time: int
    current_slot: int
    current_epoch: int
    head: str
    finalized: Tuple[str, int, int]
    justified: Tuple[str, int, int]
    head_balance: spec.Gwei
    finalized_balance: spec.Gwei
    justified_balance: spec.Gwei
    leafs: Sequence[str]
    leafs_since_justification: Sequence[str]
    leafs_since_finalization: Sequence[str]
    orphans: Sequence[Tuple[str, int]]
    attester_slashings: Sequence[int]
    proposer_slashings: Sequence[int]
    finality_delay: int
    validators_left: Sequence[Tuple[spec.ValidatorIndex, int]]
    current_base_reward: spec.Gwei
    balances: Dict[int, int]


def statistics(client: BeaconClient) -> Statistics:
    return Statistics(
        client_index=client.counter,
        current_time=client.current_time,
        current_slot=client.current_slot,
        current_epoch=spec.compute_epoch_at_slot(client.current_slot),
        head=str(client.head_root),
        finalized=(str(client.store.finalized_checkpoint.root),
                   int(client.store.blocks[client.store.finalized_checkpoint.root].slot),
                   int(client.store.finalized_checkpoint.epoch)),
        justified=(str(client.store.justified_checkpoint.root),
                   int(client.store.blocks[client.store.justified_checkpoint.root].slot),
                   int(client.store.justified_checkpoint.epoch)),
        head_balance=spec.get_latest_attesting_balance(client.store, client.head_root),
        finalized_balance=spec.get_latest_attesting_balance(client.store, client.store.finalized_checkpoint.root),
        justified_balance=spec.get_latest_attesting_balance(client.store, client.store.justified_checkpoint.root),
        leafs=tuple(str(root) for root in get_leaf_blocks(client.store, None)),
        leafs_since_justification=tuple(str(root) for root in get_leaf_blocks(client.store,
                                                                              client.store.justified_checkpoint.root)),
        leafs_since_finalization=tuple(str(root) for root in get_leaf_blocks(client.store,
                                                                             client.store.finalized_checkpoint.root)),
        orphans=get_orphans(client.store),
        attester_slashings=client.slashings['attester'],
        proposer_slashings=client.slashings['proposer'],
        finality_delay=spec.get_finality_delay(client.state),
        validators_left=tuple((spec.ValidatorIndex(v[0]), v[1].exit_epoch)
                              for v in enumerate(client.state.validators)
                              if v[1].exit_epoch != spec.FAR_FUTURE_EPOCH),
        current_base_reward=spec.get_base_reward(client.state, spec.ValidatorIndex(0)),
        balances={index: client.state.balances[index]
                  for index, _ in enumerate(client.state.validators)}
    )


def get_leaf_blocks(store: spec.Store, forced_parent: Optional[spec.Root]) -> Set[spec.Root]:
    blocks = set(store.blocks.keys())
    if forced_parent:
        blocks = get_chain_blocks(store, forced_parent)

    has_children = set()
    for blockroot in blocks:
        has_children.add(store.blocks[blockroot].parent_root)
    return blocks.difference(has_children)


def get_chain_blocks(store: spec.Store, base_block: spec.Root) -> Set[spec.Root]:
    chain = set()
    children = get_children(store, base_block)
    while len(children) > 0:
        chain = chain.union(children)
        new_children = set()
        for child in children:
            new_children = new_children.union(get_children(store, child))
        children = new_children
    return chain


def get_children(store: spec.Store, parent: spec.Root) -> Set[spec.Root]:
    children = set(root for root, block in store.blocks.items() if block.parent_root == parent)
    return children


def get_orphans(store: spec.Store) -> Sequence[Tuple[str, int]]:
    return tuple((str(root), int(block.slot))
                 for root, block in store.blocks.items()
                 if block.parent_root not in store.blocks)

