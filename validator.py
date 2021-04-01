import cProfile
import json
import sys
import traceback
from importlib import reload
from multiprocessing.context import Process
from multiprocessing import Queue, JoinableQueue
from pathlib import Path
import random
from time import sleep, time
from typing import Tuple, Sequence, Optional, List, Dict, Set

from graphviz import Graph, Digraph
from py_ecc.bls12_381 import curve_order
from py_ecc.bls import G2ProofOfPossession as Bls
from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist
from eth_typing import BLSPubkey
from remerkleable.complex import Container

import eth2spec.phase0.spec as spec
from cache import AttestationCache, BlockCache
from eth2spec.config import config_util
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, MessageEvent, MESSAGE_TYPE, \
    RequestDeposit, Event, SimulationEndEvent
from helpers import popcnt, queue_element_or_none


class Validator(Process):
    index: Optional[spec.ValidatorIndex]
    counter: int
    privkey: int
    pubkey: BLSPubkey
    recv_queue: JoinableQueue
    send_queue: Queue
    state: spec.BeaconState
    store: spec.Store
    head_root: spec.Root

    committee: Optional[Tuple[Sequence[spec.ValidatorIndex], spec.CommitteeIndex, spec.Slot]]
    slot_last_attested: Optional[spec.Slot]
    last_attestation_data: spec.AttestationData

    current_slot: spec.Slot
    current_time: uint64
    proposer_current_slot: Optional[spec.ValidatorIndex]

    attestation_cache: AttestationCache
    block_cache: BlockCache

    should_quit: bool

    def __init__(self,
                 counter: int,
                 recv_queue: JoinableQueue,
                 send_queue: Queue,
                 configpath: str,
                 configname: str,
                 keydir: Optional[str],
                 debug=False):
        super().__init__()
        self.privkey, self.pubkey = self.__init_keys(counter, keydir)
        self.counter = counter
        self.recv_queue = recv_queue
        self.send_queue = send_queue
        self.current_slot = spec.Slot(0)
        self.current_time = uint64(0)
        self.slot_last_attested = None
        self.attestation_cache = AttestationCache()
        self.should_quit = False
        self.event_cache = None
        self.specconf = (configpath, configname)
        self.index = None
        self.debug = debug
        self.debugfile = None
        self.colorstep = 2
        self.mycolor = '#7FFFFF'
        self.profile = False
        if self.counter % 14 == 0:
            self.profile = True

    def __debug(self, obj, typ: str):
        if not self.debug:
            return
        if not self.debugfile:
            self.debugfile = open(f'log_{self.index}_{int(time())}.json', 'w')
            json.dump({'level': 'begin',
                       'index': self.index,
                       'timestamp': self.current_time,
                       'slot': self.current_slot,
                       'message': f'My color is {self.mycolor}'}, self.debugfile)
        logobj = ''
        if isinstance(obj, Container):
            logobj = str(obj)
        elif isinstance(obj, dict):
            logobj = dict()
            for key, value in obj.items():
                logobj[key] = str(value)
        else:
            logobj = str(obj)
        json.dump(
            {'level': typ,
             'index': self.index,
             'timestamp': self.current_time,
             'slot': self.current_slot,
             'message': logobj}, self.debugfile)
        self.debugfile.write('\n')
        if 'Error' in typ:
            self.debugfile.flush()

    def __color(self, otheridx=None):
        idx = self.index if self.index is not None else self.counter
        if otheridx is not None:
            idx = otheridx
        return '#{:06x}'.format(0x7F7F7F + (self.colorstep * idx))

    def run(self):
        self.mycolor = self.__color()
        config_util.prepare_config(self.specconf[0], self.specconf[1])
        # noinspection PyTypeChecker
        reload(spec)
        spec.bls.bls_active = False
        if self.profile:
            pr = cProfile.Profile()
            pr.enable()
        while not self.should_quit:
            item = self.recv_queue.get()
            self.__handle_event(item)
            self.recv_queue.task_done()
        if self.profile:
            pr.disable()
            pr.dump_stats(f"profile_{self.index}_{time()}.prof")
        if self.debug:
            print(f'{self.index} Write Log file')
            self.debugfile.close()

    def __handle_event(self, event: Event):
        if event.time < self.current_time:
            self.send_queue.put(
                SimulationEndEvent(time=self.current_time,
                                   message=f"Own time {self.current_time} is later than event time {event.time}. {event}/{self.event_cache}")
            )
        self.event_cache = event
        self.current_time = event.time
        actions = {
            MessageEvent: self.__handle_message_event,
            SimulationEndEvent: self.__handle_simulation_end,
            LatestVoteOpportunity: self.handle_latest_voting_opportunity,
            AggregateOpportunity: self.handle_aggregate_opportunity,
            NextSlotEvent: self.handle_next_slot_event
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
        payload = decoder[event.message_type].decode_bytes(event.message)
        # noinspection PyArgumentList
        actions[event.message_type](payload)

    def handle_genesis_state(self, state: spec.BeaconState):
        self.state = state

    def handle_genesis_block(self, block: spec.BeaconBlock):
        self.index = spec.ValidatorIndex(max(
            genesis_validator[0]
            for genesis_validator in enumerate(self.state.validators)
            if genesis_validator[1].pubkey == self.pubkey
        ))
        self.store = spec.get_forkchoice_store(self.state, block)
        self.block_cache = BlockCache(block)
        self.head_root = spec.hash_tree_root(block)
        self.update_committee()

    def __handle_simulation_end(self, event: SimulationEndEvent):
        self.__debug(event, 'SimulationEnd')
        # Mark all remaining tasks as done
        next_task = queue_element_or_none(self.recv_queue)
        while next_task is not None:
            self.recv_queue.task_done()
            next_task = queue_element_or_none(self.recv_queue)
        self.should_quit = True
        if self.index == 60:
            self.graph()
        print(f'Goodbye from {self.index}!')

    @staticmethod
    def __init_keys(counter: int, keydir: Optional[str]) -> Tuple[int, BLSPubkey]:
        pubkey = None
        privkey = None
        pubkey_file = None
        privkey_file = None
        keys = None
        if keydir is not None:
            keys = Path(keydir)

        if keys.is_dir():
            pubkey_file = (keys / f'{counter}.pubkey')
            privkey_file = (keys / f'{counter}.privkey')
            if pubkey_file.is_file() and privkey_file.is_file():
                print('Read keys from file')
                pubkey = pubkey_file.read_bytes()
                privkey = int(privkey_file.read_text())
        if not privkey or not pubkey:
            print('Generate and save keys')
            privkey = random.randint(1, curve_order)
            pubkey = Bls.SkToPk(privkey)
            if keys is not None:
                assert isinstance(pubkey_file, Path)
                pubkey_file.write_bytes(pubkey)
                privkey_file.write_text(str(privkey))
        return privkey, pubkey

    def update_committee(self):
        self.committee = spec.get_committee_assignment(
            self.state,
            spec.compute_epoch_at_slot(self.current_slot),
            spec.ValidatorIndex(self.index)
        )

    def propose_block(self, head_state: spec.BeaconState, other=False):
        # If the attestation's target epoch is the current epoch,
        # the source checkpoint must match with the proposer's
        #
        # ep(target) == ep(now) => cp_justified == cp_source
        # <=> ep(target) != ep(now) v cp_justified == cp_source
        min_slot_to_include = spec.compute_start_slot_at_epoch(
            spec.compute_epoch_at_slot(self.current_slot)
        ) if self.current_slot > spec.SLOTS_PER_EPOCH else spec.Slot(0)
        max_slot_to_include = self.current_slot - 1
        candidate_attestations = tuple(
            attestation
            for attestation in self.attestation_cache.attestations_not_seen_in_block(
                min_slot_to_include,
                max_slot_to_include
            )
            if attestation.data.target.epoch != spec.get_current_epoch(self.state)
            or self.state.current_justified_checkpoint == attestation.data.source
        )
        block = spec.BeaconBlock(
            slot=head_state.slot,
            proposer_index=spec.ValidatorIndex(self.index),
            parent_root=self.head_root,
            state_root=spec.Bytes32(bytes(0 for _ in range(0, 32)))
        )
        randao_reveal = spec.get_epoch_signature(head_state, block, self.privkey)
        eth1_data = head_state.eth1_data

        proposer_slashings: List[spec.ProposerSlashing, spec.MAX_PROPOSER_SLASHINGS] = list(
            slashing for slashing in self.block_cache.search_slashings()
            if spec.is_slashable_validator(
                head_state.validators[slashing.signed_header_1.message.proposer_index],
                spec.get_current_epoch(head_state)
            )
        )
        attester_slashings: List[spec.AttesterSlashing, spec.MAX_ATTESTER_SLASHINGS] = list(
            slashing for slashing in self.attestation_cache.search_slashings()
            if all(
                spec.is_slashable_validator(head_state.validators[vindex], spec.get_current_epoch(head_state))
                for vindex in set(slashing.attestation_1.attesting_indices)
                    .intersection(set(slashing.attestation_2.attesting_indices)))
        )
        if len(proposer_slashings) > spec.MAX_PROPOSER_SLASHINGS:
            proposer_slashings = proposer_slashings[0:spec.MAX_PROPOSER_SLASHINGS]
        if len(attester_slashings) > spec.MAX_ATTESTER_SLASHINGS:
            attester_slashings = attester_slashings[0:spec.MAX_ATTESTER_SLASHINGS]
        if len(candidate_attestations) > spec.MAX_ATTESTATIONS:
            candidate_attestations = candidate_attestations[0:spec.MAX_ATTESTATIONS]
        if other:
            candidate_attestations = candidate_attestations[0:len(candidate_attestations) - 2]

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
        except AssertionError as e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            self.__debug(text, 'FindNewStateRootError')
            self.send_queue.put(SimulationEndEvent(time=self.current_time))
            return
        block.state_root = new_state_root
        signed_block = spec.SignedBeaconBlock(
            message=block,
            signature=spec.get_block_signature(self.state, block, self.privkey)
        )
        encoded_signed_block = signed_block.encode_bytes()

        self.send_queue.put(MessageEvent(
            time=self.current_time,
            message=encoded_signed_block,
            message_type='SignedBeaconBlock',
            fromidx=self.index,
            toidx=None
        ))
        self.graph(show=False)
        self.__debug(signed_block, 'ProposeBlock')

    def handle_next_slot_event(self, message: NextSlotEvent):
        self.current_slot = spec.Slot(message.slot)

        spec.on_tick(self.store, message.time)
        if message.slot % spec.SLOTS_PER_EPOCH == 0:
            self.update_committee()

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
                print(f'[VALIDATOR {self.index}] Could not validate attestation')
                self.__debug(attestation, 'ValidateAttestationOnSlotBoundaryError')

        # Advance state for checking
        self.__compute_head()
        head_state = self.state.copy()
        if head_state.slot < message.slot:
            spec.process_slots(head_state, spec.Slot(message.slot))
        self.proposer_current_slot = spec.get_beacon_proposer_index(head_state)

        # Propose block if needed
        if spec.ValidatorIndex(self.index) == self.proposer_current_slot and not self.__slashed():
            print(f"[VALIDATOR {self.index}] I'M PROPOSER!!!")
            self.propose_block(head_state)
            if message.slot == 4:
                print('YOLO')
                self.propose_block(head_state, True)

    def handle_latest_voting_opportunity(self, message: LatestVoteOpportunity):
        if self.slot_last_attested is None:
            self.attest()
            return
        if self.slot_last_attested < message.slot:
            self.attest()

    def handle_aggregate_opportunity(self, message: AggregateOpportunity):
        if not self.slot_last_attested == message.slot:
            return
        slot_signature = spec.get_slot_signature(self.state, spec.Slot(message.slot), self.privkey)
        if not spec.is_aggregator(self.state, spec.Slot(message.slot), self.committee[1], slot_signature):
            return
        attestations_to_aggregate = [
            attestationcache.attestation
            for attestationcache
            in self.attestation_cache.cache_by_time[spec.Slot(message.slot)][self.committee[1]]
            if attestationcache.attestation.data == self.last_attestation_data
            and len(attestationcache.attesting_indices) == 1
        ]
        aggregation_bits = list(0 for _ in range(len(self.committee[0])))

        for attestation in attestations_to_aggregate:
            for idx, bit in enumerate(attestation.aggregation_bits):
                if bit:
                    aggregation_bits[idx] = 1
        # noinspection PyArgumentList
        aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*aggregation_bits)

        attestation = spec.Attestation(
            aggregation_bits=aggregation_bits,
            data=self.last_attestation_data,
            signature=spec.get_aggregate_signature(attestations_to_aggregate)
        )
        aggregate_and_proof = spec.get_aggregate_and_proof(
            self.state, spec.ValidatorIndex(self.index), attestation, self.privkey
        )
        aggregate_and_proof_signature = spec.get_aggregate_and_proof_signature(
            self.state, aggregate_and_proof, self.privkey
        )
        signed_aggregate_and_proof = spec.SignedAggregateAndProof(
            message=aggregate_and_proof,
            signature=aggregate_and_proof_signature
        )
        encoded_aggregate_and_proof = signed_aggregate_and_proof.encode_bytes()
        self.send_queue.put(MessageEvent(
            time=self.current_time,
            message=encoded_aggregate_and_proof,
            message_type='SignedAggregateAndProof',
            fromidx=self.index,
            toidx=None
        ))
        self.__debug(signed_aggregate_and_proof, 'AggregateAndProofSend')

    def handle_attestation(self, attestation: spec.Attestation):
        self.__debug(attestation, 'AttestationRecv')
        self.attestation_cache.add_attestation(attestation, self.state)

    def handle_aggregate(self, aggregate: spec.SignedAggregateAndProof):
        # TODO check signature
        self.handle_attestation(aggregate.message.aggregate)

    def handle_block(self, block: spec.SignedBeaconBlock):
        # Try to process an old chains first
        old_chain = self.block_cache.longest_outstanding_chain(self.store)
        if len(old_chain) > 0:
            self.__debug(len(old_chain), 'OrphanedBlocksToHandle')
        for cblock in old_chain:
            spec.on_block(self.store, cblock)
            self.block_cache.accept_block(cblock)

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
            print(f'[VALIDATOR {self.index}] Could not validate block: {e}')
            chain = []
        self.__debug(str(len(chain)), 'BlocksToHandle')
        for cblock in chain:
            try:
                spec.on_block(self.store, cblock)
                self.block_cache.accept_block(cblock)
                for aslashing in cblock.message.body.attester_slashings:
                    self.attestation_cache.accept_slashing(aslashing)
                for pslashing in cblock.message.body.proposer_slashings:
                    print(f"!!!!!!!! SLASHING DETECTED {pslashing} !!!!!!!")
                    self.block_cache.accept_slashing(pslashing)
                for attestation in block.message.body.attestations:
                    self.attestation_cache.add_attestation(attestation, self.state, seen_in_block=True)
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]
                print(f'[VALIDATOR {self.index}] Could not validate block (assert line {line}, stmt {text}! {str(block.message)}!')

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
        if self.__slashed():
            return
        if self.committee is None:
            self.update_committee()
        if not (self.current_slot == self.committee[2] and self.index in self.committee[0]):
            return
        self.__compute_head()
        head_block = self.store.blocks[self.head_root]
        head_state = self.state.copy()
        if head_state.slot < self.current_slot:
            spec.process_slots(head_state, self.current_slot)

        # From validator spec / Attesting / Note
        start_slot = spec.compute_start_slot_at_epoch(spec.get_current_epoch(head_state))
        epoch_boundary_block_root = spec.hash_tree_root(head_block) \
            if start_slot == head_state.slot \
            else spec.get_block_root(head_state, spec.get_current_epoch(head_state))

        attestation_data = spec.AttestationData(
            slot=self.committee[2],
            index=self.committee[1],
            beacon_block_root=spec.hash_tree_root(head_block),
            source=head_state.current_justified_checkpoint,
            target=spec.Checkpoint(epoch=spec.get_current_epoch(head_state), root=epoch_boundary_block_root)
        )

        aggregation_bits_list = list(0 for _ in range(0, len(self.committee[0])))
        aggregation_bits_list[self.committee[0].index(self.index)] = 1
        # noinspection PyArgumentList
        aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*aggregation_bits_list)

        attestation = spec.Attestation(
            data=attestation_data,
            aggregation_bits=aggregation_bits,
            signature=spec.get_attestation_signature(self.state, attestation_data, self.privkey)
        )

        self.attestation_cache.add_attestation(attestation, self.state)
        self.last_attestation_data = attestation_data
        self.slot_last_attested = head_state.slot

        encoded_attestation = attestation.encode_bytes()
        message = MessageEvent(
            time=self.current_time,
            message=encoded_attestation,
            message_type='Attestation',
            fromidx=self.index,
            toidx=None
        )
        self.send_queue.put(message)
        self.__debug(attestation, 'AttestationSend')
        target = attestation.data.target
        target_slot = spec.compute_start_slot_at_epoch(target.epoch)

    def __compute_head(self):
        self.head_root = spec.get_head(self.store)
        self.state = self.store.block_states[self.head_root]

    def __slashed(self) -> bool:
        return self.state.validators[self.index].slashed

    def graph(self, show=True):
        g = Digraph('G', filename=f'graph_{int(time())}_{self.index}.gv')
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
                    #print(f'Epoch {epoch} is justified (latest justified is {str(self.store.justified_checkpoint)}')
                if epoch <= finalized_epoch:
                    epoch_style = 'bold'
                    epoch_color = 'black'
                    #print(f'Epoch {epoch} is finalized (latest finalized is {str(self.store.finalized_checkpoint)}')
                e.attr(color=epoch_color, style=epoch_style)
                for slot in blocks_by_slot_and_epoch[epoch].keys():
                    with e.subgraph(name=f'cluster_slot_{int(slot)}') as s:
                        s.attr(label=f'Slot {int(slot)}', style='dashed')
                        # print(f'At slot: {int(slot)}')
                        for root, block in blocks_by_slot_and_epoch[epoch][slot]:
                            color = self.__color(block.proposer_index)
                            shape = 'box'
                            if root == head_root:
                                shape = 'octagon'
                            if root == justified_root:
                                shape = 'doubleoctagon'
                                #print(f'{root} at {block.slot} is justified')
                            if root == finalized_root:
                                shape = 'tripleoctagon'
                                #print(f'{root} at {block.slot} is finalized')
                            #print(f'{str(root)} at {str(slot)}={str(block.slot)} by {str(block.proposer_index)}')
                            s.node(str(root), shape=shape, color=color, style='filled')
        for epoch in blocks_by_slot_and_epoch.keys():
            for slot in blocks_by_slot_and_epoch[epoch].keys():
                for root, block in blocks_by_slot_and_epoch[epoch][slot]:
                    g.edge(str(root), str(block.parent_root))
        for validator, message in self.store.latest_messages.items():
            color = self.__color(validator)
            g.node(str(validator), shape='circle', color=color, style='filled')
            g.edge(str(validator), str(message.root))
        # print(blocks_by_slot_and_epoch)

        g.save()
        if show:
            g.view()
