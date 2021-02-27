import pathlib
import random
from typing import Tuple, Sequence, Optional, Dict, List

from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist
from remerkleable.complex import Container

import eth2spec.phase0.spec as spec

from py_ecc.bls import G2ProofOfPossession as Bls
from py_ecc.bls12_381 import curve_order
from eth_typing import BLSSignature, BLSPubkey

import eth2spec.phase0.spec as spec
import eth2spec.test.utils as utils
from cache import AttestationCache
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, MessageEvent
from helpers import popcnt


class Validator:
    index: uint64
    privkey: int
    pubkey: BLSPubkey
    state: spec.BeaconState
    store: spec.Store

    committee: Optional[Tuple[Sequence[spec.ValidatorIndex], spec.CommitteeIndex, spec.Slot]]
    slot_last_attested: spec.Slot
    last_attestation_data: spec.AttestationData
    current_slot: spec.Slot

    attestation_cache: AttestationCache

    def __init__(self, simulator, index, privkey, pubkey):
        self.simulator = simulator
        self.index = index
        self.privkey = privkey
        self.pubkey = pubkey

        self.state = spec.BeaconState()
        self.store = None
        self.committee = None
        self.slot_last_attested = None
        self.current_slot = spec.Slot(0)
        self.attestation_cache = AttestationCache()

    def update_committee(self):
        self.committee = spec.get_committee_assignment(
            self.state,
            spec.get_current_epoch(self.state),
            spec.ValidatorIndex(self.index)
        )

    def propose_block(self):
        head = spec.get_head(self.store)
        attestations = tuple(
            self.attestation_cache.all_attestations_with_unseen_validators(spec.get_current_slot(self.store) - 1)
        )
        pass

    def handle_next_slot_event(self, message: NextSlotEvent):
        spec.on_tick(self.store, message.time)

        # Keep one epoch of past attestations
        self.attestation_cache.cleanup(message.slot, uint64(spec.SLOTS_PER_EPOCH))

        # Handle last slot's attestations
        for slot, committee, attestation in self.attestation_cache.attestations(message.slot - 1):
            try:
                spec.on_attestation(self.store, attestation)
                # print(f"[VALIDATOR {self.index}] ACKd Attestation by {attestation.data.index}/{attestation.aggregation_bits} for slot {attestation.data.slot}")
            except AssertionError:
                print(f"COULD FINALLY NOT VALIDATE ATTESTATION {attestation} !!!")
                self.attestation_cache.remove_attestation(slot, committee, attestation)

        # Propose block if needed
        if spec.is_proposer(self.state, spec.ValidatorIndex(self.index)):
            print(f"[VALIDATOR {self.index}] I'M PROPOSER!!!")
            self.propose_block()


    def handle_last_voting_opportunity(self, message: LatestVoteOpportunity):
        if self.slot_last_attested and self.slot_last_attested < message.slot:
            self.attest()
        if not self.slot_last_attested:
            self.attest()

    def handle_aggregate_opportunity(self, message: AggregateOpportunity):
        if not self.slot_last_attested == message.slot:
            return
        slot_signature = spec.get_slot_signature(self.state, message.slot, self.privkey)
        if not spec.is_aggregator(self.state, message.slot, self.committee[1], slot_signature):
            return
        attestations_to_aggregate = [
            attestation
            for attestation
            in self.attestation_cache.attestation_cache[message.slot][self.committee[1]]
            if attestation.data == self.last_attestation_data and popcnt(attestation.aggregation_bits) == 1
        ]
        aggregation_bits = list(0 for _ in range(len(self.committee[0])))

        for attestation in attestations_to_aggregate:
            for idx, bit in enumerate(attestation.aggregation_bits):
                if bit:
                    aggregation_bits[idx] = 1
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
        self.simulator.network.send(signed_aggregate_and_proof, self.index, None)

    def handle_attestation(self, attestation: spec.Attestation):
        print(f'[Validator {self.index}] ATTESTATION (from {attestation.aggregation_bits})')
        self.attestation_cache.add_attestation(attestation)
        if spec.get_current_slot(self.store) > attestation.data.slot:
            # Only validate attestations for previous epochs
            # Attestations for the current epoch are validated on slot boundaries
            spec.on_attestation(self.store, attestation)

    def handle_aggregate(self, aggregate: spec.SignedAggregateAndProof):
        # TODO check signature
        self.handle_attestation(aggregate.message.aggregate)

    def handle_block(self, block: spec.SignedBeaconBlock):
        spec.on_block(self.store, block)
        spec.state_transition(self.state, block)
        for attestation in block.message.body.attestations:
            self.attestation_cache.add_attestation(attestation, from_block=True)
        print(f"Handled block {block.message.state_root}")

    def handle_message_event(self, message: MessageEvent):
        actions = {
            spec.Attestation: self.handle_attestation,
            spec.SignedAggregateAndProof: self.handle_aggregate,
            spec.SignedBeaconBlock: self.handle_block
        }
        actions[type(message.message)](message.message)

    def attest(self):
        if not self.committee:
            self.update_committee()
        if not (self.state.slot == self.committee[2] and self.index in self.committee[0]):
            return None
        head_block = self.store.blocks[spec.get_head(self.store)]
        head_state = self.state.copy()
        while head_state.slot < head_block.slot:
            spec.process_slot(head_state)

        # From validator spec / Attesting / Note
        start_slot = spec.compute_start_slot_at_epoch(spec.get_current_epoch(head_state))
        epoch_boundary_block_root = spec.hash_tree_root(head_block) \
            if start_slot == head_state.slot \
            else spec.get_block_root(self.state, spec.get_current_epoch(head_state))

        attestation_data = spec.AttestationData(
            slot=self.committee[2],
            index=self.committee[1],
            beacon_block_root=spec.hash_tree_root(head_block),
            source=head_state.current_justified_checkpoint,
            target=spec.Checkpoint(epoch=spec.get_current_epoch(head_state), root=epoch_boundary_block_root)
        )

        aggregation_bits_list = list(0 for _ in range(0, len(self.committee[0])))
        aggregation_bits_list[self.committee[0].index(self.index)] = 1
        aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*aggregation_bits_list)

        attestation = spec.Attestation(
            data=attestation_data,
            aggregation_bits=aggregation_bits,
            signature=spec.get_attestation_signature(self.state, attestation_data, self.privkey)
        )

        self.last_attestation_data = attestation_data
        self.slot_last_attested = self.state.slot
        self.simulator.network.send(attestation, self.index, None)

    def save_state(self, outdir: pathlib.Path):
        with open(outdir / f'{self.index}.state', 'wb') as statefp:
            statefp.write(self.state.encode_bytes())
        with open(outdir / f'{self.index}.store', 'wb') as storefp:
            storefp.write(Container.encode_bytes(self.store))

    def load_state(self, dir: pathlib.Path):
        with open(dir / f'{self.index}.state', 'rb') as statefp:
            self.state = spec.BeaconState.decode_bytes(statefp.read())
        # with open(dir / f'{self.index}.store', 'rb') as storefp:
        #     self.state = Container.decode_bytes(spec.Store, storefp.read())
