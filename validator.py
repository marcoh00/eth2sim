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
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, MessageEvent


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

    attestation_cache: Dict[spec.Slot, Dict[spec.CommitteeIndex, List[spec.Attestation]]]

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
        self.attestation_cache = dict()

    def update_committee(self):
        self.committee = spec.get_committee_assignment(
            self.state,
            spec.get_current_epoch(self.state),
            spec.ValidatorIndex(self.index)
        )

    def handle_next_slot_event(self, message: NextSlotEvent):
        spec.on_tick(self.store, spec.MIN_GENESIS_TIME + spec.GENESIS_DELAY + message.time)
        # Handle last slot's attestations
        for slot, committee_attestation_mapping in self.attestation_cache:
            if slot < message.slot:
                for committee, attestations in committee_attestation_mapping:
                    remove_attestation = list()
                    for attestation in attestations:
                        try:
                            spec.on_attestation(self.store, attestation)
                            bits = 0
                            for bit in attestation.aggregation_bits:
                                if bit:
                                    bits += 1
                            if bits <= 1:
                                remove_attestation.append(attestation)
                        except AssertionError:
                            print(f"COULD FINALLY NOT VALIDATE ATTESTATION {attestation} !!!")
                            remove_attestation.append(attestation)
                        assert isinstance(attestations, list)
                        remove_attestation.append(attestation)
                    for remove in remove_attestation:
                        attestations.remove(remove)

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
            in self.attestation_cache[message.slot][self.committee[1]]
            if attestation.data == self.last_attestation_data
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
        current_slot = spec.get_current_slot(self.store)
        if current_slot not in self.attestation_cache:
            self.attestation_cache[current_slot] = dict()
        if attestation.data.index not in self.attestation_cache[current_slot]:
            self.attestation_cache[current_slot][attestation.data.index] = list()
        self.attestation_cache[current_slot][attestation.data.index].append(attestation)
        try:
            spec.on_attestation(self.store, attestation)
        except AssertionError as e:
            # Attestations from current slot only become relevant in the next slot
            # Try again when the next slot arrives
            print(f'[Validator {self.index}] ATTESTATION INVALID ({str(e)})')

    def handle_aggregate(self, aggregate: spec.AggregateAndProof):
        pass

    def handle_block(self, block: spec.BeaconBlock):
        pass

    def handle_message_event(self, message: MessageEvent):
        actions = {
            spec.Attestation: self.handle_attestation,
            spec.SignedAggregateAndProof: self.handle_aggregate,
            spec.BeaconBlock: self.handle_block
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
