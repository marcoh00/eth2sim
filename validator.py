from importlib import reload
from typing import Tuple, Sequence, Optional, List

from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist
from eth_typing import BLSPubkey

import eth2spec.phase0.spec as spec
from cache import AttestationCache
from eth2spec.config import config_util
from events import NextSlotEvent, LatestVoteOpportunity, AggregateOpportunity, MessageEvent, MESSAGE_TYPE
from helpers import popcnt, PickleableStore


class Validator:
    index: uint64
    privkey: int
    pubkey: BLSPubkey
    state: spec.BeaconState
    store: spec.Store

    committee: Optional[Tuple[Sequence[spec.ValidatorIndex], spec.CommitteeIndex, spec.Slot]]
    slot_last_attested: Optional[spec.Slot]
    last_attestation_data: spec.AttestationData

    current_slot: spec.Slot
    proposer_current_slot: Optional[spec.ValidatorIndex]

    attestation_cache: AttestationCache

    def __init__(self, index, privkey, pubkey):
        self.index = index
        self.privkey = privkey
        self.pubkey = pubkey

        self.state = spec.BeaconState()
        self.committee = None
        self.slot_last_attested = None
        self.current_slot = spec.Slot(0)
        self.attestation_cache = AttestationCache()
        self.proposer_current_slot = None

    def update_committee(self):
        self.committee = spec.get_committee_assignment(
            self.state,
            spec.compute_epoch_at_slot(self.current_slot),
            spec.ValidatorIndex(self.index)
        )

    def propose_block(self, head_state: spec.BeaconState)\
            -> Optional[Tuple[spec.ValidatorIndex, Optional[Sequence[spec.ValidatorIndex]], MESSAGE_TYPE]]:
        head = spec.get_head(self.store)
        candidate_attestations = tuple(
            self.attestation_cache.all_attestations_with_unseen_validators(spec.get_current_slot(self.store) - 1)
        )
        block = spec.BeaconBlock(
            slot=head_state.slot,
            proposer_index=spec.ValidatorIndex(self.index),
            parent_root=head,
            state_root=spec.Bytes32(bytes(0 for _ in range(0, 32)))
        )
        randao_reveal = spec.get_epoch_signature(head_state, block, self.privkey)
        eth1_data = head_state.eth1_data

        # TODO implement slashings
        proposer_slashings: List[spec.ProposerSlashing, spec.MAX_PROPOSER_SLASHINGS] = list()
        attester_slashings: List[spec.AttesterSlashing, spec.MAX_ATTESTER_SLASHINGS] = list()
        if len(candidate_attestations) <= spec.MAX_ATTESTATIONS:
            attestations = candidate_attestations
        else:
            attestations = candidate_attestations[0:spec.MAX_ATTESTATIONS]

        deposits: List[spec.Deposit, spec.MAX_DEPOSITS] = list()
        voluntary_exits: List[spec.VoluntaryExit, spec.MAX_VOLUNTARY_EXITS] = list()

        body = spec.BeaconBlockBody(
            randao_reveal=randao_reveal,
            eth1_data=eth1_data,
            graffiti=spec.Bytes32(bytes(255 for _ in range(0, 32))),
            proposer_slashings=proposer_slashings,
            attester_slashings=attester_slashings,
            attestations=attestations,
            deposits=deposits,
            voluntary_exits=voluntary_exits
        )
        block.body = body
        new_state_root = spec.compute_new_state_root(self.state, block)
        block.state_root = new_state_root
        signed_block = spec.SignedBeaconBlock(
            message=block,
            signature=spec.get_block_signature(self.state, block, self.privkey)
        )
        return spec.ValidatorIndex(self.index), None, signed_block

    def handle_next_slot_event(self, message: NextSlotEvent, head_state=None)\
            -> Optional[Tuple[spec.ValidatorIndex, Optional[Sequence[spec.ValidatorIndex]], MESSAGE_TYPE]]:
        if head_state is None:
            self.pre_slot_update(message)
            head_state = self.internal_slot_state_transition(message)
        return self.post_slot_state_update(message, head_state)

    def pre_slot_update(self, event: NextSlotEvent):
        self.current_slot = event.slot
        # Keep one epoch of past attestations
        self.attestation_cache.cleanup(event.slot, uint64(spec.SLOTS_PER_EPOCH))

    def internal_slot_state_transition(self, event: NextSlotEvent) -> spec.BeaconState:
        spec.on_tick(self.store, event.time)
        for attestation in self.attestation_cache.attestations(event.slot - 1):
            try:
                spec.on_attestation(self.store, attestation[2])
            except AssertionError:
                print(f"COULD FINALLY NOT VALIDATE ATTESTATION {attestation} !!!")
        head_state = self.state.copy()
        if head_state.slot < event.slot:
            spec.process_slots(head_state, event.slot)
        self.proposer_current_slot = spec.get_beacon_proposer_index(head_state)
        return head_state

    def post_slot_state_update(self, event: NextSlotEvent, head_state: spec.BeaconState)\
            -> Optional[Tuple[spec.ValidatorIndex, Optional[Sequence[spec.ValidatorIndex]], MESSAGE_TYPE]]:
        if event.slot % spec.SLOTS_PER_EPOCH == 0:
            self.update_committee()

        # Propose block if needed
        if spec.ValidatorIndex(self.index) == self.proposer_current_slot:
            print(f"[VALIDATOR {self.index}] I'M PROPOSER!!!")
            return self.propose_block(head_state)
        return None

    def handle_latest_voting_opportunity(self, message: LatestVoteOpportunity)\
            -> Optional[Tuple[spec.ValidatorIndex, Optional[Sequence[spec.ValidatorIndex]], MESSAGE_TYPE]]:
        self.current_slot = message.slot
        if self.slot_last_attested is None:
            return self.attest()
        if self.slot_last_attested < message.slot:
            return self.attest()
        return None

    def handle_aggregate_opportunity(self, message: AggregateOpportunity)\
            -> Optional[Tuple[spec.ValidatorIndex, Optional[Sequence[spec.ValidatorIndex]], MESSAGE_TYPE]]:
        self.current_slot = message.slot
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
        # noinspection PyArgumentList
        aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*aggregation_bits)
        if not popcnt(aggregation_bits) == 4:
            print('WARNING')

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
        return spec.ValidatorIndex(self.index), None, signed_aggregate_and_proof

    def handle_attestation(self, attestation: spec.Attestation):
        previous_epoch = spec.get_previous_epoch(self.state)
        attestation_epoch = spec.compute_epoch_at_slot(attestation.data.slot)
        if attestation_epoch >= previous_epoch:
            self.attestation_cache.add_attestation(attestation)
        else:
            print(f"[WARNING] Validator {self.index} received Attestation for the past")
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

    def handle_message_event(self, message: MessageEvent):
        actions = {
            spec.Attestation: self.handle_attestation,
            spec.SignedAggregateAndProof: self.handle_aggregate,
            spec.SignedBeaconBlock: self.handle_block
        }
        # noinspection PyArgumentList
        actions[type(message.message)](message.message)

    def attest(self) -> Optional[Tuple[spec.ValidatorIndex, Optional[Sequence[spec.ValidatorIndex]], MESSAGE_TYPE]]:
        if not self.committee:
            self.update_committee()
        if not (self.current_slot == self.committee[2] and self.index in self.committee[0]):
            return None
        head_block = self.store.blocks[spec.get_head(self.store)]
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

        self.attestation_cache.add_attestation(attestation)
        self.last_attestation_data = attestation_data
        self.slot_last_attested = head_state.slot
        return spec.ValidatorIndex(self.index), None, attestation

    @staticmethod
    def test_encode_decode_state(state: spec.BeaconState) -> spec.BeaconState:
        state = state.encode_bytes()
        result = spec.BeaconState.decode_bytes(state)
        return result

    @staticmethod
    def test_encode_decode_store(state: spec.Store) -> spec.Store:
        state = PickleableStore.from_store(state)
        result = state.into_store()
        return result


def offload_handle_block(block: spec.SignedBeaconBlock, state: spec.BeaconState, store: spec.Store) \
        -> Tuple[spec.BeaconState, spec.Store]:
    spec.on_block(store, block)
    spec.state_transition(state, block)
    return state, store


def encode_offload_slot_arguments(validator: Validator, event: NextSlotEvent)\
        -> Tuple[NextSlotEvent, Sequence[bytes], bytes, PickleableStore]:
    return (
        event,
        tuple(
            attestation[2].encode_bytes() for attestation in validator.attestation_cache.attestations(event.slot - 1)
        ),
        validator.state.encode_bytes(),
        PickleableStore.from_store(validator.store)
    )


def decode_offload_slot_arguments(args: Tuple[NextSlotEvent, Sequence[bytes], bytes, PickleableStore])\
        -> Tuple[NextSlotEvent, Sequence[spec.Attestation], spec.BeaconState, spec.Store]:
    return (
        args[0],
        tuple(spec.Attestation.decode_bytes(b) for b in args[1]),
        spec.BeaconState.decode_bytes(args[2]),
        PickleableStore.into_store(args[3])
    )


def encode_offload_slot_results(head_state: spec.BeaconState,
                                store: spec.Store,
                                proposer_slot: Optional[spec.ValidatorIndex])\
        -> Tuple[bytes, PickleableStore, Optional[bytes]]:
    return (
        head_state.encode_bytes(),
        PickleableStore.from_store(store),
        proposer_slot.encode_bytes() if proposer_slot is not None else None
    )


def decode_offload_slot_results(args: Tuple[bytes, PickleableStore, Optional[bytes]])\
        -> Tuple[spec.BeaconState, spec.Store, Optional[spec.ValidatorIndex]]:
    return (
        spec.BeaconState.decode_bytes(args[0]),
        args[1].into_store(),
        spec.ValidatorIndex.decode_bytes(args[2]) if args[2] is not None else None
    )


def apply_offload_slot_results_and_get_head_state(validator: Validator,
                                                  args: Tuple[bytes, PickleableStore, Optional[bytes]])\
        -> spec.BeaconState:
    state, store, proposer_index = decode_offload_slot_results(args)
    validator.store = store
    validator.proposer_current_slot = proposer_index
    return state


def offload_slot_state_transition(args: Tuple[NextSlotEvent, Sequence[bytes], bytes, PickleableStore]) \
        -> Tuple[bytes, PickleableStore, Optional[bytes]]:
    event, attestations, state, store = decode_offload_slot_arguments(args)
    spec.on_tick(store, event.time)
    for attestation in attestations:
        try:
            spec.on_attestation(store, attestation)
        except AssertionError:
            print(f"COULD FINALLY NOT VALIDATE ATTESTATION {attestation} !!!")
    head_state = state.copy()
    if head_state.slot < event.slot:
        spec.process_slots(head_state, event.slot)
    proposer_current_slot = spec.get_beacon_proposer_index(head_state)
    return encode_offload_slot_results(head_state, store, proposer_current_slot)


def encode_offload_block_arguments(block: spec.SignedBeaconBlock, state: spec.BeaconState, store: spec.Store)\
        -> Tuple[bytes, bytes, PickleableStore]:
    return block.encode_bytes(), state.encode_bytes(), PickleableStore.from_store(store)


def decode_offload_block_arguments(args: Tuple[bytes, bytes, PickleableStore])\
        -> Tuple[spec.SignedBeaconBlock, spec.BeaconState, spec.Store]:
    return (
        spec.SignedBeaconBlock.decode_bytes(args[0]),
        spec.BeaconState.decode_bytes(args[1]),
        args[2].into_store()
    )


def decode_offload_block_results(args: Tuple[bytes, PickleableStore]) -> Tuple[spec.BeaconState, spec.Store]:
    return spec.BeaconState.decode_bytes(args[0]), args[1].into_store()


def offload_block_processing(args: Tuple[Sequence[bytes], Sequence[bytes], PickleableStore])\
        -> Tuple[bytes, PickleableStore]:
    block, state, store = decode_offload_block_arguments(args)
    spec.on_block(store, block)
    spec.state_transition(state, block)
    return state.encode_bytes(), PickleableStore.from_store(store)
