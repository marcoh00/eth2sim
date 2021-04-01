from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Sequence, Iterable, Tuple

import remerkleable
from remerkleable.basic import uint64

from eth2spec.phase0 import spec
from eth2spec.utils.ssz import ssz_typing
from helpers import popcnt, indices_inside_committee


@dataclass
class CachedAttestation:
    attesting_indices: ssz_typing.List[spec.ValidatorIndex, spec.MAX_VALIDATORS_PER_COMMITTEE]
    seen_in_block: bool
    known_to_forkchoice: bool
    attestation: spec.Attestation

    def into_indexed_attestation(self) -> spec.IndexedAttestation:
        return spec.IndexedAttestation(
            attesting_indices=self.attesting_indices,
            data=self.attestation.data,
            signature=self.attestation.signature
        )

    @classmethod
    def from_attestation(cls, attestation: spec.Attestation, state: spec.BeaconState):
        indexed_attestation = spec.get_indexed_attestation(state, attestation)
        return cls(
            attesting_indices=indexed_attestation.attesting_indices,
            seen_in_block=False,
            known_to_forkchoice=False,
            attestation=attestation
        )


class AttestationCache:
    cache_by_time: Dict[spec.Slot, Dict[spec.CommitteeIndex, List[CachedAttestation]]]
    cache_by_validator: Dict[spec.ValidatorIndex, List[CachedAttestation]]
    accepted_attester_slashings: Dict[spec.ValidatorIndex, List[Tuple[spec.AttestationData, spec.AttestationData]]]

    def __init__(self):
        self.cache_by_time = dict()
        self.cache_by_validator = dict()
        self.accepted_attester_slashings = dict()

    def add_attestation(self, attestation: spec.Attestation, state: spec.BeaconState, seen_in_block=False):
        existing_cached_attestation = self.__find_attestaion(attestation)
        if existing_cached_attestation is not None:
            if seen_in_block:
                existing_cached_attestation.seen_in_block = True
            return
        cached_attestation = CachedAttestation.from_attestation(attestation, state)
        cached_attestation.seen_in_block = seen_in_block
        slot, committee = attestation.data.slot, attestation.data.index

        self.__ensure_key_exists(self.cache_by_time, slot, dict)
        self.__ensure_key_exists(self.cache_by_time[slot], committee, list)
        self.cache_by_time[slot][committee].append(cached_attestation)
        for validator in cached_attestation.attesting_indices:
            self.__ensure_key_exists(self.cache_by_validator, validator, list)
            self.cache_by_validator[validator].append(cached_attestation)

    def accept_attestation(self, attestation: spec.Attestation, forkchoice=False, block=False):
        cached_attestation = self.__find_attestaion(attestation)
        if forkchoice:
            cached_attestation.known_to_forkchoice = True
        if block:
            cached_attestation.seen_in_block = True

    def accept_slashing(self, slashing: spec.AttesterSlashing):
        validators = set(slashing.attestation_1.attesting_indices)\
            .intersection(set(slashing.attestation_2.attesting_indices))
        for validator in validators:
            self.__ensure_key_exists(self.accepted_attester_slashings, validator, list())
            self.accepted_attester_slashings[validator].append(
                (slashing.attestation_1.data, slashing.attestation_2.data)
            )

    def attestations_not_known_to_forkchoice(self, min_slot: spec.Slot, max_slot: spec.Slot)\
            -> Iterable[spec.Attestation]:
        yield from self.filter_attestations(min_slot, max_slot, lambda a: not a.known_to_forkchoice)

    def attestations_not_seen_in_block(self, min_slot: spec.Slot, max_slot: spec.Slot) -> Iterable[spec.Attestation]:
        yield from self.filter_attestations(min_slot, max_slot, lambda a: not a.seen_in_block)

    def filter_attestations(self, min_slot: spec.Slot, max_slot: spec.Slot, filter_func) -> Iterable[spec.Attestation]:
        for slot in self.cache_by_time.keys():
            if min_slot <= slot <= max_slot:
                for committee, attestations in self.cache_by_time[slot].items():
                    for attestation in attestations:
                        if filter_func(attestation):
                            yield attestation.attestation

    def cleanup_time_cache(self, min_slot: spec.Slot):
        self.cleanup_old_time_cache_attestations(min_slot)
        self.cleanup_redundant_time_cache_attestations()

    def cleanup_redundant_time_cache_attestations(self):
        for slot, committee_list_dict in self.cache_by_time.items():
            for committee in committee_list_dict.keys():
                attestations = self.cache_by_time[slot][committee]
                attestations.sort(key=lambda a: len(a.attesting_indices), reverse=True)
                kept_attestations = []
                validators_seen_in_kept_attestations = set()
                for attestation in attestations:
                    # Here be dragons. We dont compare fork choice-known or seen-in-block fields yet!
                    if any(validator not in validators_seen_in_kept_attestations for validator in attestation.attesting_indices):
                        kept_attestations.append(attestation)
                        for validator in attestation.attesting_indices:
                            validators_seen_in_kept_attestations.add(validator)
                self.cache_by_time[slot][committee] = kept_attestations

    def cleanup_old_time_cache_attestations(self, min_slot: spec.Slot):
        slots_to_clean = tuple(slot for slot in self.cache_by_time.keys() if slot < min_slot)
        for slot in slots_to_clean:
            del self.cache_by_time[slot]

    def search_slashings(self, validator: Optional[spec.ValidatorIndex] = None):
        if validator is not None:
            yield from self.__search_slashings(validator)
        else:
            for validator in self.cache_by_validator.keys():
                yield from self.__search_slashings(validator)

    def __search_slashings(self, validator: spec.ValidatorIndex) -> Iterable[spec.AttesterSlashing]:
        for i in range(len(self.cache_by_validator[validator])):
            for j in range(i, len(self.cache_by_validator[validator])):
                attestation1 = self.cache_by_validator[validator][i]
                attestation2 = self.cache_by_validator[validator][j]
                if spec.is_slashable_attestation_data(
                    attestation1.attestation.data,
                    attestation2.attestation.data
                ):
                    if not self.__already_slashed(validator, attestation1, attestation2):
                        yield spec.AttesterSlashing(
                            attestation1=attestation1.into_indexed_attestation(),
                            attestation2=attestation2.into_indexed_attestation()
                        )

    def __already_slashed(self,
                          validator: spec.ValidatorIndex,
                          attestation1: CachedAttestation,
                          attestation2: CachedAttestation) -> bool:
        for slashed_data1, slashed_data2 in self.accepted_attester_slashings[validator]:
            if (attestation1.attestation.data == slashed_data1
                    and attestation2.attestation.data == slashed_data2)\
                    or (attestation1.attestation.data == slashed_data2
                        and attestation2.attestation.data == slashed_data1):
                return True
        return False

    def __find_attestaion(self, attestation: spec.Attestation) -> Optional[CachedAttestation]:
        slot, committee = attestation.data.slot, attestation.data.index
        if slot in self.cache_by_time and committee in self.cache_by_time[slot]:
            for cached_attestation in self.cache_by_time[slot][committee]:
                if cached_attestation.attestation == attestation:
                    return cached_attestation
        return None

    @staticmethod
    def __ensure_key_exists(target: dict, key, default_generator):
        if key not in target:
            target[key] = default_generator()

"""
class AttestationCache:
    attestation_cache: Dict[spec.Slot, Dict[spec.CommitteeIndex, List[spec.Attestation]]]
    seen_in_block: Dict[spec.Slot, Dict[spec.CommitteeIndex, Set[uint64]]]

    def __init__(self):
        self.attestation_cache = dict()
        self.seen_in_block = dict()

    def add_attestation(self, attestation: spec.Attestation, from_block: bool = False):
        slot = attestation.data.slot
        committee = attestation.data.index

        if slot not in self.attestation_cache:
            self.attestation_cache[slot] = dict()
        if committee not in self.attestation_cache[slot]:
            self.attestation_cache[slot][committee] = list()
        self.attestation_cache[slot][committee].append(attestation)

        if from_block:
            indices = indices_inside_committee(attestation.aggregation_bits)
            if slot not in self.seen_in_block:
                self.seen_in_block[slot] = dict()
            if committee not in self.seen_in_block[slot]:
                self.seen_in_block[slot][committee] = set()
            for index in indices:
                self.seen_in_block[slot][committee].add(index)

    def all_attestations_with_unseen_validators(self, max_slot: spec.Slot):
        for slot, commitee_mapping in self.attestation_cache.items():
            if slot <= max_slot:
                for committee in commitee_mapping.keys():
                    yield from self.attestations_with_unseen_validators(slot, committee)

    def attestations_with_unseen_validators(self, slot: spec.Slot, committee: spec.CommitteeIndex)\
            -> List[spec.Attestation]:
        attestations = self.attestation_cache[slot][committee]\
            if slot in self.attestation_cache and committee in self.attestation_cache[slot]\
            else list()
        seen_indices = set(self.seen_in_block[slot][committee])\
            if slot in self.seen_in_block and committee in self.seen_in_block[slot]\
            else set()

        for attestation in attestations:
            include = False
            indices_in_attestation = indices_inside_committee(attestation.aggregation_bits)
            for index in indices_in_attestation:
                if index not in seen_indices:
                    include = True
                    seen_indices.add(index)
            if include:
                yield attestation

    def attestations(self, max_slot: spec.Slot):
        for slot, committee_mapping in self.attestation_cache.items():
            if slot <= max_slot:
                for committeee, attestations in committee_mapping.items():
                    for attestation in attestations:
                        yield slot, committeee, attestation

    def remove_attestation(self, slot: spec.Slot, committee: spec.CommitteeIndex, attestation: spec.Attestation):
        self.attestation_cache[slot][committee].remove(attestation)

    def cleanup(self, slot: spec.Slot, keep_slots: uint64):
        self.clean_old_slots(slot, keep_slots)
        self.clean_redundant_attestations()

    def clean_old_slots(self, slot: spec.Slot, keep_slots: uint64):
        if slot > keep_slots:
            lower_bound = slot - keep_slots
            slots = tuple(self.attestation_cache.keys())
            for cache_slot in slots:
                if cache_slot < lower_bound:
                    del self.attestation_cache[cache_slot]
            slots = tuple(self.seen_in_block.keys())
            for cache_slot in slots:
                if cache_slot < lower_bound:
                    del self.seen_in_block[cache_slot]

    def clean_redundant_attestations(self):
        for slot, committee_attestation_map in self.attestation_cache.items():
            for committee, attestations in committee_attestation_map.items():
                attestations.sort(key=lambda a: popcnt(a.aggregation_bits), reverse=True)
                keep: List[spec.Attestation] = list()
                for attestation in attestations:
                    current_attestation_indexes = indices_inside_committee(attestation.aggregation_bits)
                    # Get all indexes which are kept by now
                    keep_indexes = []
                    for keep_attestation in keep:
                        for kept_indices in indices_inside_committee(keep_attestation.aggregation_bits):
                            keep_indexes.append(kept_indices)
                    keep_current_attestation = any(index for index in current_attestation_indexes
                                                   if index not in keep_indexes)
                    if keep_current_attestation:
                        keep.append(attestation)
                committee_attestation_map[committee] = keep

    def __str__(self):
        print(str(self.attestation_cache))
"""

class BlockCache:
    blocks: Dict[spec.Root, spec.SignedBeaconBlock]
    accepted: List[Optional[spec.Root]]
    # For additional (slashed) blocks at the same height
    outstanding: Set[spec.Root]

    slashable: Set[spec.Root]
    slashed: Dict[spec.ValidatorIndex, Tuple[spec.Root, spec.Root]]

    def __init__(self, genesis: spec.BeaconBlock):
        genesis_root = spec.hash_tree_root(genesis)
        signed_genesis = spec.SignedBeaconBlock(message=genesis)
        self.blocks = {
            genesis_root: signed_genesis
        }
        # One 'None' for genesis (which is in fact not a signed beacon block)
        self.accepted = [genesis_root]
        self.outstanding = set()
        self.slashable = set()
        self.slashed = dict()

    def add_block(self, block: spec.SignedBeaconBlock, root: Optional[spec.Root] = None):
        if root is None:
            root = spec.hash_tree_root(block.message)
        self.blocks[root] = block
        if root in self.accepted:
            return
        while len(self.accepted) <= block.message.slot:
            self.accepted.append(None)

        self.outstanding.add(root)
        block_root_current_slot = self.accepted[block.message.slot]
        if block_root_current_slot is not None and not block_root_current_slot == root:
            self.slashable.add(root)
            raise ValueError('Second block at same height!')

    def search_slashings(self) -> Iterable[spec.ProposerSlashing]:
        for slashable_root in self.slashable:
            slashable_block = self.blocks[slashable_root]
            original_root = self.accepted[slashable_block.message.slot]
            original_block = self.blocks[original_root]
            assert original_block.message.proposer_index == slashable_block.message.proposer_index

            # Already slashed?
            proposer = slashable_block.message.proposer_index
            if (original_root, slashable_root) not in self.slashed[proposer]\
                    and (slashable_root, original_root) not in self.slashed[proposer]:
                yield self.__produce_slashing(original_block, slashable_block)

    @staticmethod
    def __produce_slashing(block1: spec.SignedBeaconBlock,
                           block2: spec.SignedBeaconBlock) -> spec.ProposerSlashing:
        header1 = spec.BeaconBlockHeader(
            slot=block1.message.slot,
            proposer_index=block1.message.proposer_index,
            parent_root=block1.message.parent_root,
            state_root=block1.message.state_root,
            body_root=spec.hash_tree_root(block1.message.body)
        )
        header2 = spec.BeaconBlockHeader(
            slot=block2.message.slot,
            proposer_index=block2.message.proposer_index,
            parent_root=block2.message.parent_root,
            state_root=block2.message.state_root,
            body_root=spec.hash_tree_root(block2.message.body)
        )

        signed1 = spec.SignedBeaconBlockHeader(
            message=header1,
            signature=block1.signature
        )
        signed2 = spec.SignedBeaconBlockHeader(
            message=header2,
            signature=block2.signature
        )

        return spec.ProposerSlashing(
            signed_header_1=signed1,
            signed_header_2=signed2
        )

    def chain_for_block(self, block: spec.SignedBeaconBlock, store: spec.Store) -> Sequence[spec.SignedBeaconBlock]:
        root = spec.hash_tree_root(block.message)
        chain = []
        self.__chain_for_block(block, root, chain, store)
        return chain

    def __chain_for_block(self, block: spec.SignedBeaconBlock, root: spec.Root, chain: List[spec.SignedBeaconBlock], store: spec.Store):
        if root in store.blocks:
            return
        if block.message.slot == 0:
            return
        if block.message.parent_root not in self.blocks:
            raise KeyError(f"Parent block {block.message.parent_root} of block {root} not known")
        self.__chain_for_block(self.blocks[block.message.parent_root], block.message.parent_root, chain, store)
        chain.append(block)

    def accept_block(self, *kargs: spec.SignedBeaconBlock):
        for block in kargs:
            root = spec.hash_tree_root(block.message)
            self.accepted[block.message.slot] = root
            if root in self.outstanding:
                self.outstanding.remove(root)

    def accept_slashing(self, slashing: spec.ProposerSlashing):
        proposer = slashing.signed_header_1.message.proposer_index
        self.slashed[proposer] = (slashing.signed_header_1.message.body_root,
                                  slashing.signed_header_2.message.body_root)

    def longest_outstanding_chain(self, store: spec.Store) -> Sequence[spec.SignedBeaconBlock]:
        chain = []
        for outstanding in (self.blocks[oblock] for oblock in self.outstanding):
            try:
                ochain = self.chain_for_block(outstanding, store)
            except KeyError:
                ochain = []
            if len(ochain) >= len(chain):
                chain = ochain
        return chain
