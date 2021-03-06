from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Sequence, Iterable, Tuple

from queue import Queue
import traceback
from eth2spec.phase0 import spec
from eth2spec.utils.ssz import ssz_typing

from os import getpid


@dataclass
class CachedAttestation:
    attesting_indices: ssz_typing.List[
        spec.ValidatorIndex, spec.MAX_VALIDATORS_PER_COMMITTEE
    ]
    seen_in_block: bool
    known_to_forkchoice: bool
    denied: True
    attestation: spec.Attestation

    def into_indexed_attestation(self) -> spec.IndexedAttestation:
        return spec.IndexedAttestation(
            attesting_indices=self.attesting_indices,
            data=self.attestation.data,
            signature=self.attestation.signature,
        )

    @classmethod
    def from_attestation(cls, attestation: spec.Attestation, state: spec.BeaconState):
        indexed_attestation = spec.get_indexed_attestation(state, attestation)
        return cls(
            attesting_indices=indexed_attestation.attesting_indices,
            seen_in_block=False,
            known_to_forkchoice=False,
            denied=False,
            attestation=attestation
        )


class AttestationCache:
    cache_by_time: Dict[spec.Slot, Dict[spec.CommitteeIndex, List[CachedAttestation]]]
    cache_by_validator: Dict[spec.ValidatorIndex, List[CachedAttestation]]
    queued_attestations: Queue
    state_cache: Dict[spec.Epoch, Dict[spec.Root, spec.BeaconState]]
    counter: int

    def __init__(self, counter=-1):
        self.cache_by_time = dict()
        self.cache_by_validator = dict()
        self.queued_attestations = None
        self.state_cache = dict()
        self.counter = counter

    # TODO do not immediately set the validators for future attestations. The committee is not known now!!
    def add_attestation(
        self,
        attestation: spec.Attestation,
        store: spec.Store,
        seen_in_block=False,
        raise_on_error=False,
        marker=None,
    ):
        existing_cached_attestation = self.__find_attestaion(attestation)
        if existing_cached_attestation is not None:
            if seen_in_block:
                existing_cached_attestation.seen_in_block = True
            return

        # Check if we have a state to determine the correct committee
        attestation_block_root = attestation.data.beacon_block_root
        if attestation_block_root not in store.block_states:
            if raise_on_error:
                raise ValueError(
                    f"[BEACON NODE {self.counter}][ATTESTATION CACHE] Received attestation for block not known to fork choice: root=[{attestation_block_root}] slot=[{attestation.data.slot}]"
                )
            else:
                print(
                    f"[BEACON NODE {self.counter}][ATTESTATION CACHE] Received attestation for block not known to fork choice: root=[{attestation_block_root}] slot=[{attestation.data.slot}]"
                )
                if self.queued_attestations is None:
                    self.queued_attestations = Queue()
                self.queued_attestations.put((attestation, marker))
                return

        attestation_block_state = store.block_states[attestation_block_root].copy()
        attestation_slot_epoch = spec.compute_epoch_at_slot(attestation.data.slot)
        attestation_block_slot_epoch = spec.compute_epoch_at_slot(
            attestation_block_state.slot
        )

        if attestation_slot_epoch > attestation_block_slot_epoch + 1:
            if (
                attestation_slot_epoch in self.state_cache
                and attestation_block_root in self.state_cache[attestation_slot_epoch]
            ):
                attestation_block_state = self.state_cache[attestation_slot_epoch][
                    attestation_block_root
                ]
            else:
                print(
                    f"[BEACON NODE {self.counter}][ATTESTATION CACHE] Attestation Cache needs to manually forward the state. epoch_block_slot=[{attestation_block_slot_epoch}] epoch_attestation_slot=[{attestation_slot_epoch}]"
                )
                spec.process_slots(attestation_block_state, attestation.data.slot)
                self.__ensure_key_exists(self.state_cache, attestation_slot_epoch, dict)
                self.state_cache[attestation_slot_epoch][
                    attestation_block_root
                ] = attestation_block_state

        cached_attestation = CachedAttestation.from_attestation(
            attestation, attestation_block_state
        )
        cached_attestation.seen_in_block = seen_in_block
        slot, committee = attestation.data.slot, attestation.data.index

        self.__ensure_key_exists(self.cache_by_time, slot, dict)
        self.__ensure_key_exists(self.cache_by_time[slot], committee, list)
        self.cache_by_time[slot][committee].append(cached_attestation)
        for validator in cached_attestation.attesting_indices:
            self.__ensure_key_exists(self.cache_by_validator, validator, list)
            self.cache_by_validator[validator].append(cached_attestation)

    def accept_attestation(
        self, attestation: spec.Attestation, forkchoice=False, block=False
    ):
        cached_attestation = self.__find_attestaion(attestation)
        if forkchoice:
            cached_attestation.known_to_forkchoice = True
        if block:
            cached_attestation.seen_in_block = True
    
    def deny_attestation(
        self, attestation: spec.Attestation
    ):
        cached = self.__find_attestaion(attestation)
        cached.denied = True

    def add_queued_attestations(self, store: spec.Store):
        if self.queued_attestations is None:
            self.queued_attestations = Queue()
        unsuccessful = Queue()
        while not self.queued_attestations.empty():
            attestation, marker = self.queued_attestations.get()
            try:
                self.add_attestation(
                    attestation, store, raise_on_error=True, marker=marker
                )
                print(
                    f"[BEACON NODE {self.counter}][ATTESTATION CACHE] Successfully added queued attestation into cache: root=[{attestation.data.beacon_block_root}] slot=[{attestation.data.slot}]"
                )
            except ValueError:
                unsuccessful.put((attestation, marker))
        self.queued_attestations = unsuccessful

    # def accept_slashing(self, slashing: spec.AttesterSlashing):
    #     validators = set(slashing.attestation_1.attesting_indices)\
    #         .intersection(set(slashing.attestation_2.attesting_indices))
    #     for validator in validators:
    #         self.__ensure_key_exists(self.accepted_attester_slashings, validator, list())
    #         self.accepted_attester_slashings[validator].append(
    #             (slashing.attestation_1.data, slashing.attestation_2.data)
    #         )

    def attestations_not_known_to_forkchoice(
        self, min_slot: spec.Slot, max_slot: spec.Slot
    ) -> Iterable[spec.Attestation]:
        yield from self.filter_attestations(
            min_slot, max_slot, lambda a: not a.known_to_forkchoice and not a.denied
        )

    def attestations_not_seen_in_block(
        self, min_slot: spec.Slot, max_slot: spec.Slot
    ) -> Iterable[spec.Attestation]:
        yield from self.filter_attestations(
            min_slot, max_slot, lambda a: not a.seen_in_block and a.known_to_forkchoice and not a.denied
        )

    def filter_attestations(
        self, min_slot: spec.Slot, max_slot: spec.Slot, filter_func
    ) -> Iterable[spec.Attestation]:
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
                    if any(
                        validator not in validators_seen_in_kept_attestations
                        for validator in attestation.attesting_indices
                    ):
                        kept_attestations.append(attestation)
                        for validator in attestation.attesting_indices:
                            validators_seen_in_kept_attestations.add(validator)
                self.cache_by_time[slot][committee] = kept_attestations

    def cleanup_old_time_cache_attestations(self, min_slot: spec.Slot):
        slots_to_clean = tuple(
            slot for slot in self.cache_by_time.keys() if slot < min_slot
        )
        for slot in slots_to_clean:
            del self.cache_by_time[slot]

    def search_slashings(
        self, state: spec.BeaconState, validator: Optional[spec.ValidatorIndex] = None
    ):
        if validator is not None:
            yield from self.int_search_slashings(state, validator)
        else:
            for validator in self.cache_by_validator.keys():
                yield from self.int_search_slashings(state, validator)

    def int_search_slashings(
        self, state: spec.BeaconState, validator: spec.ValidatorIndex
    ) -> Iterable[spec.AttesterSlashing]:
        slashed_validators = []
        for i in range(len(self.cache_by_validator[validator])):
            for j in range(i, len(self.cache_by_validator[validator])):
                attestation1 = self.cache_by_validator[validator][i]
                attestation2 = self.cache_by_validator[validator][j]
                if spec.is_slashable_attestation_data(
                    attestation1.attestation.data, attestation2.attestation.data
                ):
                    if (
                        not state.validators[validator].slashed
                        and validator not in slashed_validators
                    ):
                        indexed_1 = attestation1.into_indexed_attestation()
                        indexed_2 = attestation2.into_indexed_attestation()
                        yield spec.AttesterSlashing(
                            attestation_1=indexed_1, attestation_2=indexed_2
                        )
                        slashed_validators += set(
                            indexed_1.attesting_indices
                        ).intersection(indexed_2.attesting_indices)
                elif spec.is_slashable_attestation_data(
                    attestation2.attestation.data, attestation1.attestation.data
                ):
                    if (
                        not state.validators[validator].slashed
                        and validator not in slashed_validators
                    ):
                        indexed_1 = attestation1.into_indexed_attestation()
                        indexed_2 = attestation2.into_indexed_attestation()
                        yield spec.AttesterSlashing(
                            attestation_1=indexed_2, attestation_2=indexed_1
                        )
                        slashed_validators += set(
                            indexed_1.attesting_indices
                        ).intersection(indexed_2.attesting_indices)

    def __find_attestaion(
        self, attestation: spec.Attestation
    ) -> Optional[CachedAttestation]:
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


class BlockCache:
    blocks: Dict[spec.Root, spec.SignedBeaconBlock]
    accepted: List[Optional[spec.Root]]
    outstanding: Set[spec.Root]
    slashable: Set[Tuple[spec.Root]]

    def __init__(self, genesis: spec.BeaconBlock):
        genesis_root = spec.hash_tree_root(genesis)
        signed_genesis = spec.SignedBeaconBlock(message=genesis)
        self.blocks = {genesis_root: signed_genesis}
        # One 'None' for genesis (which is in fact not a signed beacon block)
        self.accepted = [genesis_root]
        self.outstanding = set()
        self.slashable = set()

    def add_block(
        self, block: spec.SignedBeaconBlock, root: Optional[spec.Root] = None
    ):
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
            self.slashable.add((block_root_current_slot, root))
            # raise ValueError('Second block at same height!')

    def search_slashings(
        self, state: spec.BeaconState
    ) -> Iterable[spec.ProposerSlashing]:
        useless_roots = set()
        for roots in self.slashable:
            slashable_block = self.blocks[roots[0]]
            original_block = self.blocks[roots[1]]
            if (
                original_block.message.proposer_index
                != slashable_block.message.proposer_index
            ):
                # Someone else proposed a block here. Looks like a long fork!
                # While this is unfortunate, it is not slashable...
                useless_roots.add(roots)
                continue

            # Already slashed?
            proposer = slashable_block.message.proposer_index
            if not state.validators[proposer].slashed:
                yield self.__produce_slashing(original_block, slashable_block)
        self.slashable.difference_update(useless_roots)

    @staticmethod
    def __produce_slashing(
        block1: spec.SignedBeaconBlock, block2: spec.SignedBeaconBlock
    ) -> spec.ProposerSlashing:
        header1 = spec.BeaconBlockHeader(
            slot=block1.message.slot,
            proposer_index=block1.message.proposer_index,
            parent_root=block1.message.parent_root,
            state_root=block1.message.state_root,
            body_root=spec.hash_tree_root(block1.message.body),
        )
        header2 = spec.BeaconBlockHeader(
            slot=block2.message.slot,
            proposer_index=block2.message.proposer_index,
            parent_root=block2.message.parent_root,
            state_root=block2.message.state_root,
            body_root=spec.hash_tree_root(block2.message.body),
        )

        signed1 = spec.SignedBeaconBlockHeader(
            message=header1, signature=block1.signature
        )
        signed2 = spec.SignedBeaconBlockHeader(
            message=header2, signature=block2.signature
        )

        return spec.ProposerSlashing(signed_header_1=signed1, signed_header_2=signed2)

    def chain_for_block(
        self, block: spec.SignedBeaconBlock, store: spec.Store
    ) -> Sequence[spec.SignedBeaconBlock]:
        root = spec.hash_tree_root(block.message)
        chain = []
        self.__chain_for_block(block, root, chain, store)
        return chain

    def __chain_for_block(
        self,
        block: spec.SignedBeaconBlock,
        root: spec.Root,
        chain: List[spec.SignedBeaconBlock],
        store: spec.Store,
    ):
        if root in store.blocks:
            return
        if block.message.slot == 0:
            return
        if block.message.parent_root not in self.blocks:
            raise KeyError(
                f"Parent block {block.message.parent_root} of block {root} not known"
            )
        self.__chain_for_block(
            self.blocks[block.message.parent_root],
            block.message.parent_root,
            chain,
            store,
        )
        chain.append(block)

    def accept_block(self, *kargs: spec.SignedBeaconBlock):
        for block in kargs:
            root = spec.hash_tree_root(block.message)
            self.accepted[block.message.slot] = root
            if root in self.outstanding:
                self.outstanding.remove(root)

    # def accept_slashing(self, slashing: spec.ProposerSlashing):
    #     proposer = slashing.signed_header_1.message.proposer_index
    #     self.slashed[proposer] = (slashing.signed_header_1.message.body_root,
    #                               slashing.signed_header_2.message.body_root)

    def longest_outstanding_chain(
        self, store: spec.Store
    ) -> Sequence[spec.SignedBeaconBlock]:
        chain = []
        for outstanding in (self.blocks[oblock] for oblock in self.outstanding):
            try:
                ochain = self.chain_for_block(outstanding, store)
            except KeyError:
                ochain = []
            if len(ochain) >= len(chain):
                chain = ochain
        return chain

    def leafs_for_block(self, root: spec.Root) -> Sequence[spec.Root]:
        children = list()
        for blockroot, block in self.blocks.items():
            if block.message.parent_root == root:
                children.append(blockroot)
        leafs = list()
        if len(children) == 0:
            leafs.append(root)
        else:
            for child in children:
                for leaf in self.leafs_for_block(child):
                    leafs.append(leaf)
        return leafs
