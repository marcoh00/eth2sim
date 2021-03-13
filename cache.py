from typing import Dict, List, Set

from remerkleable.basic import uint64

from eth2spec.phase0 import spec
from helpers import popcnt, indices_inside_committee


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
