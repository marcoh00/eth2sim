from eth2spec.phase0 import spec
from typing import Dict, List, Optional, Tuple
from beaconnode import BeaconNode


class BlockSlashingBeaconNode(BeaconNode):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        print("CONSTRUCTED BLOCK SLASHER")

    def post_next_slot_event(self, head_state, indexed_validators):
        if self.proposer_current_slot in indexed_validators and not self.slashed(
            self.proposer_current_slot
        ):
            print(
                f"[BEACON NODE {self.counter}] Produce slashable block for validator {self.proposer_current_slot}"
            )
            self.propose_block(
                indexed_validators[self.proposer_current_slot], head_state, slashme=True
            )
        return super().post_next_slot_event(head_state, indexed_validators)


class AttesterSlashingSameHeightClient(BeaconNode):

    past_checkpoint: Dict[spec.ValidatorIndex, spec.Checkpoint]

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.past_checkpoint = dict()
        print("CONSTRUCTED ATTESTER SLASHER W/ TOO LOW ATTESTATIONS")

    def produce_attestation_data(
        self,
        validator_index,
        validator_committee,
        epoch_boundary_block_root,
        head_block,
        head_state,
    ) -> Optional[spec.AttestationData]:
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        attestation_data = super().produce_attestation_data(
            validator_index,
            validator_committee,
            epoch_boundary_block_root,
            head_block,
            head_state,
        )

        if current_epoch > 0:
            if validator_index in self.past_checkpoint:
                print(
                    f"[BEACON NODE {self.counter}] Use saved Checkpoint to produce a slashable attestation for validator {validator_index}"
                )
                attestation_data.target = self.past_checkpoint[validator_index]
            else:
                print(
                    f"[BEACON NODE {self.counter}] Save Checkpoint for validator {validator_index}"
                )
                self.past_checkpoint[validator_index] = attestation_data.target

        return attestation_data


class AttesterSlashingWithinSpan(BeaconNode):

    first_attestation: Dict[
        spec.ValidatorIndex, Tuple[spec.Checkpoint, spec.Checkpoint]
    ]
    attestation_target_inside_span: Dict[spec.ValidatorIndex, spec.Checkpoint]

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.first_attestation = dict()
        self.attestation_target_inside_span = dict()
        print("CONSTRUCTED ATTESTER SLASHER W/ ATTESTATIONS WITHIN SPAN")

    def produce_attestation_data(
        self,
        validator_index,
        validator_committee,
        epoch_boundary_block_root,
        head_block,
        head_state,
    ) -> Optional[spec.AttestationData]:
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        attestation_data = super().produce_attestation_data(
            validator_index,
            validator_committee,
            epoch_boundary_block_root,
            head_block,
            head_state,
        )

        # 1 Save first known source & target checkpoint for validator (s1)
        # 2 On new attestation: if a_source > s1_source and a_target > s1_target: Save second target checkpoint (s2)
        # 3 On new attestation: if a_target > s2_target: Construct attestation with a_target as target and s1_source as source

        if validator_index not in self.first_attestation:
            # Step 1: Save first known source & target checkpoints
            print(
                f"[BEACON NODE {self.counter}] Save first checkpoint seen for Validator {validator_index}"
            )
            self.first_attestation[validator_index] = (
                attestation_data.source,
                attestation_data.target,
            )
            return attestation_data

        if validator_index not in self.attestation_target_inside_span:
            # Step 2: Save target checkpoint of designated inside-span attestation
            if (
                attestation_data.source.epoch
                > self.first_attestation[validator_index][0].epoch
                and attestation_data.source.epoch
                > self.first_attestation[validator_index][1].epoch
            ):
                print(
                    f"[BEACON NODE {self.counter}] Save designated inside-span attestation target checkpoint for Validator {validator_index}"
                )
                self.attestation_target_inside_span[
                    validator_index
                ] = attestation_data.target
            return attestation_data

        # Step 3: If possible, construct an attestation which spans from the first attestation source (< step 2 attestation)
        #         to the current attestation's target (> step 2 attestation)
        if (
            attestation_data.target.epoch
            > self.attestation_target_inside_span[validator_index].epoch
        ):
            print(
                f"[BEACON NODE {self.counter}] Produce spanning attestation for Validator {validator_index}"
            )
            attestation_data.source = self.first_attestation[validator_index][0]
            return attestation_data
