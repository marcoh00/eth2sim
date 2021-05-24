from eth2spec.phase0 import spec
from typing import Dict, List, Optional, Tuple
from beaconclient import BeaconClient


class BlockSlashingBeaconClient(BeaconClient):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        print('CONSTRUCTED BLOCK SLASHER')
    
    def post_next_slot_event(self, head_state, indexed_validators):
        if self.proposer_current_slot in indexed_validators and not self.slashed(self.proposer_current_slot):
            print(f"[BEACON CLIENT {self.counter}] Produce slashable block for validator {self.proposer_current_slot}")
            self.propose_block(indexed_validators[self.proposer_current_slot], head_state, slashme=True)
        return super().post_next_slot_event(head_state, indexed_validators)

class AttesterSlashingTooLowClient(BeaconClient):

    past_checkpoint: Dict[spec.ValidatorIndex, spec.Checkpoint]

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.past_checkpoint = dict()
        print('CONSTRUCTED ATTESTER SLASHER W/ TOO LOW ATTESTATIONS')
    
    def produce_attestation_data(self, validator_index, validator_committee, epoch_boundary_block_root, head_block, head_state) -> Optional[spec.AttestationData]:
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        attestation_data = super().produce_attestation_data(validator_index, validator_committee, epoch_boundary_block_root, head_block, head_state)

        if current_epoch > 0:
            if validator_index in self.past_checkpoint:
                print(f"[BEACON CLIENT {self.counter}] Use saved Checkpoint to produce a slashable attestation for validator {validator_index}")
                attestation_data.target = self.past_checkpoint[validator_index]
            else:
                print(f"[BEACON CLIENT {self.counter}] Save Checkpoint for validator {validator_index}")
                self.past_checkpoint[validator_index] = attestation_data.target
        
        return attestation_data