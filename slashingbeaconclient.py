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