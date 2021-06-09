from eth2spec.phase0 import spec
from events import AggregateOpportunity, Event, LatestVoteOpportunity, NextSlotEvent
from beaconclient import BeaconClient
from typing import Optional

class TimeAttackedBeaconClient(BeaconClient):
    def __init__(self, *kargs, **kwargs):
        super().__init__(
            counter=kwargs['counter'],
            simulator_to_client_queue=kwargs['simulator_to_client_queue'],
            client_to_simulator_queue=kwargs['client_to_simulator_queue'],
            configpath=kwargs['configpath'],
            configname=kwargs['configname'],
            validator_builders=kwargs['validator_builders'],
            validator_first_counter=kwargs['validator_first_counter'],
            debug=kwargs['debug'],
            profile=kwargs['profile'],
        )
        self.attack_start_slot = kwargs['attack_start_slot']
        self.attack_end_slot = kwargs['attack_end_slot']
        self.timedelta = kwargs['timedelta']
        self.attack_started = False
        self.is_first_attack_slot = False
        self.is_first_synced_slot = False
    
    def pre_event_handling(self, message: Event):
        if self.attack_started and type(message) in (LatestVoteOpportunity, AggregateOpportunity):
            message.time += self.timedelta * spec.SECONDS_PER_SLOT
            message.slot += self.timedelta
            #print(f'Change type of {type(message)}')
    
    def pre_next_slot_event(self, message: NextSlotEvent):
        # Update internal state of attack
        if self.attack_start_slot <= message.slot < self.attack_end_slot:
            self.is_first_attack_slot = False
            if not self.attack_started:
                self.attack_started = True
                self.is_first_attack_slot = True
            print(f'[BEACON CLIENT {self.counter}] Time attacked. Regular: time=[{message.time}] slot=[{message.slot}]')
            message.time += self.timedelta * spec.SECONDS_PER_SLOT
            message.slot += self.timedelta
            self.current_slot = message.slot
            print(f'[BEACON CLIENT {self.counter}] Time attacked. Attacked: time=[{message.time}] slot=[{message.slot}]')
        elif self.attack_started:
            self.is_first_synced_slot = True
        
        self.ensure_checkpoints_are_known()
        return super().pre_next_slot_event(message)
    
    def ensure_checkpoints_are_known(self):
        """
        Because all old attestations are deleted after `pre_next_slot_event`, they will never be passed to the fork choice rule.
        While this makes sense for regular operations (as they seem to be old, they will not be accepted anyway),
        this prohibits the store from caching the checkpoint states of latest justified checkpoints.
        These states are only saved on receiving attestations.
        We now need to make sure the states are cached by ourselves.
        """
        for attestation in self.attestation_cache.attestations_not_known_to_forkchoice(spec.Slot(0), self.current_slot):
            spec.store_target_checkpoint_state(self.store, attestation.data.target)
    
    def post_next_slot_event(self, head_state, indexed_validators):
        if self.is_first_attack_slot:
            print(f'[BEACON CLIENT {self.counter}] Time attack started with delta {self.timedelta}')
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
            self.is_first_attack_slot = False
        
        if self.is_first_synced_slot:
            print(f'[BEACON CLIENT {self.counter}] Time attack stopped')
            self.committee = dict()
            self.slot_last_attested = None
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
            self.attack_started = False
            self.is_first_synced_slot = False
        
        print(f'[BEACON CLIENT {self.counter}] Time-attacked client is at slot {self.current_slot} right now')
        return super().post_next_slot_event(head_state, indexed_validators)
    
    def produce_attestation_data(self, validator_index, validator_committee, epoch_boundary_block_root, head_block, head_state) -> Optional[spec.AttestationData]:
        attestation = super().produce_attestation_data(validator_index, validator_committee, epoch_boundary_block_root, head_block, head_state)
        print(f"[BEACON CLIENT {self.counter}] Time-attacked client produces attestation: slot=[{attestation.slot}] srcep=[{attestation.source.epoch}] targetep=[{attestation.target.epoch}] validator=[{validator_index}]")
        return attestation