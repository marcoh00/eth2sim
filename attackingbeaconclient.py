from eth2spec.phase0 import spec
from events import NextSlotEvent
from beaconclient import BeaconClient

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
    
    def pre_next_slot_event(self, message: NextSlotEvent):
        if self.attack_start_slot <= message.slot < self.attack_end_slot:
            print(f'Time attacked. Pre-change time=[{message.time}] slot=[{message.slot}]')
            message.time += self.timedelta * spec.SECONDS_PER_SLOT
            message.slot += self.timedelta
            self.current_slot = message.slot
            print(f'Time attacked. Post-change time=[{message.time}] slot=[{message.slot}]')
        return super().pre_next_slot_event(message)
    
    def post_next_slot_event(self, head_state, indexed_validators):
        if self.attack_start_slot <= self.current_slot < self.attack_end_slot and not self.attack_started:
            print('Update committee because attack was started')
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
            self.attack_started = True
        if self.attack_started and self.current_slot - self.timedelta >= self.attack_end_slot:
            print('Update committee because attack ended')
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
            self.attack_started = False
        return super().post_next_slot_event(head_state, indexed_validators)