from eth2spec.phase0 import spec
from events import AggregateOpportunity, Event, LatestVoteOpportunity, NextSlotEvent, SimulationEndEvent
from beaconclient import BeaconClient
from typing import Optional, Sequence, Dict, Tuple
from enum import Enum

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
    
    def pre_attest(self):
        self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
    
    def pre_propose(self):
        self.update_committee(spec.compute_epoch_at_slot(self.current_slot))

class BalancingAttackingBeaconClient(BeaconClient):
    class AttackState(Enum):
        ATTACK_PLANNED = 0
        ATTACK_STARTED_FIRST_EPOCH = 1
        ATTACK_STARTED_SUBSEQUENT_EPOCHS = 2
        ATTACK_STOPPED = 3
    
    swayers_first_epoch_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    swayers_first_epoch_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    swayers_subsequent_epochs_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    swayers_subsequent_epochs_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    fillers_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    fillers_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]

    state_left: spec.BeaconState
    head_state_left: spec.BeaconState

    state_right: spec.BeaconState
    head_state_right: spec.BeaconState

    block_root_left: Optional[spec.Root]
    block_root_right: Optional[spec.Root]
    
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
        self.attack_state = self.AttackState.ATTACK_PLANNED
        self.left_ids = []
        self.right_ids = []
        self.swayers = []
        self.left_balance = spec.Gwei(0)
        self.right_balance = spec.Gwei(0)
    
    def on_epoch_start(self):
        if self.attack_state == self.AttackState.ATTACK_PLANNED:
            feasible = self.is_attack_feasible()
            if feasible:
                self.client_to_simulator_queue.put(SimulationEndEvent(self.current_time, 0, feasible))
                self.assign_roles()
                self.attack_state = self.AttackState.ATTACK_STARTED_FIRST_EPOCH
    
    def is_attack_feasible(self) -> bool:
        """
        Must be called at the start of an epoch.

        As per [NTT20]:
        Thus, sufficient for an epoch to be opportune to start the attack is that the following conditions are all satisfied:
        - E(0)(a): The proposer of slot 0 is adversarial.
        - E(0)(b): Slot 0 has >= 6 adversarial validators
                   (the adversarial proposer, two swayers for epoch 0, two swayers for epoch 1, potentially one filler).
        - E(0)(c): Slots i= 1, ..., (C - 2) have >= 5 adversarial validators
                   (two swayers for epoch 0, two swayers for epoch 1, potentially one filler).
        - E(0)(d): Slot (C - 1) has >= 3 adversarial validators
                   (two swayers for epoch 1, potentially one filler).
        """

        # Proposer of slot 0 is adversarial
        # As this is the start of the epoch and `compute_head` was already called by now, we have a progressed state
        # This means, we can just check if the current beacon client proposes a block

        proposer_index = spec.get_beacon_proposer_index(self.head_state)
        proposer_adversarial = proposer_index in self.indexed_validators and not self.slashed(proposer_index)

        # The committee for the whole epoch has been calculated by now so we can check if there are enough validators inside the current slot

        adversarial_validators_for_first_slot = self.attesting_validators_at_slot(self.current_slot)
        enough_adversarial_validators_for_first_slot = len(adversarial_validators_for_first_slot.keys()) >= 2 + 2 + self.number_of_fillers_needed()

        # Check if all slots from the second till the one before the last have at least 4 swayers and the needed fillers

        first_slot_of_epoch = self.current_slot.copy()
        assert spec.compute_start_slot_at_epoch(spec.compute_epoch_at_slot(first_slot_of_epoch)) == first_slot_of_epoch
        last_slot_of_epoch = first_slot_of_epoch + spec.SLOTS_PER_EPOCH - 1

        adversial_validators_for_middle_slots = [self.attesting_validators_at_slot(slot) for slot in range(first_slot_of_epoch, last_slot_of_epoch)]
        enough_adversarial_validators_for_middle_slots = all(len(adversial_validators_for_middle_slot.keys()) >= 2 + 2 + self.number_of_fillers_needed()
            for adversial_validators_for_middle_slot in adversial_validators_for_middle_slots)

        # Check if the last slot has enough adversarial validators (two swayers plus needed fillers)

        adversarial_validators_for_last_slot = self.attesting_validators_at_slot(last_slot_of_epoch)
        enough_adversarial_validators_for_last_slot = len(adversarial_validators_for_last_slot.keys()) >= 2 + self.number_of_fillers_needed()

        attack_feasible = proposer_adversarial and enough_adversarial_validators_for_first_slot and enough_adversarial_validators_for_middle_slots and enough_adversarial_validators_for_last_slot
        print(f'[BEACON CLIENT {self.counter}] Feasability Check: feasible=[{attack_feasible}] | proposer=[{proposer_adversarial}] first=[{enough_adversarial_validators_for_first_slot}] middle=[{enough_adversarial_validators_for_middle_slots}] last=[{enough_adversarial_validators_for_last_slot}]')
        return attack_feasible
    
    def number_of_fillers_needed(self) -> int:
        return 0
    
    def assign_roles(self):
        adversarial_validators_for_first_slot = tuple(self.attesting_validators_at_slot(self.current_slot).keys())

        first_slot_of_epoch = self.current_slot.copy()
        assert spec.compute_start_slot_at_epoch(spec.compute_epoch_at_slot(first_slot_of_epoch)) == first_slot_of_epoch
        last_slot_of_epoch = first_slot_of_epoch + spec.SLOTS_PER_EPOCH - 1
        
        adversial_validators_for_middle_slots = [tuple(self.attesting_validators_at_slot(slot).keys()) for slot in range(first_slot_of_epoch, last_slot_of_epoch)]
        adversarial_validators_for_last_slot = tuple(self.attesting_validators_at_slot(last_slot_of_epoch).keys())

        # Roles for first slot
        swayers_first_slot_current_epoch, swayers_first_slot_subsequent_epochs, fillers = adversarial_validators_for_first_slot[2..4]
        last_filler = len(adversarial_validators_for_first_slot.keys())

    @staticmethod
    def __do_assign(validators: Sequence[spec.ValidatorIndex]) -> Tuple[Sequence[spec.ValidatorIndex], Sequence[spec.ValidatorIndex], Sequence[spec.ValidatorIndex]]:
        validators_len = len(validators)
        validators_len_even = validators_len % 2 == 0
        return (
            validators[0:2],
            validators[2:4],
            validators[2:validators_len] if validators_len_even else validators[2:validators_len - 1]
        )