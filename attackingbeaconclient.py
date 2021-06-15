from eth2spec.phase0 import spec
from events import AggregateOpportunity, Event, LatestVoteOpportunity, NextSlotEvent, SimulationEndEvent, BeaconClientInfo, MessageEvent
from beaconclient import BeaconClient
from typing import Optional, Sequence, Dict, Tuple
from enum import Enum
from validator import Validator
from dataclasses import dataclass

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
    
    @dataclass
    class ClientWeightMap:
        beacon_client_counter: int
        weight: spec.Gwei
    
    attack_state: AttackState
    attack_started_slot: Optional[spec.Slot]

    beacon_client_validator_map: Dict[int, Sequence[spec.ValidatorIndex]]
    beacon_clients_left: Sequence[ClientWeightMap]
    beacon_clients_right: Sequence[ClientWeightMap]

    swayers_first_epoch_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    swayers_first_epoch_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]

    swayers_subsequent_epochs_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    swayers_subsequent_epochs_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]

    fillers_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    fillers_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]

    state_left: Optional[spec.BeaconState]
    head_state_left: Optional[spec.BeaconState]

    state_right: Optional[spec.BeaconState]
    head_state_right: Optional[spec.BeaconState]

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
        self.attack_started_slot = None
        self.beacon_client_validator_map = dict()
        self.beacon_clients_left = list()
        self.beacon_clients_right = list()
        self.swayers_first_epoch_left = dict()
        self.swayers_first_epoch_right = dict()
        self.swayers_subsequent_epochs_left = dict()
        self.swayers_subsequent_epochs_right = dict()
        self.fillers_left = dict()
        self.fillers_right = dict()
        self.state_left = None
        self.head_state_left = None
        self.state_right = None
        self.head_state_right = None
        self.block_root_left = None
        self.block_root_right = None
    
    def handle_beacon_client_info(self, event: BeaconClientInfo):
        for beacon_client_counter in event.beacon_clients.keys():
            if beacon_client_counter == self.counter:
                continue
            self.beacon_client_validator_map[beacon_client_counter] = tuple(spec.ValidatorIndex(index) for index in event.beacon_clients[beacon_client_counter])
    
    def determine_positition_of_beacon_clients(self):
        all_clients_with_weight = list(self.ClientWeightMap(beacon_client_counter=i, weight=spec.Gwei(0)) for i in self.beacon_client_validator_map.keys())
        self.update_weights(all_clients_with_weight)
        all_clients_with_weight.sort(key=lambda c: c.weight, reverse=True)

        beacon_clients_left = list()
        beacon_clients_right = list()
        number_of_clients_even = len(all_clients_with_weight) % 2 == 0
        number_of_clients_to_assign = len(all_clients_with_weight) if number_of_clients_even else len(all_clients_with_weight) - 1
        for i in range(0, number_of_clients_to_assign, 2):
            beacon_clients_left.append(all_clients_with_weight[i])
            beacon_clients_right.append(all_clients_with_weight[i + 1])
        if not number_of_clients_even:
            beacon_clients_right.append(all_clients_with_weight[-1])

        self.beacon_clients_left = beacon_clients_left
        self.beacon_clients_right = beacon_clients_right
        print(self.weight_for_side())
    
    def weight_for_side(self) -> Tuple[spec.Gwei, spec.Gwei]:
        left = sum(client.weight for client in self.beacon_clients_left)
        right = sum(client.weight for client in self.beacon_clients_right)
        return left, right
    
    def update_weights(self, clients: Sequence[ClientWeightMap]):
        active_indices = spec.get_active_validator_indices(self.head_state, spec.get_current_epoch(self.head_state))
        for client in clients:
            client.weight = spec.Gwei(sum(self.head_state.validators[validator_index].effective_balance
                    for validator_index in active_indices
                    if validator_index in self.beacon_client_validator_map[client.beacon_client_counter]
            ))

    def on_epoch_start(self):
        if self.attack_state == self.AttackState.ATTACK_PLANNED:
            self.determine_positition_of_beacon_clients()
            feasible = self.is_attack_feasible()
            if feasible:
                # self.client_to_simulator_queue.put(SimulationEndEvent(self.current_time, 0, feasible))
                self.attack_state = self.AttackState.ATTACK_STARTED_FIRST_EPOCH
                self.attack_started_slot = self.current_slot.copy()
    
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

        adversarial_validators_for_first_slot = tuple(self.attesting_validators_at_slot(self.current_slot).keys())
        enough_adversarial_validators_for_first_slot_condition = 2 + 2 + self.number_of_fillers_needed() + 4 * self.number_of_extra_swayers()
        enough_adversarial_validators_for_first_slot = len(adversarial_validators_for_first_slot) >= 2 + 2 + self.number_of_fillers_needed()

        # Check if all slots from the second till the one before the last have at least 4 swayers and the needed fillers

        first_slot_of_epoch = self.current_slot.copy()
        assert spec.compute_start_slot_at_epoch(spec.compute_epoch_at_slot(first_slot_of_epoch)) == first_slot_of_epoch
        last_slot_of_epoch = first_slot_of_epoch + spec.SLOTS_PER_EPOCH - 1

        adversial_validators_for_middle_slots = [
            set(self.attesting_validators_at_slot(slot).keys()).difference((proposer_index,))
            for slot in range(first_slot_of_epoch + 1, last_slot_of_epoch)
        ]
        enough_adversarial_validators_for_middle_slots = all(len(adversial_validators_for_middle_slot) >= 2 + 2 + self.number_of_fillers_needed() + (4 * self.number_of_extra_swayers())
            for adversial_validators_for_middle_slot in adversial_validators_for_middle_slots)

        # Check if the last slot has enough adversarial validators (two swayers plus needed fillers)

        adversarial_validators_for_last_slot = set(self.attesting_validators_at_slot(last_slot_of_epoch).keys()).difference((proposer_index,))
        enough_adversarial_validators_for_last_slot = len(adversarial_validators_for_last_slot) >= 2 + self.number_of_fillers_needed() + (2 * self.number_of_extra_swayers())

        attack_feasible = proposer_adversarial and enough_adversarial_validators_for_first_slot and enough_adversarial_validators_for_middle_slots and enough_adversarial_validators_for_last_slot
        print(f'[BEACON CLIENT {self.counter}] Feasability Check: feasible=[{attack_feasible}] | proposer=[{proposer_adversarial}] first=[{enough_adversarial_validators_for_first_slot}] middle=[{enough_adversarial_validators_for_middle_slots}] last=[{enough_adversarial_validators_for_last_slot}]')

        if attack_feasible:
            self.assign(adversarial_validators_for_first_slot, spec.Slot(0))
            for slot_minus_one, validators in enumerate(adversial_validators_for_middle_slots):
                slot = spec.Slot(slot_minus_one + 1)
                self.assign(validators, slot)
            self.assign(adversarial_validators_for_last_slot, spec.Slot(spec.SLOTS_PER_EPOCH - spec.Slot(1)))
            print(f'[BEACON CLIENT 0] swayers_left_first=[{self.swayers_first_epoch_left}] swayers_right_first=[{self.swayers_first_epoch_right}] swayers_left_subsequent=[{self.swayers_subsequent_epochs_left}] swayers_right_subsequent=[{self.swayers_subsequent_epochs_right}] fillers_left=[{self.fillers_left}] fillers_right=[{self.fillers_right}]')
        return attack_feasible
    
    def side_with_more_weight(self) -> str:
        left_balance, right_balance = self.weight_for_side()
        if left_balance > right_balance:
            return 'left'
        elif right_balance > left_balance:
            return 'right'
        else:
            return 'draw'
    
    def number_of_fillers_needed(self) -> int:
        left_balance, right_balance = self.weight_for_side()
        if left_balance == right_balance:
            return 0
        elif left_balance > right_balance:
            difference = left_balance - right_balance
        else:
            difference = right_balance - left_balance
        return difference // spec.Gwei(32 * 10**9)
    
    def number_of_extra_swayers(self) -> int:
        return 0

    def assign(self, validators: Sequence[spec.ValidatorIndex], slot: spec.Slot) -> Tuple[Sequence[spec.ValidatorIndex], Sequence[spec.ValidatorIndex], Sequence[spec.ValidatorIndex]]:
        is_last_slot_of_epoch = (slot + spec.Slot(1)) % spec.SLOTS_PER_EPOCH == 0
        validators = tuple(validators)
        validators_len = len(validators)
        validators_len_even = validators_len % 2 == 0
        
        swayers_begin = 2 + 2 + (4 * self.number_of_extra_swayers())
        swayers_end = 2 + (2 * self.number_of_extra_swayers())
        swayers = swayers_end if is_last_slot_of_epoch else swayers_begin
        fillers = self.number_of_fillers_needed()

        swayers_for_slot = swayers // 2
        swayers_per_direction = swayers_for_slot // 2

        swayers_dicts = [self.swayers_subsequent_epochs_left, self.swayers_subsequent_epochs_right]
        if not is_last_slot_of_epoch:
            swayers_dicts.append(self.swayers_first_epoch_left)
            swayers_dicts.append(self.swayers_first_epoch_right)
        
        for swayers_dict in swayers_dicts:
            swayers_dict[slot] = list()
        
        validator_index = 0
        dict_index = 0
        while sum(len(swayers_dict[slot]) for swayers_dict in swayers_dicts) < swayers:
            swayers_dicts[dict_index][slot].append(validators[validator_index])
            validator_index += 1
            dict_index = (dict_index + 1) % len(swayers_dicts)
        
        if not validators_len_even:
            validators_len -= 1
        
        self.fillers_left[slot] = []
        self.fillers_right[slot] = []

        fillers_needed = self.number_of_fillers_needed()
        if fillers_needed > 0:
            more_weight_on = self.side_with_more_weight()
            side_to_be_filled = self.fillers_left if more_weight_on == 'right' else self.fillers_right
            for i in range(0, fillers_needed):
                side_to_be_filled[slot].append(validators[validator_index])
                validator_index += 1

        validators_left = len(validators) - validator_index
        validators_left = validators_left if validators_left % 2 == 0 else validators_left - 1
        for validator_index_filler in range(validator_index, validators_left, 2):
            self.fillers_left[slot].append(validators[validator_index_filler])
            self.fillers_right[slot].append(validators[validator_index_filler + 1])
    
    def propose_block(self, validator: Validator, head_state: spec.BeaconState, slashme=False):
        if self.attack_state == self.AttackState.ATTACK_STARTED_FIRST_EPOCH and self.current_slot == self.attack_started_slot:
            self.propose_blocks_start_attack(validator, head_state)
        elif self.attack_state == self.AttackState.ATTACK_STARTED_FIRST_EPOCH or self.attack_state == self.AttackState.ATTACK_STARTED_SUBSEQUENT_EPOCHS:
            # TODO Calculate an head state depending on the proposer being left or right
            super().propose_block(validator, head_state)
        else:
            super().propose_block(validator, head_state)
    
    def attest(self):
        pass
    
    def propose_blocks_start_attack(self, validator: Validator, head_state: spec.BeaconState):
        block_left = spec.BeaconBlock(
            slot=head_state.slot,
            proposer_index=validator.index,
            parent_root=self.head_root,
            state_root=spec.Bytes32(bytes(0 for _ in range(0, 32)))
        )
        block_right = block_left.copy()

        randao_reveal = spec.get_epoch_signature(head_state, block_left, validator.privkey)
        eth1_data = head_state.eth1_data
        candidate_attestations = list()
        proposer_slashings: List[spec.ProposerSlashing, spec.MAX_PROPOSER_SLASHINGS] = list()
        attester_slashings: List[spec.AttesterSlashing, spec.MAX_ATTESTER_SLASHINGS] = list()
        deposits: List[spec.Deposit, spec.MAX_DEPOSITS] = list()
        voluntary_exits: List[spec.VoluntaryExit, spec.MAX_VOLUNTARY_EXITS] = list()

        body_left = spec.BeaconBlockBody(
            randao_reveal=randao_reveal,
            eth1_data=eth1_data,
            graffiti=spec.Bytes32(bytes(0 for _ in range(0, 32))),
            proposer_slashings=proposer_slashings,
            attester_slashings=attester_slashings,
            attestations=candidate_attestations,
            deposits=deposits,
            voluntary_exits=voluntary_exits
        )
        body_right = body_left.copy()
        body_right.graffiti = spec.Bytes32(bytes(1 for _ in range(0, 32)))

        block_left.body = body_left
        block_right.body = body_right

        try:
            new_state_root_left = spec.compute_new_state_root(self.state, block_left)
            new_state_root_right = spec.compute_new_state_root(self.state, block_right)
        except AssertionError:
            validator_status = [str(validator) for validator in self.state.validators if validator.slashed]
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            self.__debug({'text': text, 'block': str(block), 'validators': str(validator_status)}, 'FindNewStateRootError')
            self.client_to_simulator_queue.put(SimulationEndEvent(time=self.current_time, priority=0, message=text))
            return
        
        block_left.state_root = new_state_root_left
        block_right.state_root = new_state_root_right

        signed_block_left = spec.SignedBeaconBlock(
            message=block_left,
            signature=spec.get_block_signature(self.state, block_left, validator.privkey)
        )
        signed_block_right = spec.SignedBeaconBlock(
            message=block_right,
            signature=spec.get_block_signature(self.state, block_right, validator.privkey)
        )

        self.block_root_left = spec.hash_tree_root(block_left)
        self.block_root_right = spec.hash_tree_root(block_right)

        print(f"block_root_left=[{self.block_root_left}] block_root_right=[{self.block_root_right}]")

        encoded_signed_block_left = signed_block_left.encode_bytes()
        encoded_signed_block_right = signed_block_right.encode_bytes()

        message_left = MessageEvent(
            time=self.current_time,
            priority=20,
            message=encoded_signed_block_left,
            message_type='SignedBeaconBlock',
            fromidx=self.counter,
            toidx=self.counter
        )
        message_right = MessageEvent(
            time=self.current_time,
            priority=20,
            message=encoded_signed_block_right,
            message_type='SignedBeaconBlock',
            fromidx=self.counter,
            toidx=self.counter
        )
        self.client_to_simulator_queue.put(message_left)
        self.client_to_simulator_queue.put(message_right)

        for beacon_client_map in self.beacon_clients_left:
            message_left_to_left = MessageEvent(
                time=self.current_time,
                priority=20,
                message=encoded_signed_block_left,
                message_type='SignedBeaconBlock',
                fromidx=self.counter,
                toidx=beacon_client_map.beacon_client_counter,
                custom_latency=0
            )
            message_right_to_left = MessageEvent(
                time=self.current_time,
                priority=20,
                message=encoded_signed_block_right,
                message_type='SignedBeaconBlock',
                fromidx=self.counter,
                toidx=beacon_client_map.beacon_client_counter,
                custom_latency=spec.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 6)
            )
            self.client_to_simulator_queue.put(message_left_to_left)
            self.client_to_simulator_queue.put(message_right_to_left)
        
        for beacon_client_map in self.beacon_clients_right:
            message_right_to_right = MessageEvent(
                time=self.current_time,
                priority=20,
                message=encoded_signed_block_right,
                message_type='SignedBeaconBlock',
                fromidx=self.counter,
                toidx=beacon_client_map.beacon_client_counter,
                custom_latency=0
            )
            message_left_to_right = MessageEvent(
                time=self.current_time,
                priority=20,
                message=encoded_signed_block_left,
                message_type='SignedBeaconBlock',
                fromidx=self.counter,
                toidx=beacon_client_map.beacon_client_counter,
                custom_latency=spec.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 6)
            )
            self.client_to_simulator_queue.put(message_right_to_right)
            self.client_to_simulator_queue.put(message_left_to_right)

    def attack_slot(self) -> spec.Slot:
        return self.current_slot - self.attack_started_slot
    
    def compute_head(self):
        super().compute_head()
