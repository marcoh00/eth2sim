import sys
import traceback

from queue import Queue
from eth2spec.phase0 import spec
from events import (
    AggregateOpportunity,
    Event,
    LatestVoteOpportunity,
    NextSlotEvent,
    SimulationEndEvent,
    BeaconClientInfo,
    MessageEvent,
)
from beaconclient import BeaconClient
from typing import Optional, Sequence, Dict, Tuple, List, Set
from enum import Enum
from validator import Validator
from dataclasses import dataclass


class TimeAttackedBeaconClient(BeaconClient):
    def __init__(self, *kargs, **kwargs):
        super().__init__(
            counter=kwargs["counter"],
            simulator_to_client_queue=kwargs["simulator_to_client_queue"],
            client_to_simulator_queue=kwargs["client_to_simulator_queue"],
            configpath=kwargs["configpath"],
            configname=kwargs["configname"],
            validator_builders=kwargs["validator_builders"],
            validator_first_counter=kwargs["validator_first_counter"],
            debug=kwargs["debug"],
            profile=kwargs["profile"],
        )
        self.attack_start_slot = kwargs["attack_start_slot"]
        self.attack_end_slot = kwargs["attack_end_slot"]
        self.timedelta = kwargs["timedelta"]
        self.attack_started = False
        self.is_first_attack_slot = False
        self.is_first_synced_slot = False

    def pre_event_handling(self, message: Event):
        if self.attack_started and type(message) in (
            LatestVoteOpportunity,
            AggregateOpportunity,
        ):
            message.time += self.timedelta * spec.SECONDS_PER_SLOT
            message.slot += self.timedelta

    def pre_next_slot_event(self, message: NextSlotEvent):
        # Update internal state of attack
        if self.attack_start_slot <= message.slot < self.attack_end_slot:
            self.is_first_attack_slot = False
            if not self.attack_started:
                self.attack_started = True
                self.is_first_attack_slot = True
            print(
                f"[BEACON CLIENT {self.counter}] Time attacked. Regular: time=[{message.time}] slot=[{message.slot}]"
            )
            message.time += self.timedelta * spec.SECONDS_PER_SLOT
            message.slot += self.timedelta
            self.current_slot = message.slot
            print(
                f"[BEACON CLIENT {self.counter}] Time attacked. Attacked: time=[{message.time}] slot=[{message.slot}]"
            )
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
        for attestation in self.attestation_cache.attestations_not_known_to_forkchoice(
            spec.Slot(0), self.current_slot
        ):
            spec.store_target_checkpoint_state(self.store, attestation.data.target)

    def post_next_slot_event(self, head_state, indexed_validators):
        if self.is_first_attack_slot:
            print(
                f"[BEACON CLIENT {self.counter}] Time attack started with delta {self.timedelta}"
            )
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
            self.is_first_attack_slot = False

        if self.is_first_synced_slot:
            print(f"[BEACON CLIENT {self.counter}] Time attack stopped")
            self.committee = dict()
            self.slot_last_attested = None
            self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
            self.attack_started = False
            self.is_first_synced_slot = False

        print(
            f"[BEACON CLIENT {self.counter}] Time-attacked client is at slot {self.current_slot} right now"
        )
        return super().post_next_slot_event(head_state, indexed_validators)

    def produce_attestation_data(
        self,
        validator_index,
        validator_committee,
        epoch_boundary_block_root,
        head_block,
        head_state,
    ) -> Optional[spec.AttestationData]:
        attestation = super().produce_attestation_data(
            validator_index,
            validator_committee,
            epoch_boundary_block_root,
            head_block,
            head_state,
        )
        print(
            f"[BEACON CLIENT {self.counter}] Time-attacked client produces attestation: slot=[{attestation.slot}] srcep=[{attestation.source.epoch}] targetep=[{attestation.target.epoch}] validator=[{validator_index}]"
        )
        return attestation

    def pre_attest(self):
        self.update_committee(spec.compute_epoch_at_slot(self.current_slot))

    def pre_propose(self):
        self.update_committee(spec.compute_epoch_at_slot(self.current_slot))


class BalancingAttackingBeaconClient(BeaconClient):
    class AttackState(Enum):
        ATTACK_PLANNED = 0
        ATTACK_STARTED_FIRST_EPOCH_FIRST_SLOT = 1
        ATTACK_STARTED_FIRST_EPOCH = 2
        ATTACK_STARTED_SUBSEQUENT_EPOCHS = 3
        ATTACK_STOPPED = 4

        @classmethod
        def attack_running(cls, instance):
            return instance in (
                cls.ATTACK_STARTED_FIRST_EPOCH,
                cls.ATTACK_STARTED_FIRST_EPOCH_FIRST_SLOT,
                cls.ATTACK_STARTED_SUBSEQUENT_EPOCHS,
            )

    @dataclass
    class ClientWeightMap:
        beacon_client_counter: int
        weight: spec.Gwei

    attack_state: AttackState
    attack_started_slot: Optional[spec.Slot]

    beacon_client_validator_map: Dict[int, Sequence[spec.ValidatorIndex]]
    validator_beacon_client_map: Dict[spec.ValidatorIndex, int]

    all_clients_left: Set[int]
    all_clients_right: Set[int]

    committee_left: Dict[
        spec.Slot, Dict[spec.ValidatorIndex, Tuple[spec.CommitteeIndex, int, int]]
    ]
    committee_count_left: Dict[spec.Epoch, int]

    committee_right: Dict[
        spec.Slot, Dict[spec.ValidatorIndex, Tuple[spec.CommitteeIndex, int, int]]
    ]
    committee_count_right: Dict[spec.Epoch, int]

    # Slot indices indicate actual slots
    beacon_clients_left: Dict[spec.Slot, Sequence[int]]
    beacon_clients_right: Dict[spec.Slot, Sequence[int]]
    beacon_clients_neither: Dict[spec.Slot, Sequence[int]]
    fillers_needed_left: Dict[spec.Slot, int]
    fillers_needed_right: Dict[spec.Slot, int]

    # Slot indices indicate number of slot inside epoch
    swayers_first_epoch_left: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    swayers_first_epoch_right: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]

    swayers_subsequent_epochs_left: Set[spec.ValidatorIndex]
    swayers_subsequent_epochs_right: Set[spec.ValidatorIndex]
    unparticipating_validators: Set[spec.ValidatorIndex]

    fillers_first_epoch: Dict[spec.Slot, Sequence[spec.ValidatorIndex]]
    fillers_subsequent_epochs: Set[spec.ValidatorIndex]

    attestations_saved_left: Dict[
        spec.Epoch, Queue[Tuple[spec.Attestation, spec.ValidatorIndex]]
    ]
    attestations_saved_right: Dict[
        spec.Epoch, Queue[Tuple[spec.Attestation, spec.ValidatorIndex]]
    ]

    state_left: Optional[spec.BeaconState]
    head_state_left: Optional[spec.BeaconState]

    state_right: Optional[spec.BeaconState]
    head_state_right: Optional[spec.BeaconState]

    block_root_left: Optional[spec.Root]
    block_root_right: Optional[spec.Root]
    first_block_root_left: Optional[spec.Root]
    first_block_root_right: Optional[spec.Root]

    def __init__(self, *kargs, **kwargs):
        super().__init__(
            counter=kwargs["counter"],
            simulator_to_client_queue=kwargs["simulator_to_client_queue"],
            client_to_simulator_queue=kwargs["client_to_simulator_queue"],
            configpath=kwargs["configpath"],
            configname=kwargs["configname"],
            validator_builders=kwargs["validator_builders"],
            validator_first_counter=kwargs["validator_first_counter"],
            debug=kwargs["debug"],
            profile=kwargs["profile"],
        )
        self.attack_state = self.AttackState.ATTACK_PLANNED
        self.attack_started_slot = None
        self.beacon_client_validator_map = dict()
        self.validator_beacon_client_map = dict()
        self.committee_left = dict()
        self.committee_right = dict()
        self.committee_count_left = dict()
        self.committee_count_right = dict()
        self.all_clients_left = set()
        self.all_clients_right = set()
        self.beacon_clients_left = dict()
        self.beacon_clients_right = dict()
        self.beacon_clients_neither = dict()
        self.fillers_needed_left = dict()
        self.fillers_needed_right = dict()
        self.unparticipating_validators = set()
        self.swayers_first_epoch_left = dict()
        self.swayers_first_epoch_right = dict()
        self.swayers_subsequent_epochs_left = set()
        self.swayers_subsequent_epochs_right = set()
        self.fillers_first_epoch = dict()
        self.fillers_subsequent_epochs = set()
        self.attestations_saved_left = dict()
        self.attestations_saved_right = dict()
        self.state_left = None
        self.head_state_left = None
        self.state_right = None
        self.head_state_right = None
        self.block_root_left = None
        self.block_root_right = None
        self.first_block_root_left = None
        self.first_block_root_right = None

    def handle_beacon_client_info(self, event: BeaconClientInfo):
        for beacon_client_counter in event.beacon_clients.keys():
            if beacon_client_counter == self.counter:
                continue
            self.beacon_client_validator_map[beacon_client_counter] = tuple(
                spec.ValidatorIndex(index)
                for index in event.beacon_clients[beacon_client_counter]
            )
            for index in event.beacon_clients[beacon_client_counter]:
                self.validator_beacon_client_map[
                    spec.ValidatorIndex(index)
                ] = beacon_client_counter

    def determine_positition_of_beacon_clients(self, slot: spec.Slot):
        self.beacon_clients_left[slot] = list()
        self.beacon_clients_right[slot] = list()

        honest_indices_inside_slot_committees = tuple(
            set(self.committee_left[slot].keys())
            .union(self.committee_right[slot].keys())
            .difference(self.indexed_validators.keys())
        )
        honest_clients_inside_slot_committees = tuple(
            set(
                self.validator_beacon_client_map[index]
                for index in honest_indices_inside_slot_committees
            )
        )

        # We assert that one beacon client hosts exactly one validator
        # TODO remove this constraint and try to construct two buckets
        assert len(honest_clients_inside_slot_committees) == len(
            honest_indices_inside_slot_committees
        )

        unknown_side_clients = list()
        for honest_index in honest_clients_inside_slot_committees:
            if honest_index in self.all_clients_left:
                self.beacon_clients_left[slot].append(honest_index)
            elif honest_index in self.all_clients_right:
                self.beacon_clients_right[slot].append(honest_index)
            else:
                unknown_side_clients.append(honest_index)

        unknown_side_clients_len = len(unknown_side_clients)
        unknown_side_clients_len_even = unknown_side_clients_len % 2 == 0
        unknown_side_clients_indices_upto = (
            unknown_side_clients_len
            if unknown_side_clients_len_even
            else unknown_side_clients_len - 1
        )
        for i in range(0, unknown_side_clients_indices_upto, 2):
            self.beacon_clients_left[slot].append(
                honest_clients_inside_slot_committees[i]
            )
            self.all_clients_left.add(honest_clients_inside_slot_committees[i])
            self.beacon_clients_right[slot].append(
                honest_clients_inside_slot_committees[i + 1]
            )
            self.all_clients_right.add(honest_clients_inside_slot_committees[i + 1])

        self.fillers_needed_left[slot] = 0
        self.fillers_needed_right[slot] = 0
        if not unknown_side_clients_len_even:
            self.beacon_clients_left[slot].append(
                honest_clients_inside_slot_committees[-1]
            )
            self.all_clients_left.add(honest_clients_inside_slot_committees[-1])
            self.fillers_needed_right[slot] = 1
        self.beacon_clients_neither[slot] = tuple(
            set(self.beacon_client_validator_map.keys())
            .difference(honest_clients_inside_slot_committees)
            .difference((self.counter,))
        )

        self.log(
            {
                "slot": slot,
                "beacon_clients_left": self.beacon_clients_left[slot],
                "beacon_clients_right": self.beacon_clients_right[slot],
                "all_clients_left": self.all_clients_left,
                "all_clients_right": self.all_clients_right,
                "validators_left": tuple(
                    self.beacon_client_validator_map[beacon_client]
                    for beacon_client in self.beacon_clients_left[slot]
                ),
                "validators_right": tuple(
                    self.beacon_client_validator_map[beacon_client]
                    for beacon_client in self.beacon_clients_right[slot]
                ),
                "beacon_clients_neither": self.beacon_clients_neither[slot],
                "fillers_needed_left": self.fillers_needed_left[slot],
                "fillers_needed_right": self.fillers_needed_right[slot],
            },
            "BeaconClientPosition",
        )

    def weight_for_side(self) -> Tuple[spec.Gwei, spec.Gwei]:
        left = sum(client.weight for client in self.beacon_clients_left)
        right = sum(client.weight for client in self.beacon_clients_right)
        return left, right

    def update_weights(self, clients: Sequence[ClientWeightMap]):
        active_indices = spec.get_active_validator_indices(
            self.head_state, spec.get_current_epoch(self.head_state)
        )
        for client in clients:
            client.weight = spec.Gwei(
                sum(
                    self.head_state.validators[validator_index].effective_balance
                    for validator_index in active_indices
                    if validator_index
                    in self.beacon_client_validator_map[client.beacon_client_counter]
                )
            )

    def on_epoch_start(self):
        epoch = spec.compute_epoch_at_slot(self.current_slot)
        assert spec.compute_start_slot_at_epoch(epoch) == self.current_slot
        self.attestations_saved_left[epoch] = Queue()
        self.attestations_saved_right[epoch] = Queue()
        self.update_committee(spec.compute_epoch_at_slot(self.current_slot))
        self.update_committee(
            spec.compute_epoch_at_slot(self.current_slot + spec.SLOTS_PER_EPOCH)
        )
        for slot_i in range(
            self.current_slot, self.current_slot + (2 * spec.SLOTS_PER_EPOCH)
        ):
            self.determine_positition_of_beacon_clients(spec.Slot(slot_i))
        if self.attack_state == self.AttackState.ATTACK_PLANNED:
            feasible = self.is_attack_feasible()
            if feasible:
                # self.client_to_simulator_queue.put(SimulationEndEvent(self.current_time, 0, feasible))
                self.attack_state = (
                    self.AttackState.ATTACK_STARTED_FIRST_EPOCH_FIRST_SLOT
                )
                self.attack_started_slot = self.current_slot.copy()
        if self.attack_state == self.AttackState.ATTACK_STARTED_FIRST_EPOCH:
            self.attack_state = self.AttackState.ATTACK_STARTED_SUBSEQUENT_EPOCHS

    def pre_next_slot_event(self, message: MessageEvent):
        if self.attack_state == self.AttackState.ATTACK_STARTED_FIRST_EPOCH_FIRST_SLOT:
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
        proposer_adversarial = (
            proposer_index in self.indexed_validators
            and not self.slashed(proposer_index)
        )

        # The committee for the whole epoch has been calculated by now so we can check if there are enough validators inside the current slot

        adversarial_validators_for_first_slot = tuple(
            self.attesting_validators_at_slot(self.current_slot).keys()
        )
        enough_adversarial_validators_for_all_but_last_slot_condition = (
            2
            + 2
            + self.number_of_fillers_needed()
            + (4 * self.number_of_extra_swayers())
        )
        enough_adversarial_validators_for_first_slot = (
            len(adversarial_validators_for_first_slot)
            >= enough_adversarial_validators_for_all_but_last_slot_condition
        )

        # Check if all slots from the second till the one before the last have at least 4 swayers and the needed fillers
        first_slot_of_epoch = self.current_slot.copy()
        assert (
            spec.compute_start_slot_at_epoch(
                spec.compute_epoch_at_slot(first_slot_of_epoch)
            )
            == first_slot_of_epoch
        )
        last_slot_of_epoch = first_slot_of_epoch + spec.SLOTS_PER_EPOCH - 1

        adversial_validators_for_middle_slots = [
            set(self.attesting_validators_at_slot(slot).keys()).difference(
                (proposer_index,)
            )
            for slot in range(first_slot_of_epoch + 1, last_slot_of_epoch)
        ]
        enough_adversarial_validators_for_middle_slots = all(
            len(adversial_validators_for_middle_slot)
            >= enough_adversarial_validators_for_all_but_last_slot_condition
            for adversial_validators_for_middle_slot in adversial_validators_for_middle_slots
        )

        # Check if the last slot has enough adversarial validators (two swayers plus needed fillers)

        adversarial_validators_for_last_slot = set(
            self.attesting_validators_at_slot(last_slot_of_epoch).keys()
        ).difference((proposer_index,))
        enough_adversarial_validators_for_last_slot = len(
            adversarial_validators_for_last_slot
        ) >= 2 + self.number_of_fillers_needed() + (2 * self.number_of_extra_swayers())

        attack_feasible = (
            proposer_adversarial
            and enough_adversarial_validators_for_first_slot
            and enough_adversarial_validators_for_middle_slots
            and enough_adversarial_validators_for_last_slot
        )
        print(
            f"[BEACON CLIENT {self.counter}] Feasability Check: feasible=[{attack_feasible}] | proposer=[{proposer_adversarial}] first=[{enough_adversarial_validators_for_first_slot}] middle=[{enough_adversarial_validators_for_middle_slots}] last=[{enough_adversarial_validators_for_last_slot}]"
        )

        if attack_feasible:
            self.assign(adversarial_validators_for_first_slot, spec.Slot(0))
            for slot_minus_one, validators in enumerate(
                adversial_validators_for_middle_slots
            ):
                slot = spec.Slot(slot_minus_one + 1)
                self.assign(validators, slot)
            self.assign(
                adversarial_validators_for_last_slot,
                spec.Slot(spec.SLOTS_PER_EPOCH - spec.Slot(1)),
            )
            print(
                f"[BEACON CLIENT 0] swayers_left_first=[{self.swayers_first_epoch_left}] swayers_right_first=[{self.swayers_first_epoch_right}] swayers_left_subsequent=[{self.swayers_subsequent_epochs_left}] swayers_right_subsequent=[{self.swayers_subsequent_epochs_right}] fillers_first_epoch=[{self.fillers_first_epoch}] fillers_subsequent_epochs=[{self.fillers_subsequent_epochs}] unparticipating=[{self.unparticipating_validators}]"
            )
            self.log(
                {
                    "swayers_left_first": self.swayers_first_epoch_left,
                    "swayers_right_first": self.swayers_first_epoch_right,
                    "swayers_left_subsequent": self.swayers_subsequent_epochs_left,
                    "swayers_right_subsequent": self.swayers_subsequent_epochs_right,
                    "fillers_first_epoch": self.fillers_first_epoch,
                    "fillers_subsequent_epoch": self.fillers_subsequent_epochs,
                    "unparticipating": self.unparticipating_validators,
                },
                "AttackValidatorAssignment",
            )
        return attack_feasible

    def side_with_more_weight(self) -> str:
        left_balance, right_balance = self.weight_for_side()
        if left_balance > right_balance:
            return "left"
        elif right_balance > left_balance:
            return "right"
        else:
            return "draw"

    def number_of_fillers_needed(self) -> int:
        return 2

    def number_of_extra_swayers(self) -> int:
        return 0

    def assign(
        self, validators: Sequence[spec.ValidatorIndex], slot: spec.Slot
    ) -> Tuple[
        Sequence[spec.ValidatorIndex],
        Sequence[spec.ValidatorIndex],
        Sequence[spec.ValidatorIndex],
    ]:
        is_last_slot_of_epoch = (slot + spec.Slot(1)) % spec.SLOTS_PER_EPOCH == 0
        validators = tuple(validators)

        swayers_begin = 2 + 2 + (4 * self.number_of_extra_swayers())
        swayers_end = 2 + (2 * self.number_of_extra_swayers())
        swayers = swayers_end if is_last_slot_of_epoch else swayers_begin
        fillers = self.number_of_fillers_needed()

        swayers_for_slot = swayers // 2
        swayers_per_direction = swayers_for_slot // 2

        if not is_last_slot_of_epoch:
            # Here we assign swayers per epoch
            collections = (
                self.swayers_first_epoch_left,
                self.swayers_first_epoch_right,
            )
            for swayer_collection in collections:
                swayer_collection[slot] = list()
            # Assign swayers // 2 swayers to these collections
            for swayeridx in range(0, swayers // 2):
                collections[swayeridx % 2][slot].append(validators[swayeridx])
        # Assign validator ids for subsequent epochs
        start = 0 if is_last_slot_of_epoch else swayers // 2
        collections = (
            self.swayers_subsequent_epochs_left,
            self.swayers_subsequent_epochs_right,
        )
        for swayeridx in range(start, swayers, 1):
            collections[swayeridx % 2].add(validators[swayeridx])

        # TODO call number_of_fillers_needed and add the appropriate amount of fillers
        self.fillers_first_epoch[slot] = [
            validators[swayers],
        ]
        self.fillers_subsequent_epochs.add(validators[swayers + 1])
        for unparticipating in validators[swayers + 1 :]:
            self.unparticipating_validators.add(unparticipating)

    def propose_block(
        self, validator: Validator, head_state: spec.BeaconState, slashme=False
    ):
        if self.attack_state == self.AttackState.ATTACK_STARTED_FIRST_EPOCH_FIRST_SLOT:
            self.propose_blocks_start_attack(validator, head_state)
        elif self.AttackState.attack_running(self.attack_state):
            # TODO Calculate an head state depending on the proposer being left or right
            super().propose_block(validator, head_state)
        else:
            super().propose_block(validator, head_state)

    def handle_aggregate_opportunity(self, message: AggregateOpportunity):
        pass

    def attest(self):
        """
        First epoch:
        - Fillers which actually need to fill for first epoch instantly attest to their side
        - Fillers which dont need to fill cache their attestations, if they weren't needed,
          they are released at the end of the epoch to everyone
        - Swayers for first epoch send their messages just before NSE
          to the correct side and at the end of the slot to the other side
        - Fillers for subsequent epochs vote for their side and save their votes.
          If the votes were not needed they are released just before the last slot in the following epoch
        - Swayers for subsequent epochs vote and save their messages.
          The votes are used in the next epoch
        Subsequent epoch:
        - Fillers vote for their side and save the vote
        - Swayers vote for their side and save the vote
        - Fillers and swayers votes from the previous epoch are transmitted as needed
        """
        self.compute_head()

        if not self.AttackState.attack_running(self.attack_state):
            super().attest()
        elif self.attack_state == self.AttackState.ATTACK_STARTED_SUBSEQUENT_EPOCHS:
            self.save_votes()
            self.attest_subsequent_epoch()
        elif True:
            self.save_votes()
            self.attest_first_epoch()

    def get_attestation_left(
        self, validator_index: spec.ValidatorIndex
    ) -> spec.Attestation:
        validator = self.indexed_validators[validator_index]
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)

        start_slot = spec.compute_start_slot_at_epoch(current_epoch)
        epoch_boundary_block_root = (
            self.block_root_left
            if start_slot == self.current_slot
            else spec.get_block_root(self.head_state_left, current_epoch)
        )

        validator_committee = self.committee_left[self.current_slot][validator_index][0]
        attestation_data = self.produce_attestation_data(
            validator_index,
            validator_committee,
            epoch_boundary_block_root,
            self.block_root_left,
            self.head_state_left,
        )
        return self.produce_attestation(
            attestation_data,
            validator_index,
            validator,
            custom_committee=self.committee_left,
        )

    def get_attestation_right(
        self, validator_index: spec.ValidatorIndex
    ) -> spec.Attestation:
        validator = self.indexed_validators[validator_index]
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)

        start_slot = spec.compute_start_slot_at_epoch(current_epoch)
        epoch_boundary_block_root = (
            self.block_root_right
            if start_slot == self.current_slot
            else spec.get_block_root(self.head_state_right, current_epoch)
        )

        validator_committee = self.committee_right[self.current_slot][validator_index][
            0
        ]
        attestation_data = self.produce_attestation_data(
            validator_index,
            validator_committee,
            epoch_boundary_block_root,
            self.block_root_right,
            self.head_state_right,
        )
        return self.produce_attestation(
            attestation_data,
            validator_index,
            validator,
            custom_committee=self.committee_right,
        )

    def save_votes(self):
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        reverse_swayer_side = current_epoch % 2 == 0

        for attesting_validator_left in self.attesting_validators_at_slot(
            self.current_slot, committee=self.committee_left
        ).keys():
            assert self.head_state_left.slot == self.current_slot
            assert (
                self.head_state_left.latest_block_header.slot
                == self.store.blocks[self.block_root_left].slot
            )
            swayer_set = (
                self.swayers_subsequent_epochs_right
                if reverse_swayer_side
                else self.swayers_subsequent_epochs_left
            )
            if attesting_validator_left in swayer_set:
                self.attestations_saved_left[current_epoch].put(
                    (
                        self.get_attestation_left(attesting_validator_left),
                        attesting_validator_left,
                    )
                )

        for attesting_validator_right in self.attesting_validators_at_slot(
            self.current_slot, committee=self.committee_right
        ).keys():
            assert self.head_state_right.slot == self.current_slot
            assert (
                self.head_state_right.latest_block_header.slot
                == self.store.blocks[self.block_root_right].slot
            )
            swayer_set = (
                self.swayers_subsequent_epochs_left
                if reverse_swayer_side
                else self.swayers_subsequent_epochs_right
            )
            if attesting_validator_right in swayer_set:
                self.attestations_saved_right[current_epoch].put(
                    (
                        self.get_attestation_right(attesting_validator_right),
                        attesting_validator_right,
                    )
                )

        self.log(
            {
                "epoch": current_epoch,
                "left": self.attestations_saved_left[current_epoch].qsize(),
                "right": self.attestations_saved_right[current_epoch].qsize(),
                "reversed": reverse_swayer_side,
            },
            "SavedAttestationsAvailable",
        )
        print(
            f"attestations_left=[{self.attestations_saved_left[current_epoch].qsize()}] attestations_right=[{self.attestations_saved_right[current_epoch].qsize()}]"
        )

    def attest_first_epoch(self):
        slot_inside_epoch = self.current_slot - self.attack_started_slot
        # Fillers for current slot
        for _ in range(0, self.fillers_needed_left[self.current_slot]):
            validator = self.fillers_first_epoch[slot_inside_epoch].pop()
            attestation = self.get_attestation_left(validator)
            self.send_message(
                message_type="Attestation", message=attestation, toidx=None
            )
            self.log(
                {"validator": validator, "attestation": attestation},
                "CurrentEpochFillerLeft",
            )
        for _ in range(0, self.fillers_needed_right[self.current_slot]):
            validator = self.fillers_first_epoch[slot_inside_epoch].pop()
            attestation = self.get_attestation_right(validator)
            self.send_message(
                message_type="Attestation", message=attestation, toidx=None
            )
            self.log(
                {"validator": validator, "attestation": attestation},
                "CurrentEpochFillerRight",
            )

        # Swayers for next slot
        if slot_inside_epoch != spec.SLOTS_PER_EPOCH - 1:
            assert len(self.swayers_first_epoch_left[slot_inside_epoch]) == len(
                self.swayers_first_epoch_right[slot_inside_epoch]
            )
            for validator_left, validator_right in zip(
                self.swayers_first_epoch_left[slot_inside_epoch],
                self.swayers_first_epoch_right[slot_inside_epoch],
            ):
                attestation_left = self.get_attestation_left(validator_left)
                attestation_right = self.get_attestation_right(validator_right)
                self.distribute_targeted_message(
                    message_type="Attestation",
                    left_message=attestation_left.encode_bytes(),
                    right_message=attestation_right.encode_bytes(),
                    latency=4,
                    latency_otherside=10,
                    priority=-1,
                    target_slot=self.current_slot + 1,
                )
                self.log(
                    {"validator": validator_left, "attestation": attestation_left},
                    "CurrentEpochSwayerLeft",
                )
                self.log(
                    {"validator": validator_right, "attestation": attestation_right},
                    "CurrentEpochSwayerRight",
                )
        else:
            # This is the last slot of the epoch and every swayer for subsequent epochs
            # should have already saved one vote by now.
            # This means we can already use two saved votes to sway the honest clients.
            current_epoch = spec.compute_epoch_at_slot(self.current_slot)
            swayers_per_side = (
                len(self.swayers_subsequent_epochs_left) // spec.SLOTS_PER_EPOCH
            )
            for _ in range(0, swayers_per_side):
                attestation_left, validator_left = self.attestations_saved_left[
                    current_epoch
                ].get()
                attestation_right, validator_right = self.attestations_saved_right[
                    current_epoch
                ].get()
                self.distribute_targeted_message(
                    message_type="Attestation",
                    left_message=attestation_left.encode_bytes(),
                    right_message=attestation_right.encode_bytes(),
                    latency=4,
                    latency_otherside=10,
                    priority=-1,
                    target_slot=self.current_slot + 1,
                )
                self.log(
                    {"validator": validator_left, "attestation": attestation_left},
                    "CurrentEpochSwayerLeftSavedLastSlot",
                )
                self.log(
                    {"validator": validator_right, "attestation": attestation_right},
                    "CurrentEpochSwayerRightSavedLastSlot",
                )

    def attest_subsequent_epoch(self):
        current_epoch = spec.compute_epoch_at_slot(self.current_slot)
        slot_inside_epoch = self.current_slot % spec.SLOTS_PER_EPOCH
        attestation_epoch = (
            current_epoch
            if slot_inside_epoch == (spec.SLOTS_PER_EPOCH - 1)
            else current_epoch - 1
        )

        self.log(attestation_epoch, "AttestationEpochForSubsequentEpoch")

        for _ in range(0, self.fillers_needed_left[self.current_slot]):
            attestation, validator = self.attestations_saved_left[
                attestation_epoch
            ].get()
            self.send_message(
                message_type="Attestation", message=attestation, toidx=None
            )
            self.log(
                {"validator": validator, "attestation": attestation},
                "SubsequentEpochFillerLeft",
            )
        for _ in range(0, self.fillers_needed_right[self.current_slot]):
            attestation, validator = self.attestations_saved_right[
                attestation_epoch
            ].get()
            self.send_message(
                message_type="Attestation", message=attestation, toidx=None
            )
            self.log(
                {"validator": validator, "attestation": attestation},
                "SubsequentEpochFillerRight",
            )

        swayers_per_side = (
            len(self.swayers_subsequent_epochs_left) // spec.SLOTS_PER_EPOCH
        )
        for _ in range(0, swayers_per_side):
            attestation_left, validator_left = self.attestations_saved_left[
                attestation_epoch
            ].get()
            attestation_right, validator_right = self.attestations_saved_right[
                attestation_epoch
            ].get()
            self.distribute_targeted_message(
                message_type="Attestation",
                left_message=attestation_left.encode_bytes(),
                right_message=attestation_right.encode_bytes(),
                latency=4,
                latency_otherside=10,
                priority=-1,
                target_slot=self.current_slot + 1,
            )
            self.log(
                {"validator": validator_left, "attestation": attestation_left},
                "SubsequentEpochSwayerLeft",
            )
            self.log(
                {"validator": validator_right, "attestation": attestation_right},
                "SubsequentEpochSwayerRight",
            )

    def send_message(self, message_type, message, toidx=None, latency=0, priority=30):
        message = MessageEvent(
            time=self.current_time,
            priority=priority,
            message=message.encode_bytes(),
            message_type=message_type,
            fromidx=self.counter,
            toidx=toidx,
            custom_latency=latency,
        )
        self.log(
            {
                "from": self.counter,
                "to": toidx,
                "latency": latency,
                "priority": priority,
                "type": message_type,
                "message": message,
            },
            f"{message_type}Send",
        )
        self.client_to_simulator_queue.put(message)

    def log_sent_message(self, message: MessageEvent, tag=None):
        tag = "MessageSent" if tag is None else tag
        self.log(
            {
                "fromidx": message.fromidx,
                "toidx": message.toidx,
                "marker": message.marker,
                "latency": message.custom_latency,
                "prio": message.priority,
            },
            tag,
        )

    def propose_blocks_start_attack(
        self, validator: Validator, head_state: spec.BeaconState
    ):
        block_left = spec.BeaconBlock(
            slot=head_state.slot,
            proposer_index=validator.index,
            parent_root=self.head_root,
            state_root=spec.Bytes32(bytes(0 for _ in range(0, 32))),
        )
        block_right = block_left.copy()

        randao_reveal = spec.get_epoch_signature(
            head_state, block_left, validator.privkey
        )
        eth1_data = head_state.eth1_data
        candidate_attestations = list()
        proposer_slashings: List[
            spec.ProposerSlashing, spec.MAX_PROPOSER_SLASHINGS
        ] = list()
        attester_slashings: List[
            spec.AttesterSlashing, spec.MAX_ATTESTER_SLASHINGS
        ] = list()
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
            voluntary_exits=voluntary_exits,
        )
        body_right = body_left.copy()
        body_right.graffiti = spec.Bytes32(bytes(1 for _ in range(0, 32)))

        block_left.body = body_left
        block_right.body = body_right

        try:
            new_state_root_left = spec.compute_new_state_root(self.state, block_left)
            new_state_root_right = spec.compute_new_state_root(self.state, block_right)
        except AssertionError:
            validator_status = [
                str(validator)
                for validator in self.state.validators
                if validator.slashed
            ]
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            self.__debug(
                {
                    "text": text,
                    "block_left": str(block_left),
                    "block_right": str(block_right),
                    "validators": str(validator_status),
                },
                "FindNewStateRootError",
            )
            self.client_to_simulator_queue.put(
                SimulationEndEvent(time=self.current_time, priority=0, message=text)
            )
            return

        block_left.state_root = new_state_root_left
        block_right.state_root = new_state_root_right

        signed_block_left = spec.SignedBeaconBlock(
            message=block_left,
            signature=spec.get_block_signature(
                self.state, block_left, validator.privkey
            ),
        )
        signed_block_right = spec.SignedBeaconBlock(
            message=block_right,
            signature=spec.get_block_signature(
                self.state, block_right, validator.privkey
            ),
        )

        self.block_root_left = spec.hash_tree_root(block_left)
        self.block_root_right = spec.hash_tree_root(block_right)
        self.first_block_root_left = self.block_root_left
        self.first_block_root_right = self.block_root_right

        print(
            f"block_root_left=[{self.block_root_left}] block_root_right=[{self.block_root_right}]"
        )

        encoded_signed_block_left = signed_block_left.encode_bytes()
        encoded_signed_block_right = signed_block_right.encode_bytes()

        self.distribute_targeted_message(
            "SignedBeaconBlock",
            encoded_signed_block_left,
            encoded_signed_block_right,
            latency=0,
            latency_otherside=spec.SLOTS_PER_EPOCH - 1,
            priority=1,
        )
        print("--- left")
        print(self.beacon_clients_left)
        print("--- right")
        print(self.beacon_clients_right)
        print("--- neither")
        print(self.beacon_clients_neither)

    def attack_slot(self) -> spec.Slot:
        return self.current_slot - self.attack_started_slot

    def distribute_targeted_message(
        self,
        message_type,
        left_message,
        right_message,
        latency=0,
        latency_otherside=5,
        priority=20,
        target_slot=None,
    ):
        target_slot = self.current_slot if target_slot is None else target_slot
        decoder = {
            "Attestation": spec.Attestation,
            "SignedAggregateAndProof": spec.SignedAggregateAndProof,
            "SignedBeaconBlock": spec.SignedBeaconBlock,
            "BeaconBlock": spec.BeaconBlock,
            "BeaconState": spec.BeaconState,
        }
        self.log(
            decoder[message_type].decode_bytes(left_message), "TargetedMessageLeft"
        )
        self.log(
            decoder[message_type].decode_bytes(right_message), "TargetedMessageRight"
        )
        for beacon_client in self.all_clients_left:
            if left_message is not None:
                message = MessageEvent(
                    time=self.current_time,
                    priority=priority,
                    message=left_message,
                    message_type=message_type,
                    fromidx=self.counter,
                    toidx=beacon_client,
                    custom_latency=latency,
                )
                self.log_sent_message(message, f"{message_type}SentLeftToLeft")
                self.client_to_simulator_queue.put(message)
            if right_message is not None:
                message = MessageEvent(
                    time=self.current_time,
                    priority=priority,
                    message=right_message,
                    message_type=message_type,
                    fromidx=self.counter,
                    toidx=beacon_client,
                    custom_latency=latency_otherside,
                )
                self.log_sent_message(message, f"{message_type}SentLeftToRight")
                self.client_to_simulator_queue.put(message)
        for beacon_client in self.all_clients_right:
            if left_message is not None:
                message = MessageEvent(
                    time=self.current_time,
                    priority=priority,
                    message=left_message,
                    message_type=message_type,
                    fromidx=self.counter,
                    toidx=beacon_client,
                    custom_latency=latency_otherside,
                )
                self.log_sent_message(message, f"{message_type}SentRightToLeft")
                self.client_to_simulator_queue.put(message)
            if right_message is not None:
                message = MessageEvent(
                    time=self.current_time,
                    priority=priority,
                    message=right_message,
                    message_type=message_type,
                    fromidx=self.counter,
                    toidx=beacon_client,
                    custom_latency=latency,
                )
                self.log_sent_message(message, f"{message_type}SentRightToRight")
                self.client_to_simulator_queue.put(message)
        for beacon_client in (
            set(self.beacon_clients_neither[target_slot])
            .difference(self.all_clients_left)
            .difference(self.all_clients_right)
        ):
            if left_message is not None:
                message = MessageEvent(
                    time=self.current_time,
                    priority=priority,
                    message=left_message,
                    message_type=message_type,
                    fromidx=self.counter,
                    toidx=beacon_client,
                    custom_latency=latency,
                )
                self.log_sent_message(message, f"{message_type}SentNeitherLeft")
                self.client_to_simulator_queue.put(message)
            if right_message is not None:
                message = MessageEvent(
                    time=self.current_time,
                    priority=priority,
                    message=right_message,
                    message_type=message_type,
                    fromidx=self.counter,
                    toidx=beacon_client,
                    custom_latency=latency,
                )
                self.log_sent_message(message, f"{message_type}SentNeitherRight")
                self.client_to_simulator_queue.put(message)
        if left_message is not None:
            message = MessageEvent(
                time=self.current_time,
                priority=priority,
                message=left_message,
                message_type=message_type,
                fromidx=self.counter,
                toidx=self.counter,
                custom_latency=0,
            )
            self.log_sent_message(message, f"{message_type}SentSelfLeft")
            self.client_to_simulator_queue.put(message)
        if right_message is not None:
            message = MessageEvent(
                time=self.current_time,
                priority=priority,
                message=right_message,
                message_type=message_type,
                fromidx=self.counter,
                toidx=self.counter,
                custom_latency=0,
            )
            self.log_sent_message(message, f"{message_type}SentSelfRight")
            self.client_to_simulator_queue.put(message)

    def compute_head(self):
        super().compute_head()
        self.block_root_left = self.block_cache.leafs_for_block(
            self.first_block_root_left
        )[0]
        self.block_root_right = self.block_cache.leafs_for_block(
            self.first_block_root_right
        )[0]
        if (
            self.block_root_left in self.store.block_states
            and self.block_root_right in self.store.block_states
        ):
            self.state_left = self.store.block_states[self.block_root_left]
            self.state_right = self.store.block_states[self.block_root_right]

            self.head_state_left = self.state_left.copy()
            if self.head_state_left.slot < self.current_slot:
                spec.process_slots(self.head_state_left, self.current_slot)

            self.head_state_right = self.state_right.copy()
            if self.head_state_right.slot < self.current_slot:
                spec.process_slots(self.head_state_right, self.current_slot)
            self.log(
                {
                    "state_left_slot": self.state_left.slot,
                    "state_right_slot": self.state_right.slot,
                    "head_state_left_slot": self.head_state_left.slot,
                    "head_state_right_slot": self.head_state_right.slot,
                    "block_root_left": self.block_root_left,
                    "block_root_right": self.block_root_right,
                    "first_block_left": self.first_block_root_left,
                    "first_block_right": self.first_block_root_right,
                },
                "SplittedStates",
            )

    def update_committee(self, epoch: spec.Epoch, genesis=False):
        super().update_committee(epoch, genesis)
        if (
            self.block_root_left in self.store.block_states
            and self.block_root_right in self.store.block_states
        ):
            committee_count_per_slot_left, new_committee_left = self.get_committee(
                epoch, genesis=genesis, head_state=self.head_state_left
            )
            committee_count_per_slot_right, new_committee_right = self.get_committee(
                epoch, genesis=genesis, head_state=self.head_state_right
            )
            self.committee_count_left[epoch] = committee_count_per_slot_left
            self.committee_count_right[epoch] = committee_count_per_slot_right
            for slot in new_committee_left.keys():
                self.committee_left[slot] = new_committee_left[slot]
                self.committee_right[slot] = new_committee_right[slot]
                self.log(
                    {
                        "slot": slot,
                        "left": str(new_committee_left[slot]),
                        "right": str(new_committee_right[slot]),
                    },
                    "BalancingCommittee",
                )
        else:
            self.log("Fallback to default committee", "BalancingCommittee")
            for slot in self.committee.keys():
                self.committee_left[slot] = self.committee[slot]
                self.committee_right[slot] = self.committee[slot]
            for epoch in self.committee_count.keys():
                self.committee_count_left[epoch] = self.committee_count[epoch]
                self.committee_count_right[epoch] = self.committee_count[epoch]
