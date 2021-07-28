from eth2spec.utils.ssz import ssz_typing
from queue import Queue, Empty
from typing import List, Optional, Any, Sequence

from remerkleable.basic import uint64
from remerkleable.bitfields import Bitlist
from remerkleable.byte_arrays import ByteVector, Bytes32


def popcnt(bitlist: Bitlist) -> uint64:
    bits = uint64(0)
    for bit in bitlist:
        if bit:
            bits += 1
    return bits


def indices_inside_committee(bitlist: Bitlist) -> List[uint64]:
    indexes = list()
    for i, bit in enumerate(bitlist):
        if bit:
            indexes.append(uint64(i))
    return indexes


def queue_element_or_none(queue: Queue) -> Optional[Any]:
    try:
        return queue.get(False)
    except Empty:
        return None


def process_mocked_deposit(spec, state, deposit, check_included=False) -> None:
    # --------------------------------
    # No check for valid merkle branch
    # --------------------------------

    # Deposits must be processed in order
    state.eth1_deposit_index += 1

    pubkey = deposit.data.pubkey
    amount = deposit.data.amount
    validator_pubkeys = [v.pubkey for v in state.validators] if check_included else []
    if not check_included or pubkey not in validator_pubkeys:
        # -------------------------
        # No signature verification
        # -------------------------

        # Add validator and balance entries
        state.validators.append(spec.get_validator_from_deposit(state, deposit))
        state.balances.append(amount)
    else:
        # Increase balance by deposit amount
        index = spec.ValidatorIndex(validator_pubkeys.index(pubkey))
        spec.increase_balance(state, index, amount)


def initialize_beacon_state_from_mocked_eth1(
    spec, eth1_block_hash: Bytes32, eth1_timestamp: uint64, deposits: Sequence
) -> Any:
    fork = spec.Fork(
        previous_version=spec.GENESIS_FORK_VERSION,
        current_version=spec.GENESIS_FORK_VERSION,
        epoch=spec.GENESIS_EPOCH,
    )
    state = spec.BeaconState(
        genesis_time=eth1_timestamp + spec.GENESIS_DELAY,
        fork=fork,
        eth1_data=spec.Eth1Data(
            block_hash=eth1_block_hash, deposit_count=len(deposits)
        ),
        latest_block_header=spec.BeaconBlockHeader(
            body_root=spec.hash_tree_root(spec.BeaconBlockBody())
        ),
        randao_mixes=[eth1_block_hash]
        * spec.EPOCHS_PER_HISTORICAL_VECTOR,  # Seed RANDAO with Eth1 entropy
    )

    # Process deposits
    leaves = list(map(lambda deposit: deposit.data, deposits))
    last_index = -1
    for index, deposit in enumerate(deposits):
        if index % 1000 == 0:
            print(f"[SIMULATOR][State] Deposit #{index}")
        # ------------------------------------------------
        # Calculate the deposit root only one time (below)
        # ------------------------------------------------
        process_mocked_deposit(spec, state, deposit, check_included=False)
        last_index = index
    deposit_data_list = ssz_typing.List[
        spec.DepositData, 2 ** spec.DEPOSIT_CONTRACT_TREE_DEPTH
    ](*leaves[: last_index + 1])
    state.eth1_data.deposit_root = spec.hash_tree_root(deposit_data_list)

    # Process activations
    for index, validator in enumerate(state.validators):
        if index % 1000 == 0:
            print(f"[SIMULATOR][State] Validator #{index}")
        balance = state.balances[index]
        validator.effective_balance = min(
            balance - balance % spec.EFFECTIVE_BALANCE_INCREMENT,
            spec.MAX_EFFECTIVE_BALANCE,
        )
        if validator.effective_balance == spec.MAX_EFFECTIVE_BALANCE:
            validator.activation_eligibility_epoch = spec.GENESIS_EPOCH
            validator.activation_epoch = spec.GENESIS_EPOCH

    # Set genesis validators root for domain separation and chain versioning
    state.genesis_validators_root = spec.hash_tree_root(state.validators)

    return state
