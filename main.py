import argparse
import multiprocessing
from datetime import datetime
from importlib import reload
import random

from simulator import SimulationBuilder
from eth2spec.config import config_util
from eth2spec.phase0 import spec
from pathvalidation import valid_writable_path


def calc_simtime(slot, epoch=None, seconds=None):
    """
    Calculate timestamps for slots, epochs and timedeltas in seconds
    If only slot is specified, the start slot time is calculated.
    If both slot and epoch are specified, slot is relative to epoch.
    """
    time = spec.SECONDS_PER_SLOT * slot
    if epoch is not None:
        time += epoch * spec.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH
    if seconds is not None:
        time += seconds
    return int(time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configpath', type=str, required=False, default='../../configs')
    parser.add_argument('--configname', type=str, required=False, default='minimal')
    parser.add_argument('--cryptokeys', type=valid_writable_path, required=False, default='./cryptokeys')
    # parser.add_argument('--state', type=valid_writable_path, required=False, default='./state')
    parser.add_argument('--eth1blockhash', type=bytes.fromhex, required=False, default=random.randbytes(32).hex())
    args = parser.parse_args()
    config_util.prepare_config(args.configpath, args.configname)
    # noinspection PyTypeChecker
    reload(spec)
    spec.bls.bls_active = False

    print('Ethereum 2.0 Beacon Chain Simulator')
    print(f'Beacon Chain Configuration: {spec.CONFIG_NAME}')
    print(f'Eth1BlockHash for Genesis Block: {args.eth1blockhash.hex()}')
    print(f'Cryptographic Keys: {args.cryptokeys}')

    simulator = SimulationBuilder(args.configpath, args.configname, args.eth1blockhash)\
        .set_end_time(calc_simtime(0, 7, 0))\
        .add_graph_output(3, calc_simtime(1, 1, 2), False)\
        .add_statistics_output(3, calc_simtime(9, 0, 2))\
        .add_graph_output(5, calc_simtime(2, 2), False)\
        .add_statistics_output(5, calc_simtime(3, 2, 2))\
        .beacon_client(8)\
        .validators(1032)\
        .build()\
        .build()\
        .beacon_client(1)\
        .set_debug(True)\
        .set_profile(True)\
        .validators(1)\
        .build()\
        .build()\
        .beacon_client(1)\
        .validators(8127)\
        .build()\
        .build()\
        .build()
    simulator.read_genesis('mainnet')
    simulator.initialize_clients()
    simulator.start_simulation()


if __name__ == '__main__':
    start = datetime.now()
    try:
        multiprocessing.set_start_method("spawn")
        main()
        end = datetime.now()
    except KeyboardInterrupt:
        end = datetime.now()
    print(f"{end-start}")
