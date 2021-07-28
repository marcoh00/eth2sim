import argparse
import multiprocessing
import os
import sys
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
    print(f'PID: {os.getpid()}')
    print(f'Beacon Chain Configuration: {spec.CONFIG_NAME}')
    print(f'Eth1BlockHash for Genesis Block: {args.eth1blockhash.hex()}')
    print(f'Cryptographic Keys: {args.cryptokeys}')

    eth1blockhash = bytes.fromhex('0000000000000000000000000000000000000000000000000000000000000000')
    
    # fmt: off
    simulator = SimulationBuilder('../../configs', 'minimal', eth1blockhash)\
    .set_end_time(calc_simtime(slot=1, epoch=13))\
    .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
    .beacon_client(1)\
        .set_debug(True)\
        .set_mode('TimeAttacked')\
        .set_attackinfo({
            'attack_start_slot': spec.Slot(8),
            'attack_end_slot': spec.Slot(24),
            'timedelta': spec.Slot(48)
        })\
        .validators(1)\
        .build()\
    .build()\
    .beacon_client(1)\
        .set_debug(True)\
        .validators(254)\
        .build()\
    .build()\
    .beacon_client(1)\
        .validators(1)\
        .build()\
    .build()\
    .build()
    simulator.generate_genesis()
    simulator.initialize_clients()
    simulator.start_simulation()


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        end = datetime.now()
        sys.exit(0)
