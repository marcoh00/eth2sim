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
    simulator = SimulationBuilder('../../configs', 'minimal', eth1blockhash)\
    .set_end_time(calc_simtime(slot=1, epoch=8))\
    .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
    .beacon_client(1)\
        .set_debug(True)\
        .set_mode('AttesterSlashingTooLow')\
        .validators(32)\
        .build()\
    .build()\
    .beacon_client(7)\
        .set_debug(True)\
        .validators(32)\
        .build()\
    .build()\
    .build()
    simulator.generate_genesis(filename='256_8_proposerslashing')
    simulator.initialize_clients()
    simulator.start_simulation()

    # simulator = SimulationBuilder(args.configpath, args.configname, args.eth1blockhash)\
    #     .set_end_time(calc_simtime(3, 2, 4))\
    #     .add_graph_output(3, calc_simtime(1, 1, 2), False)\
    #     .add_statistics_output(3, calc_simtime(9, 0, 2))\
    #     .add_graph_output(5, calc_simtime(2, 2), False)\
    #     .add_statistics_output(5, calc_simtime(3, 2, 2))\
    #     .beacon_client(8)\
    #     .validators(1032)\
    #     .build()\
    #     .build()\
    #     .beacon_client(1)\
    #     .set_debug(True)\
    #     .set_profile(True)\
    #     .validators(1)\
    #     .build()\
    #     .build()\
    #     .beacon_client(1)\
    #     .validators(8127)\
    #     .build()\
    #     .build()\
    #     .build()
    # simulator.read_genesis('mainnet')
    # simulator.initialize_clients()
    # simulator.start_simulation()

    # simulator = SimulationBuilder('../../configs', 'minimal', args.eth1blockhash)\
    #     .set_end_time(calc_simtime(2, 6, 2))\
    #     .add_graph_output(1, calc_simtime(1, 3, 1), show=True)\
    #     .add_graph_output(0, calc_simtime(2, 6, 1), show=True) \
    #     .add_statistics_output(1, calc_simtime(1, 3, 1)) \
    #     .add_statistics_output(0, calc_simtime(2, 6, 1)) \
    #     .beacon_client(8)\
    #     .set_mode('HONEST')\
    #     .validators(32)\
    #     .build()\
    #     .build() \
    #     .beacon_client(1) \
    #     .set_debug(True) \
    #     .validators(16) \
    #     .build() \
    #     .build() \
    #     .beacon_client(10) \
    #     .validators(1) \
    #     .build() \
    #     .build() \
    #     .build()
    # simulator.generate_genesis()
    # simulator.initialize_clients()
    # simulator.start_simulation()

    #simulator = SimulationBuilder(args.configpath, args.configname, args.eth1blockhash)\
    #.set_end_time(9999999999999)\
    #.set_custom_latency_map(None, modifier=lambda l: l // 2)\
    #.beacon_client(8)\
    #.set_debug(True)\
    #.validators(1024)\
    #.build()\
    #.build()\
    #.build()
    #simulator.generate_genesis(filename='8_8192_minimal')
    #simulator.initialize_clients()
    #simulator.start_simulation()
    
    #simulator = SimulationBuilder('../../configs', 'mainnet', args.eth1blockhash) \
    #    .set_end_time(9999999999999) \
    #    .set_custom_latency_map(None, modifier=lambda l: l // 2) \
    #    .beacon_client(128) \
    #    .set_debug(True) \
    #    .validators(128) \
    #    .build() \
    #    .build() \
    #    .build()
    #simulator.generate_genesis(filename='128_16384_mainnet')#

    # simulator = SimulationBuilder('../../configs', 'mainnet', eth1blockhash) \
    #     .set_end_time(9999999999999) \
    #     .set_custom_latency_map(None, modifier=lambda l: l // 2) \
    #     .beacon_client(128) \
    #     .set_debug(True) \
    #     .validators(1061) \
    #     .build()\
    #     .build()\
    #     .beacon_client(1) \
    #     .set_debug(True) \
    #     .validators(43) \
    #     .build() \
    #     .build() \
    #     .build()
    # simulator.generate_genesis(filename='129_135851_mainnet_cheaply_mocked', mocked=True)
    # #simulator.initialize_clients()
    # #simulator.start_simulation()


if __name__ == '__main__':
    try:
        # multiprocessing.set_start_method("spawn")
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        end = datetime.now()
        sys.exit(0)
