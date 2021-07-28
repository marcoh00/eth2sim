import argparse
import multiprocessing
import os
import sys
from datetime import datetime
from importlib import reload
import pathlib
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

def simulation0(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(4)\
            .set_debug(True)\
            .validators(64)\
            .build()\
        .build()\
        .build()

def simulation1(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(11)\
            .set_debug(True)\
            .validators(21)\
            .build()\
        .build()\
        .beacon_client(1)\
            .set_debug(True)\
            .validators(25)\
            .build()\
        .build()\
        .build()

def simulation2(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(128)\
            .set_debug(True)\
            .validators(2)\
            .build()\
        .build()\
        .build()

def simulation3(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(4)\
            .set_debug(True)\
            .validators(2048)\
            .build()\
        .build()\
        .build()

def simulation4(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(11)\
            .set_debug(True)\
            .validators(682)\
            .build()\
        .build()\
        .beacon_client(1)\
            .set_debug(True)\
            .validators(690)\
            .build()\
        .build()\
        .build()

def simulation5(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(128)\
            .set_debug(True)\
            .validators(64)\
            .build()\
        .build()\
        .build()

def simulation6(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .beacon_client(4)\
            .set_debug(True)\
            .validators(4096)\
            .build()\
        .build()\
        .build()

def simulation7(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .beacon_client(8)\
            .set_debug(True)\
            .validators(2048)\
            .build()\
        .build()\
        .build()

def simulation8(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .beacon_client(3)\
            .set_debug(True)\
            .validators(33963)\
            .build()\
        .build()\
        .beacon_client(1)\
            .set_debug(True)\
            .validators(33962)\
            .build()\
        .build()\
        .build()

def simulation9(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .beacon_client(7)\
            .set_debug(True)\
            .validators(16981)\
            .build()\
        .build()\
        .beacon_client(1)\
            .set_debug(True)\
            .validators(16984)\
            .build()\
        .build()\
        .build()

def simulation10(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency * 4)\
        .beacon_client(128)\
            .set_debug(True)\
            .validators(2)\
            .build()\
        .build()\
        .build()

def simulation11(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .set_custom_latency_map(None, modifier=lambda latency: latency * 8)\
        .beacon_client(128)\
            .set_debug(True)\
            .validators(2)\
            .build()\
        .build()\
        .build()

def simulation12(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(calc_simtime(slot=1, epoch=8))\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(1)\
            .set_debug(True)\
            .set_mode('BlockSlashing')\
            .validators(32)\
            .build()\
        .build()\
        .beacon_client(7)\
            .set_debug(True)\
            .validators(32)\
            .build()\
        .build()\
    .build()

def simulation13(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(calc_simtime(slot=1, epoch=8))\
        .set_custom_latency_map(None, modifier=lambda latency: latency // 2)\
        .beacon_client(1)\
            .set_debug(True)\
            .set_mode('AttesterSlashingTooLow')\
            .validators(32)\
            .build()\
        .build()\
        .beacon_client(1)\
            .set_debug(True)\
            .set_mode('AttesterSlashingWithinSpan')\
            .validators(32)\
            .build()\
        .beacon_client(6)\
            .set_debug(True)\
            .validators(32)\
            .build()\
        .build()\
    .build()

def simulation14(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(1844674407370955161)\
        .beacon_client(8)\
            .set_debug(True)\
            .validators(32)\
            .build()\
        .build()\
        .build()

def simulation15(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(calc_simtime(slot=1, epoch=13))\
        .beacon_client(1)\
            .set_debug(True)\
            .set_mode('TimeAttacked')\
            .set_attackinfo({
                #'attack_start_slot': spec.Slot(32),
                #'attack_end_slot': spec.Slot(48),
                #'timedelta': spec.Slot(40)
                'attack_start_slot': spec.Slot(16),
                'attack_end_slot': spec.Slot(32),
                'timedelta': spec.Slot(40)
            })\
            .validators(32)\
            .build()\
        .build()\
        .beacon_client(7)\
            .set_debug(True)\
            .validators(32)\
            .build()\
        .build()\
        .build()

def simulation16(config, blockhash):
    # fmt: off
    return SimulationBuilder('../../configs', config, blockhash)\
        .set_end_time(calc_simtime(slot=1, epoch=4))\
        .set_custom_latency_map(((0, 0),))\
        .beacon_client(1)\
            .set_debug(True)\
            .set_mode('BalancingAttacking')\
            .validators(64)\
            .build()\
        .build()\
        .beacon_client(16)\
            .set_debug(True)\
            .validators(1)\
            .build()\
        .build()\
        .build()

def main():
    # 0 minimal 
    simtype = int(sys.argv[1])
    blockhash = bytes.fromhex('0000000000000000000000000000000000000000000000000000000000000000')
    configmap = {
        0:  ('minimal', simulation0), # minimal 4/256
        1:  ('minimal', simulation1), # minimal, 12/256
        2:  ('minimal', simulation2), # minimal, 128/256
        3:  ('minimal', simulation3), # minimal, 4/8192
        4:  ('minimal', simulation4), # minimal, 12/8192
        5:  ('minimal', simulation5), # minimal, 128/8192
        6:  ('mainnet', simulation6), # mainnet, 4/16384
        7:  ('mainnet', simulation7), # mainnet, 8/16384
        8:  ('mainnet', simulation8), # mainnet, 4/13581
        9:  ('mainnet', simulation9), # mainnet, 8/13581
        10: ('minimal', simulation10), # latency, 128/256
        11: ('mainnet-minimized', simulation11), # latency, 128/256
        12: ('minimal', simulation12), # proposer slashing
        13: ('minimal', simulation13), # attester slashings
        14: ('mainnet-minimized', simulation14), # latency, 128/256
        15: ('minimal', simulation15), # time attacked
        16: ('minimal', simulation16) # balancing attacking
    }
    mocked = simtype > 2
    mocked_filename = "mocked_" if mocked else ""

    config_util.prepare_config('../../configs', configmap[simtype][0])
    # noinspection PyTypeChecker
    reload(spec)
    spec.bls.bls_active = False

    print('Ethereum 2.0 Beacon Chain Simulator')
    print(f'PID: {os.getpid()}')
    print(f'Beacon Chain Configuration: {spec.CONFIG_NAME}')
    print(f'Eth1BlockHash for Genesis Block: {blockhash}')
    print(f'Bundled Simulation: {simtype}')
    print(f'Mocked: {mocked}')

    simulator = configmap[simtype][1](configmap[simtype][0], blockhash)

    filename = f"{mocked_filename}{simtype}"
    state_file = f"state_{filename}.ssz"
    if pathlib.Path(state_file).is_file():
        simulator.read_genesis(filename)
    else:
        simulator.generate_genesis(filename=simtype, mocked=mocked)
    
    simulator.initialize_clients()
    simulator.start_simulation()


if __name__ == '__main__':
    try:
        # multiprocessing.set_start_method("spawn")
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        end = datetime.now()
        sys.exit(0)
