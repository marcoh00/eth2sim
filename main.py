import argparse
import multiprocessing
from datetime import datetime
from importlib import reload
import random

from builder import SimulationBuilder
from eth2spec.config import config_util
from eth2spec.phase0 import spec
from pathvalidation import valid_writable_path


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

    simulator = SimulationBuilder(args.configpath, args.configname, args.eth1blockhash) \
        .beacon_client(4) \
        .validators(18) \
        .build() \
        .build() \
        .build()
    simulator.generate_genesis()
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
