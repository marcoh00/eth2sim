#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import re
import datetime


REGEX_BCSIM = re.compile(r'!!! \[(\d+)] Miner \d+ mined a new Block')
REGEX_SIMBLOCK = re.compile(r'\[(\d+)] simblock.block.ProofOfWorkBlock@[a-f0-9]*:\d*')
REGEX_VIBES = re.compile(r'(\d+:\d+:\d+).\d* \[VSystem-akka.actor.default-dispatcher-\d*] DEBUG com.vibes.actors.NodeActor - BLOCK MINED AT LEVEL')
REGEX_ETH2SIM = re.compile(r'\[(\d+)] Block Message \d+ by Beacon Client \d+')

MODE_TO_REGEX = {
    'bcsim': REGEX_BCSIM,
    'simblock': REGEX_SIMBLOCK,
    'vibes': REGEX_VIBES,
    'eth2sim': REGEX_ETH2SIM
}

NORMALIZE_BCSIM = lambda d: int(d)
NORMALIZE_SIMBLOCK = lambda d: int(int(d) / 1000)
NORMALIZE_VIBES = lambda d: int(datetime.datetime.strptime(d, '%H:%M:%S').timestamp())
NORMALIZE_ETH2SIM = NORMALIZE_BCSIM

MODE_TO_NORMALIZE = {
    'bcsim': NORMALIZE_BCSIM,
    'simblock': NORMALIZE_SIMBLOCK,
    'vibes': NORMALIZE_VIBES,
    'eth2sim': NORMALIZE_ETH2SIM
}


def main():
    # usage: mode, infile, outfile
    mode = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]

    regex, normalize = MODE_TO_REGEX[mode], MODE_TO_NORMALIZE[mode]

    with open(infile, 'r') as fp:
        results = []
        block = 1
        first_timestamp = None
        last_timestamp = None
        for line in fp.readlines():
            #print(line)
            regexresult = regex.match(line)
            if regexresult:
                normalized = normalize(regexresult.group(1))
                if first_timestamp is None:
                    first_timestamp = normalized
                if last_timestamp is not None and last_timestamp == normalized:
                    print(f'[!] timestamp {normalized} ({normalized - first_timestamp}/{block}) is a duplicate')
                    results[-1]['blocks'] = block
                else:
                    results.append({'timestamp': normalized - first_timestamp, 'blocks': block})
                last_timestamp = normalized
                block += 1
    with open(outfile, 'w') as fp:
        json.dump(results, fp)

main()
