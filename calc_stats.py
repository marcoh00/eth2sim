#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
        Simulierte Slots/h    & 0/1   & 1/1        \\
        Finality Delay/Epoche (min/max) & 0/1   & 1/1      \\
        Finality Delay/Epoche (\O) & x & x  \\
        Forks insgesamt    & 0/1   & 1/1     \\
        Forks/d    & 0/1   & 1/1        \\
        Stale Blocks insgesamt & 0/1   & 1/1      \\
        Orphan Blocks insgesamt & 0/1   & 1/1        \\
        Balance/Validator (min/max) & 0/1   & 1/1       \\
        Balance/Validator (\O) & 0/1   & 1/1     \\
        Attester Slashings insgesamt & 0/1   & 1/1      \\
        Proposer Slashings insgesamt & 0/1   & 1/1  \\ \bottomrule
"""

import json
import sys

def collect_by_epoch(data, valuegetter, drawdecision=None, aggregator=None, filter=None):
    epochs_value = dict()
    if drawdecision is None:
        drawdecision = lambda old, new: min((old, new))
    if aggregator is None:
        aggregator = lambda values: values
    if filter is None:
        filter = lambda epoch, value: True
    for stats in data:
        value = valuegetter(stats)
        if stats['current_epoch'] in epochs_value:
            old = epochs_value[stats['current_epoch']]
            new = value
            #print(f"[WARN] Epoch {stats['current_epoch']}: {old}, {new} = {drawdecision(old, new)}")
            value = drawdecision(old, new)
        if filter(stats['current_epoch'], value):
            epochs_value[stats['current_epoch']] = value
    return aggregator(epochs_value.values())
    

def main():
    # usage: infile runtime[h]
    infile = sys.argv[1]
    runtime_h = float(sys.argv[2])
    data = []
    with open(infile, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    slots_per_hour = calc_slots_h(data, runtime_h)
    mean_finality_delay = collect_by_epoch(data, lambda o: o['finality_delay'], lambda old, new: (old + new) / 2, aggregator=lambda values: (sum((value - 1 for value in values)) / len(values)) if len(values) > 0 else '-', filter=lambda epoch, value: epoch > 2)
    max_finality_delay = collect_by_epoch(data, lambda o: o['finality_delay'], lambda old, new: max(old, new), aggregator=lambda values: (max((value - 1 for value in values))) if len(values) > 0 else '-', filter=lambda epoch, value: epoch > 2)
    min_finality_delay = collect_by_epoch(data, lambda o: o['finality_delay'], lambda old, new: min(old, new), aggregator=lambda values: (min((value - 1 for value in values))) if len(values) > 0 else '-', filter=lambda epoch, value: epoch > 2)
    forks_total = len(data[-1]['leafs']) - 1
    forks_d = (24 / runtime_h) * forks_total
    stales = forks_total
    print("WARNING! STALES: CHECK FOR LONGER FORKS THAN 1 MANUALLY BY INSPECTING THE GRAPH!!!")
    orphans = len(data[-1]['orphans']) - 1
    balance_validator_min = min(data[-1]['balances'].values())
    balance_validator_max = max(data[-1]['balances'].values())
    balance_validator_mean = sum(data[-1]['balances'].values()) / len(data[-1]['balances'].values())
    attester_slashings = len(data[-1]['attester_slashings'])
    proposer_slashings = len(data[-1]['proposer_slashings'])

    result = {
        'slots_h': slots_per_hour,
        'mean_finality_delay': mean_finality_delay,
        'max_finality_delay': max_finality_delay,
        'min_finality_delay': min_finality_delay,
        'forks': forks_total,
        'forks_d': forks_d,
        'stales': stales,
        'orphans': orphans,
        'balance_validator_min': balance_validator_min,
        'balance_validator_max': balance_validator_max,
        'balance_validator_mean': balance_validator_mean,
        'attester_slashings': attester_slashings,
        'proposer_slashings': proposer_slashings
    }
    print(result)
    
def calc_slots_h(data, runtime_h):
    return data[-1]['current_slot'] / runtime_h


if __name__ == '__main__':
    main()
    sys.exit(0)
