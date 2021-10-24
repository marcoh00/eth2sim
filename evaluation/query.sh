#!/bin/sh

set -e -x

export TIMESTAMP=1629244800
export DURATIONS=1641600
export DURATIOND=19

python prometheus_query.py -m aggregated_attestation_gossip_slot_start_delay_time_bucket -n 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 -b "${DURATIOND}d" -q "-8" -t $TIMESTAMP -o attestation_agg_paper.json quantile
python prometheus_query.py -m attestation_gossip_slot_start_delay_time_bucket -n 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 -b "${DURATIOND}d" -t $TIMESTAMP -o attestation_paper.json quantile
python prometheus_query.py -m beacon_block_imported_slot_start_delay_time_bucket -n 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 -b "${DURATIOND}d" -t $TIMESTAMP -o block_paper.json quantile
python prometheus_query.py -q "(libp2p_peers_per_client / ignoring (Client) group_left sum without (Client) (libp2p_peers_per_client)) * 100" -b $DURATIONS -t $TIMESTAMP -o client_paper.json range
python prometheus_query.py -q "increase(beacon_fork_choice_reorg_total[1d])" -b $DURATIONS -t $TIMESTAMP -s 768 -o fork_paper.json range
python prometheus_query.py -q "increase(beacon_fork_choice_reorg_total[${DURATIONS}s])" -t $TIMESTAMP -o total_forks_paper.json custom
python prometheus_query.py -q "slotclock_present_epoch - beacon_head_state_finalized_epoch - 2" -b $DURATIONS -t $TIMESTAMP -s 192 -o finalitydelay_paper.json range
python prometheus_query.py -q "increase(gossipsub_aggregated_attestations_rx_total[${DURATIONS}s]) / ${DURATIONS}" -t $TIMESTAMP -o rx_attestations_paper.json custom
#                                                                                      No. of SLOTS/$DURATIONS
python prometheus_query.py -q "beacon_head_state_active_validators_total" -t $TIMESTAMP -o total_validators_paper.json -b $DURATIONS range
