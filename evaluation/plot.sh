#!/bin/sh

set -e -x

python prometheus_plot.py -i client.json -t "Verwendete Clients verbundener Peers" -x "Zeit" -y "Anteil (%)" --ylim 100
python prometheus_plot.py -i fork.json -t "Aufgetretene Forks innerhalb der letzten 24h" -x "Zeit" -y "Forks" -l ""
python prometheus_plot.py -i finalitydelay.json -t "Verzögerung der Finalisierung" -x "Zeit" -y "Verzögerung (Epochen)" --splitgt 3 -l "Vor Verbindungsabbruch" "Nach Verbindungsabbruch"
python prometheus_plot.py -i total_validators.json -t "Aktive Validatoren in der Beacon Chain" -x "Zeit" -y "Validatoren"