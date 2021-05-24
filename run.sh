#!/bin/sh

set -x

mkdir old
mv *.json *.prof *.gv *.pdf old

python3 run_official_sim.py 0 > 0.out 2> 0.err &
sleep 10
python /home/marco/Dokumente_git/masterarbeit/code/pidstat_parse.py python3 5400 1 pidstat0.json 1

mkdir -p 0
mv *.json *.prof *.gv *.pdf *.out *.err 0

sleep 10

python3 run_official_sim.py 1 > 1.out 2> 1.err &
sleep 10
python /home/marco/Dokumente_git/masterarbeit/code/pidstat_parse.py python3 18000 1 pidstat1.json 1

mkdir -p 1
mv *.json *.prof *.gv *.pdf *.out *.err 1
