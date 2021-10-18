#!/bin/sh

set -x

ulimit -n 8192

mkdir old
mv *.json *.prof *.gv *.pdf old

# ---------------------------------------------------------------------------------------------------

#python3 -u run_official_sim.py 0 > 0.out 2> 0.err &
#sleep 10
#python pidstat_parse.py python3 14400 1 pidstat0.json 1
#killall -9 python3
#mkdir -p 0
#mv *.json *.gv *.pdf *.out *.err *.prof 0

# ----------------------------------------------------------------------------------------------------

#sleep 10
#python3 -u run_official_sim.py 1 > 1.out 2> 1.err &
#sleep 10
#python pidstat_parse.py python3 14400 1 pidstat1.json 1
#killall -9 python3
#mkdir -p 1
#mv *.json *.gv *.pdf *.out *.err *.prof 1



# sleep 10
# python3 -u run_official_sim.py 2 > 2.out 2> 2.err &
# sleep 10
# python pidstat_parse.py python3 14400 1 pidstat2.json 1
# killall -9 python3
# mkdir -p 2
# mv *.json *.gv *.pdf *.out *.err *.prof 2


# sleep 10
# python3 -u run_official_sim.py 3 > 3.out 2> 3.err &
# sleep 10
# python pidstat_parse.py python3 14400 1 pidstat3.json 1
# killall -9 python3
# mkdir -p 3
# mv *.json *.gv *.pdf *.out *.err *.prof 3



# sleep 10
# python3 -u run_official_sim.py 4 > 4.out 2> 4.err &
# sleep 10
# python pidstat_parse.py python3 14400 1 pidstat4.json 1
# killall -9 python3
# mkdir -p 4
# mv *.json *.gv *.pdf *.out *.err *.prof 4



# sleep 10
# python3 -u run_official_sim.py 5 > 5.out 2> 5.err &
# sleep 10
# python pidstat_parse.py python3 14400 1 pidstat5.json 1
# killall -9 python3
# mkdir -p 5
# mv *.json *.gv *.pdf *.out *.err *.prof 5


# ---------------------------------------------------------------------------------------------------

# sleep 10
# python3 -u run_official_sim.py 10 > 10.out 2> 10.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat10.json 1
# killall -9 python3
# mkdir -p 10
# mv *.json *.gv *.pdf *.out *.err *.prof 10

# sleep 10
# python3 -u run_official_sim.py 11 > 11.out 2> 11.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat11.json 1
# killall -9 python3
# mkdir -p 11
# mv *.json *.gv *.pdf *.out *.err *.prof 11


# sleep 10
# python3 -u run_official_sim.py 6 > 6.out 2> 6.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat6.json 1
# killall -9 python3
# mkdir -p 6
# mv *.json *.gv *.pdf *.out *.err *.prof 6

# sleep 10
# python3 -u run_official_sim.py 7 > 7.out 2> 7.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat7.json 1
# killall -9 python3
# mkdir -p 7
# mv *.json *.gv *.pdf *.out *.err *.prof 7

# sleep 10
# python3 -u run_official_sim.py 8 > 8.out 2> 8.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat8.json 1
# killall -9 python3
# mkdir -p 8
# mv *.json *.gv *.pdf *.out *.err *.prof 8

# sleep 10
# python3 -u run_official_sim.py 9 > 9.out 2> 9.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat9.json 1
# killall -9 python3
# mkdir -p 9
# mv *.json *.gv *.pdf *.out *.err *.prof 9

# sleep 10
# python3 -u run_official_sim.py 14 > 14.out 2> 14.err &
# sleep 10
# python pidstat_parse.py python3 21600 1 pidstat14.json 1
# killall -9 python3
# mkdir -p 14
# mv *.json *.gv *.pdf *.out *.err *.prof 14

#sleep 10
#python3 -u run_official_sim.py 14 > 14.2.out 2> 14.2.err &
#sleep 10
#python pidstat_parse.py python3 86400 1 pidstat142.json 1
#killall -9 python3
#mkdir -p 14.2
#mv *.json *.gv *.pdf *.out *.err *.prof 14.2

sleep 10
python3 -u run_official_sim.py 17 > 17.out 2> 17.err &
sleep 10
python pidstat_parse.py python3 43200 1 pidstat17.json 1
killall -9 python3
mkdir -p 17
mv *.json *.gv *.pdf *.out *.err *.prof 17

sleep 10
python3 -u run_official_sim.py 10 > 10.out 2> 10.err &
sleep 10
python pidstat_parse.py python3 43200 1 pidstat10.json 1
killall -9 python3
mkdir -p 10
mv *.json *.gv *.pdf *.out *.err *.prof 10

sleep 10
python3 -u run_official_sim.py 11 > 11.out 2> 11.err &
sleep 10
python pidstat_parse.py python3 43200 1 pidstat11.json 1
killall -9 python3
mkdir -p 11
mv *.json *.gv *.pdf *.out *.err *.prof 11
