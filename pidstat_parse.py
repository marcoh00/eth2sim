#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import datetime
import subprocess
import json
import multiprocessing

CPU_REGEX_PROCESS = re.compile(r'^\d+:\d+:\d+\s+(\d+)\s+(\d+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+|-)\s+([0-9a-zA-Z/\-\.]+)$', re.MULTILINE)
MEM_REGEX_PROCESS = re.compile(r'^\d+:\d+:\d+\s+(\d+)\s+(\d+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+)\s+(\d+)\s+([0-9\.]+)\s+([0-9a-zA-Z/\-\.]+)$', re.MULTILINE)

MEM_REGEX_PS = re.compile(r'^[^\s]+\s+([^\s]+)\s+[^\s]+\s+[^\s]+\s+([^\s]+)\s+([^\s]+)\s+[^\s]+\s+[^\s]+\s+[^\s]+\s+[^\s]+\s+(.*?)$', re.MULTILINE)

CPU_REGEX_MULTITHREAD = re.compile(r'^\d+:\d+:\d+\s+(\d+)\s+(\d+|-)\s+(\d+|-)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+|-)\s+([0-9a-zA-Z/\-\._|]+)$', re.MULTILINE)
MEM_REGEX_MULTITHREAD = re.compile(r'^\d+:\d+:\d+\s+(\d+)\s+(\d+|-)\s+(\d+|-)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+)\s+(\d+)\s+([0-9\.]+)\s+([0-9a-zA-Z/\-\._|]+)$', re.MULTILINE)


def acc_cpu_usage(measurements: list, mt=False):
    acc = {}
    for core in range(0, multiprocessing.cpu_count()):
        acc[core] = []
    for measurement in measurements:
        command = str(measurement[-1])
        cpu = int(measurement[-2])
        pct = float(measurement[-3])
        # main process cpu info is cumulative (e.g. 500%) - we calculate this by ourselves
        if mt and '|_' not in command:
            continue
        if pct > 100:
            print(f'WARNING: invalid value for pct ({pct})')
            pct = 100
        acc[cpu].append(pct)
    sum_overall = 0
    for key, value in acc.items():
        sum_core = 0
        for pct in value:
            sum_core += pct
        acc[key] = sum_core
        sum_overall += sum_core
    sum_overall /= multiprocessing.cpu_count()
    return {'overall': sum_overall, 'per_core': acc}


def acc_memory_usage(measurements: list, mt=False, ps=False):
    rss_sum = 0
    vsz_sum = 0

    if mt:
        rss_sum = int(measurements[0][-3])
        vsz_sum = int(measurements[0][-4])
    elif ps:
        for measurement in measurements:
            vsz = int(measurement[1])
            rss = int(measurement[2])
            rss_sum += rss
            vsz_sum += vsz
    else:
        for measurement in measurements:
            rss = int(measurement[-3])
            vsz = int(measurement[-4])
            rss_sum += rss
            vsz_sum += vsz
    # In MB
    rss_sum /= 1024
    vsz_sum /= 1024
    return {'rss':  rss_sum, 'vsz': vsz_sum}


def is_pid(piproc):
    try:
        int(piproc)
    except:
        return False
    return True


def pidstat_cmdline(piproc, pidstatruntime):
    if is_pid(piproc):
        return f'LC_ALL=C pidstat -r -u -t -p {piproc} {pidstatruntime} 1'
    return f'LC_ALL=C pidstat -r -u -G {piproc} {pidstatruntime} 1'


def get_regex(piproc, useps):
    retval = [CPU_REGEX_PROCESS, MEM_REGEX_PROCESS]
    if is_pid(piproc):
        retval = [CPU_REGEX_MULTITHREAD, MEM_REGEX_MULTITHREAD]
    if useps:
        retval[1] = MEM_REGEX_PS
    return retval


def main():
    # Usage: PID/PROCESS Runtime(s) pidstat_runtime(s) outfile useps
    piproc = sys.argv[1]
    runtime = int(sys.argv[2])
    pidstatruntime = sys.argv[3]
    outfile = sys.argv[4]
    useps = False
    if len(sys.argv) > 5:
        useps = True

    now = datetime.datetime.now()
    target = now + datetime.timedelta(seconds=runtime)
    runs = runtime / int(pidstatruntime)

    cmdline = pidstat_cmdline(piproc, pidstatruntime)
    regex_cpu, regex_mem = get_regex(piproc, useps)
    mt = is_pid(piproc)

    print(f'[-] We have {multiprocessing.cpu_count()} CPUs')
    print(f'[-] Process will stop at {target}')
    if useps:
        print(f'[-] Use ps for memory measurements')

    with open(outfile, 'w', encoding='utf-8') as fp:
        fp.write('[')

    with open(outfile, 'a', encoding='utf-8') as fp:
        progress = 0
        now = datetime.datetime.now()
        while now < target:
            pidstat = subprocess.run(cmdline, shell=True, stdout=subprocess.PIPE)
            cpu = acc_cpu_usage(regex_cpu.findall(pidstat.stdout.decode('utf-8')), mt=mt)
            if useps:
                ps = subprocess.run('ps aux', shell=True, stdout=subprocess.PIPE)
                psout = ps.stdout.decode('utf-8')
                relevant = [measurement for measurement in regex_mem.findall(psout) if measurement[3].startswith(piproc)]
                mem = acc_memory_usage(relevant, mt=mt, ps=True)
            else:
                mem = acc_memory_usage(regex_mem.findall(pidstat.stdout.decode('utf-8')), mt=mt)
            print(f'[{progress}/{(target - now).seconds}] Got measurement')
            fp.write(json.dumps({
                'timestamp': now.timestamp(),
                'cpu': cpu,
                'mem': mem
            }))
            fp.write(',')
            progress += 1
            now = datetime.datetime.now()
        fp.write(']')
    if mt:
        subprocess.run(f'kill -9 {piproc}', shell=True)
    else:
        subprocess.run(f'killall -9 {piproc}', shell=True)
    print('Killed')


main()
