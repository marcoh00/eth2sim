import sys
import matplotlib.pyplot as plt
import json
import multiprocessing


def graph_blocks(title, ylabel, *kargs):
    for i in range(0, int(len(kargs) / 3)):
        plt.plot(kargs[i * 3], kargs[(i * 3) + 1], label=kargs[(i * 3) + 2])
    plt.xlim(0, 3600)
    plt.xlabel('Zeit (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def graph_memory(xlist, rss, vsz):
    plt.plot(xlist, rss, label='rss')
    plt.plot(xlist, vsz, label='vsz')
    #ax.set_ylim(0, 100)
    plt.xlim(0, 3600)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graph_cpu(xlist, ylist, cpulist=None, title='CPU-Auslastung'):
    if cpulist:
        fig, ax = plt.subplots(2, 1)
    else:
        fix, axs = plt.subplots()
        ax = (axs,)
    ax[0].plot(xlist, ylist, label='Insgesamt')
    if cpulist:
        ctr = 1
        for cpu in cpulist:
            ax[1].plot(xlist, cpu, label=f'CPU {ctr}')
            ctr += 1
        ax[1].legend()
        ax[1].set_ylim(0, 100)
        ax[1].set_xlim(0, 3600)
        ax[1].set_xlabel('Zeit (s)')
        ax[1].set_ylabel('CPU (%)')
    ax[0].set_ylim(0, 100)
    ax[0].set_xlim(0, 3600)
    ax[0].set_xlabel('Zeit (s)')
    ax[0].set_ylabel('CPU (%)')
    ax[0].legend()
    ax[0].set_title(title)
    plt.tight_layout()
    plt.show()


def normalize_tslist(tslist):
    start = None
    for value in tslist:
        if start is None:
            start = value
            yield 0
        else:
            yield value - start


def graph_multiple(data1, field, title, ylabel, filterfunc=None):
    if filterfunc is None:
        filterfunc = lambda x: False
    all = []

    ts_list = []
    blocks_list = []
    for measurement in data1:
        actual_measurement = int(field(measurement))
        # Filter obviously invalid measurement results
        if actual_measurement == 0 or filterfunc(actual_measurement):
            print(f"Ignore {actual_measurement}")
            continue
        ts_list.append(measurement['timestamp'])
        blocks_list.append(actual_measurement)
    all.append(list(normalize_tslist(ts_list)))
    all.append(blocks_list)
    all.append(sys.argv[1])

    if len(sys.argv) > 3:
        for i in range(3, len(sys.argv)):
            with open(sys.argv[i], 'r') as fp:
                data = json.load(fp)
            ts_list_add = []
            blocks_list_add = []
            for measurement in data:
                actual_measurement = int(field(measurement))
                # Filter obviously invalid measurement results
                if actual_measurement == 0 or filterfunc(actual_measurement):
                    print(f"Ignore {actual_measurement}")
                    continue
                ts_list_add.append(measurement['timestamp'])
                blocks_list_add.append(actual_measurement)
            all.append(list(normalize_tslist(ts_list_add)))
            all.append(blocks_list_add)
            all.append(sys.argv[i])

    graph_blocks(title, ylabel, *all)


def main():
    infile = sys.argv[1]
    mode = sys.argv[2]

    with open(infile, 'r') as fp:
        data = json.load(fp)

    if 'cpu' in mode:
        ts_list = []
        overall_list = []
        cpus_list = [[] for _ in range(0, multiprocessing.cpu_count())]
        for measurement in data:
            ts_list.append(measurement['timestamp'])
            overall_list.append(measurement['cpu']['overall'])
            cores = []
            for core, corevalue in measurement['cpu']['per_core'].items():
                cpus_list[int(core)].append(corevalue)
                cores.append(int(core))
            if len(cores) < multiprocessing.cpu_count():
                for core in range(0, multiprocessing.cpu_count()):
                    if core not in cores:
                        print(f'forgot {core}')
                        cpus_list[core].append(measurement['cpu']['overall'])

        ts_list = list(normalize_tslist(ts_list))
        if 'full' in mode:
            graph_cpu(ts_list, overall_list, cpus_list)
        else:
            graph_cpu(ts_list, overall_list)
        print('data read')

    if 'memory' in mode:
        ts_list = []
        rss_list = []
        vsz_list = []
        for measurement in data:
            ts_list.append(measurement['timestamp'])
            rss_list.append(measurement['mem']['rss'])
            vsz_list.append(measurement['mem']['vsz'])
        ts_list = list(normalize_tslist(ts_list))
        graph_memory(ts_list, rss_list, vsz_list)

    if 'multiproc' in mode:
        graph_multiple(data, lambda m: m['cpu']['overall'], 'CPU-Auslastung', 'CPU (%)', lambda pct: pct > 100)

    if 'multimem' in mode:
        graph_multiple(data, lambda m: m['mem']['rss'], 'Arbeitsspeicher-Auslastung', 'Speicher (MB)')

    if 'blocks' in mode:
        graph_multiple(data, lambda m: m['blocks'], 'Anzahl simulierter Blöcke', 'Blöcke (#)')


main()
