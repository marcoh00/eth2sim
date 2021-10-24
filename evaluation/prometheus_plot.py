import argparse
import datetime
import itertools
import json
import statistics
import sys

import matplotlib.pyplot as plt

def chain_lists(list_of_lists):
    for inner_list in list_of_lists:
        for inner_element in inner_list:
            yield inner_element

def plot(labels, xvalues, yvalues, title="", xlabel="", ylabel="", ylim=None):
    assert len(labels) == len(xvalues)
    assert len(xvalues) == len(yvalues)
    xset = set(single_value for value_list in xvalues for single_value in value_list)
    legend = all(label is not None and not label == "" for label in labels)
    for i in range(len(labels)):
        plt.plot(xvalues[i], yvalues[i], label=labels[i])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend(loc='upper left')
    if ylim:
        plt.ylim(0, ylim)
    else:
        plt.ylim(0)
    print(len(xset))
    plt.xticks(range(0, len(xset), len(xset) // 10))
    #plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser("Plotter for results of range queries")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--labels", "-l", type=str, nargs='+', action='extend', required=False)
    parser.add_argument("--title", "-t", default="", required=False)
    parser.add_argument("--xlabel", "-x", default="Zeit", required=False)
    parser.add_argument("--ylabel", "-y", default="", required=False)
    parser.add_argument("--splitgt", "-s", type=float, default=-1, required=False)
    parser.add_argument("--ylim", "-m", type=float, default=-1, required=False)
    args = parser.parse_args()

    use_labels = args.labels is not None and len(args.labels) > 0

    with open(args.input, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    data = data['results']
    data = data[tuple(data.keys())[0]]['data']['result']

    labels = []
    x = []
    y = []
    for num, result in enumerate(data):
        if use_labels and len(args.labels) > num:
            label = args.labels[num]
        else:
            label = result['metric'][tuple(result['metric'].keys())[0]]
        labels.append(label)

        x_values = []
        y_values = []

        for value in result['values']:
            timestamp = value[0]
            timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%d.%m.%y %H:%M")
            measurement = value[1]
            if measurement == "NaN":
                continue
            measurement = float(measurement)

            # Split if neccessary
            if args.splitgt != -1 and measurement > args.splitgt:
                print(f"{measurement} > {args.splitgt}")
                if len(x_values) > 0:
                    print(f'{label} min=[{min(y_values)}] max=[{max(y_values)}] mean=[{statistics.mean(y_values)}] median=[{statistics.median(y_values)}]')
                    x.append(x_values)
                    y.append(y_values)
                    x_values = []
                    y_values = []
                    labels.append(label)
                continue
            x_values.append(timestamp)
            y_values.append(measurement)
        
        print(f'{label} min=[{min(y_values)}] max=[{max(y_values)}] mean=[{statistics.mean(y_values)}] median=[{statistics.median(y_values)}]')
        x.append(x_values)
        y.append(y_values)
    print(f"ALL LABELS min=[{min(y_values)}] max=[{max(chain_lists(y))}] mean=[{statistics.mean(chain_lists(y))}] median=[{statistics.median(chain_lists(y))}]")
    
    if args.labels is not None and len(args.labels) == len(labels):
        labels = args.labels
    plot(labels, x, y, args.title, args.xlabel, args.ylabel, ylim=args.ylim if args.ylim is not None and args.ylim > 0 else None)
    
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
