import argparse
import datetime
import json
import pathlib
import sys
from urllib.parse import quote

import requests


QUANTILE_TEMPLATE = "histogram_quantile({}, sum(rate({}[{}])) by (le)) {}"

def calc_start(timestamp, substract):
    timestamp = datetime.datetime.fromtimestamp(timestamp)
    substract = int(substract)
    return int((timestamp - datetime.timedelta(seconds=substract)).timestamp())

def filename_custom_query(time, verb):
    counter = 0
    while True:
        filename = f"{time}_{verb}_{counter}.json"
        if not pathlib.Path(filename).is_file():
            return filename
        else:
            counter += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", choices=("custom", "quantile", "range"))
    parser.add_argument("--query", "-q", type=str, required=False, default="")
    parser.add_argument("--metric", "-m", type=str, required=False)
    parser.add_argument("--quantile", "-n", type=float, nargs="+", required=False)
    parser.add_argument("--bucket", "-b", type=str, required=False)
    parser.add_argument("--time", "-t", required=False, type=int, default=int(datetime.datetime.now().timestamp()))
    parser.add_argument("--url", "-u", required=False, default="http://ubuntu:9090")
    parser.add_argument("--step", "-s", required=False, type=int, default=3600)
    parser.add_argument("--output", "-o", required=False, type=str, default="")
    args = parser.parse_args()

    queries = []
    if args.verb == "custom" or args.verb == "range":
        queries.append(args.query)
    elif args.verb == "quantile":
        for quantile in args.quantile:
            queries.append(QUANTILE_TEMPLATE.format(quantile, args.metric, args.bucket, args.query))
    
    results = {}
    for query in queries:
        if args.verb == "range":
            url_template = f"{args.url}/api/v1/query_range?query={query}&start={calc_start(args.time, args.bucket)}&end={args.time}&step={args.step}"
            url = f"{args.url}/api/v1/query_range?query={quote(query)}&start={calc_start(args.time, args.bucket)}&end={args.time}&step={args.step}"
        else:
            url_template = f"{args.url}/api/v1/query?query={query}&time={args.time}"
            url = f"{args.url}/api/v1/query?query={quote(query)}&time={args.time}"
        print(f"[!] GET {url_template}")
        results[url_template] = requests.get(url).json()
    
    filename = args.output
    if not filename:
        filename = f"{args.time}_{args.verb}_{args.metric}_{args.bucket}.json" if args.metric and args.bucket else filename_custom_query(args.time, args.verb)
    with open(filename, 'w', encoding='utf-8') as fp:
        print(f"[!] Write query results to {filename}")
        json.dump({
            'params': sys.argv,
            'results': results
        }, fp, indent=2)

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
