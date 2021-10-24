"""
"0x2f04d5c838cd577bd1fcef36e4eed6738b21db2720502fdf4777241d204921b4" [color="#00b7ff" shape=octagon style=filled]
"0x8c2cb7c5c0e6700dcea4d8a630533a853414a0d9b64de024bd767915d73ba5bb" -> "0x1da35b1913edb2fbecb5d86e250c520969aba58695b957c4d27a3742f08c00fc"
"""

import re
import sys

FIND_HEAD = re.compile(r'\s*"(0x[a-f0-9]{4,})" \[.*?shape=octagon')
FIND_EDGE = re.compile(r'\s*"(0x[a-f0-9]{4,})" -> "(0x[a-f0-9]{4,})"')

FILENAME = 'graph_1634701032_4_417.gv'

HEADBLOCK = [None,]
BLOCKS = {}

def find_leafs():
    leafs = []
    for block in BLOCKS.keys():
        isleaf = not any(obj["ancestor"] == block for obj in BLOCKS.values())
        if isleaf:
            leafs.append(block)
    return leafs

def canonical_chain():
    chain = [HEADBLOCK[0],]
    current_block = HEADBLOCK[0]
    while BLOCKS[current_block]["ancestor"] is not None:
        current_block = BLOCKS[current_block]["ancestor"]
        chain.append(current_block)
    return chain

def fork_lengths():
    canonical = canonical_chain()
    leafs = find_leafs()
    branches = []
    for leaf in leafs:
        if leaf in canonical:
            print(f'!!! {leaf} is in canonical')
            continue
        branch = []
        current = leaf
        while current not in canonical:
            branch.append(current)
            current = BLOCKS[current]["ancestor"]
        branches.append(branch)
    return branches

def main():
    contents = ''
    with open(FILENAME, 'r') as fp:
        contents = fp.read()
    HEADBLOCK[0] = FIND_HEAD.search(contents).group(1)
    BLOCKS[HEADBLOCK[0]] = {"ancestor": None, "data": None}
    for find in FIND_EDGE.findall(contents):
        block = find[0]
        ancestor = find[1]
        if block not in BLOCKS:
            BLOCKS[block] = {"ancestor": None, "data": None}
        if ancestor not in BLOCKS:
            BLOCKS[ancestor] = {"ancestor": None, "data": None}
        BLOCKS[block]["ancestor"] = ancestor
    print(BLOCKS)
    print(HEADBLOCK)