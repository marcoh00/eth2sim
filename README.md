# How to install & use

- Clone Ethereum 2.0 repository at tag v1.0.1:

```bash
git clone https://github.com/ethereum/eth2.0-specs.git -b v1.0.1
```

- Create a virtualenv or use pipenv to be able to install packages without polluting Python's global packages. E.g.:

```bash
cd eth2.0-specs
pipenv shell
```

- Install dependencies:

```bash
pipenv install -e .
```

- If a dependency resolution error (`milgaro_bls_binding`) is shown, open setup.py and change the line containing the dependency to a recent version (https://pypi.org/project/milagro-bls-binding/):

```python
install_requires=[
    # ...
    "milagro_bls_binding==1.6.3",
    # ...
]
```

- If neccessary, try to install the eth2.0-specs package again (see above)
- Install graphviz package for graph creation:

```bash
pipenv install graphviz
```

- Clone the simulator project into `build/lib` subdirectory:

```bash
mkdir build
git clone https://github.com/marcoh00/eth2sim.git build/lib
```

- Extract the Ethereum 2.0 specification python code into the `build/lib` directory

```bash
python setup.py pyspec
python setup.py build_py
```

- A new directory called `eth2spec` should show up inside `build/lib`.

- If needed: Copy directories with configuration files like `mainnet-minimized` to `configs`.

- Run simulator code, for example a minimal simulation:

```bash
cd build/lib
python run_official_sim.py 0
```

# Simulations inside "Simulating an Ethereum 2.0 Beacon Chain Network" paper

Simulations conducted for the paper can be run using the `run_official_sim.py` script, too.
They have the following numbers:

- 10 (latency x2)
- 11 (latency x8)
- 12 (block slashing)
- 13 (attester slashing)
- 15 (time attack)
- 16 (balancing attack)
- 17 (based on measurements)
