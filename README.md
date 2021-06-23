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
git clone /PATH/TO/GIT/BUNDLE build/lib
```

- Extract the Ethereum 2.0 specification python code into the `build/lib` directory

```bash
python setup.py pyspec
python setup.py build_py
```

- A new directory called `eth2spec` should show up inside `build/lib`.

- If needed: Copy directories with configuration files like `mainnet-minimized` to `configs`.

- Run simulator code, for example the first minimal simulation (numbers correspond to the 'Nr.' column in table 5.8):

```bash
cd build/lib
python run_official_sim.py 0
```