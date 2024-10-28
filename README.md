[![REUSE status](https://api.reuse.software/badge/github.com/SAP/packerl)](https://api.reuse.software/info/github.com/SAP/packerl)

# PackeRL, M-Slim and FieldLines

*PackeRL* is an RL training and evaluation framework for routing optimization in computer networks. 
It leverages the [ns-3 network simulator](https://www.nsnam.org) for packet-level simulation dynamics. 
PackeRL is designed to be modular and flexible, allowing for easy integration of new algorithms and network models.

PackeRL uses its `scenarios` module (called _**Synnet**_ in the [PackeRL paper](https://openreview.net/pdf?id=H95g8UpYKY)) to generate network scenarios consisting of:

- A network topology (graph and configuration details like link capacities and delays),
- A list of traffic demands consisting of demand arrival time, volume and protocol (e.g. TCP or UDP), and
- [optional] A list of link failure events, each consisting of a failure time and the specific link that fails.

Besides PackeRL, this repository comes with our policy designs *M-Slim* and *FieldLines* which can be found in `rl/`.
For further information on PackeRL, and the policy designs check out the [PackeRL paper](https://openreview.net/pdf?id=H95g8UpYKY).

## Getting started

### What's in this repo?

This repo contains:
- a reference to the [ns-3 network simulator](https://www.nsnam.org) in the folder `ns3/`,
- ns3 PackeRL extension in `ns3-packerl/`, 
- ns3-ai external module in the folder `ns3-ai/`,
- the ML approach (including RL modules and algorithms written in [PyTorch](https://pytorch.org)) in the folders `rl/`, `reward/`, and `features/`,
- heuristic and ML baselines in `baselines/`
- scenario generation logic in `scenarios/` (called _**Synnet**_ in the [PackeRL paper](https://openreview.net/pdf?id=H95g8UpYKY))
- utilities in `utils/`.
- the main entry point for running experiments in `main.py`,
- configuration files in `config/`,
- the environment setup in `environment.yaml`,
- the installation script in `install.sh`.

### Downloading the resources

```
git clone <repo> --recurse-submodules
```

**or**, if you cloned the repository without its submodules, go to the root of the repositry and initialize the submodules:

```
cd packerl
git submodule update --init --recursive
```

### Installation on Ubuntu

**Requirements**

- `bash`
- `g++ >= 8` or `clang++ >= 10`
- `cmake >= 3.10`
- `make` or `ninja`
- `git`
- `conda`

**Installation commands (from repository root folder)**
```
conda env create -f environment.yaml
conda activate packerl
bash ./install.sh
```

Use the `-h` flag to display the available options for `install.sh`.

In order to enable the MAGNNETO baselines, use the `-m` flag.


### Quick Run

1. Run `python main.py config/examples/sanity_check.yaml -o -t -e <experiment>` to test the installation:
   - `sanity_check_single` tests the environment on a single experiment.
   - `sanity_check` starts multiple runs in parallel.
   - `sanity_check_wandb` checks whether visualization and logging to weights and biases works.

### CLI Parameter Explanations

- `-o` ensures you can continue editing the codebase while running experiments, as it create a temporaary copy of it.
- `-t` prepends a timestamp to the experiment name and corresponding output folder, so that you can run the same experiment multiple times without overriding or having to delete files. WARNING: PackeRL does not overwrite existing experiment folders, so you will have to delete them manually if you want to run the same experiment again if running without `-t`!
- `-e` specifies the name of the experiment you want to run from the given config file. You can specify multiple experiments by separating them with a space.
- `cw2` supports using the slurm workload manager to distribute parallel runs. To run experiments with slurm, you will have to add corresponding configuration to the used experiment config file and supply the `-s` argument in your CLI command. In this case you can omit the `-e` parameter if you want to run all experiments of a file.

### Updating the repo

1. Run `git pull --recurse-submodules` to get the latest updates for the repo and its submodules.
2. Run `bash install.sh -r` to (re-)install everything, re-building modified components.

### Current Limitations

- So far, our framework is limited to:
  - networks using `Ns3::PointToPointConnection` links,
  - single-path routing,
  - deterministic unicast traffic,
  - TCP and UDP traffic.
  - Up to 65533 demands per source node.
- Training and simulation are mainly CPU-bound tasks, so strong CPUs with many cores are recommended especially if you are starting many experiments at once.
- It is recommended to use at least 4 CPU cores per run.
- GPU training is implemented but slow. We are working on improving this.

### Development/Maintenance Notice

Please note that this is a research artifact and will thus not receive customer-facing maintenance or support.

### Citation

This repository is part of the PackeRL project ([webpage](https://alrhub.github.io/packerl-website/), [paper](https://openreview.net/pdf?id=H95g8UpYKY)) published in TMLR 2024. If you use this package in your research, please
cite the PackeRL paper:

```
@article{
boltres2024learning,
title={Learning Sub-Second Routing Optimization in Computer Networks requires Packet-Level Dynamics},
author={Andreas Boltres and Niklas Freymuth and Patrick Jahnke and Holger Karl and Gerhard Neumann},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=H95g8UpYKY},
note={}
}
```

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/packerl/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/packerl/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2024 SAP SE or an SAP affiliate company and packerl contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/packerl).