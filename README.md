# Flamingo: Multi-Round Single-Server Secure Aggregation with Applications to Private Federated Learning

Flamingo is a system for secure aggregation built for private federated learning. 
This implementation accompanies our [paper](https://eprint.iacr.org/2023/486) by Yiping Ma, Jess Woods, Sebastian Angel, Antigoni Polychroniadou and Tal Rabin at IEEE S&P (Oakland) 2023. 

WARNING: This is an academic proof-of-concept prototype and is not production-ready.

## Overview
We integrate our code into [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public), an open-source highfidelity simulator designed for AI research in financial markets (e.g., stock exchanges). 
The simulator supports tens of thousands of clients interacting with a server to facilitate transactions (and in our case to compute sums). 
It also supports configurable pairwise network latencies.

Flamingo protocol works by steps (i.e., round trips). 
A step includes waiting and processing messages. 
The waiting time is set according to the network latency distribution and a target dropout rate.
See more details in Section 8 in our paper.

The `main` branch contains the code for private sum protocol and the `fedlearn` branch contains the code for private training of machine learning models.

## Installation Instructions
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up environment.
You can donwload Miniconda by the following command:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

To install Miniconda, run
```
bash Miniconda3-latest-Linux-x86_64.sh
```

If you use bash, then run
```
source ~/.bashrc
```

Now create an environment with python 3.9.12 and then activate it.
```
conda create --name flamingo-v0 python=3.9.12
conda activate flamingo-v0
```

Use pip to install required packages.
```
pip install -r requirements.txt
```

## Private Sum 
The code is in branch `main`.

First enter into folder `pki_files`, and run
```
python setup_pki.py
```

Our program has multiple configs.
```
-c [protocol name] 
-n [number of clients (power of 2)]
-i [number of iterations] 
-p [parallel or not] 
-o [neighborhood size (multiplicative factor of 2logn)] 
-d [debug mode, if on then print all info]
```
Flamingo supports batches of clients with size power of 2 from 128,
e.g., 128, 256, 512.

Example command:
```
python abides.py -c flamingo -n 128 -i 1 -p 1 
```

If you want to print out info at every agent, add `-d True` to the above command.

## Machine Learning Applications
The code is in branch `fedlearn`.
The machine learning model we use in this repository is a multi-layer perceptron classifier (`MLPClassfier` in `sklearn`) that can pull a variety of different datasets from the [pmlb website](https://epistasislab.github.io/pmlb/index.html). Users might wish to implement more complex models themselves.

Beyond the aforementioned configs, we provide machine learning training configs below.
```
-t [dataset name]
-s [random seed (optional)]
-e [input vector length]
-x [float-as-int encoding constant (optional)]
-y [float-as-int multiplier (optional)]
```

Example command: 
```
python abides.py -c flamingo -n 128 -i 5 -p 1 -t mnist 
```

## Additional Information
The server waiting time is set in `util/param.py` according to a target dropout rate (1%).
Specifically, for a target dropout rate, we set the waiting time according to the network latency (see `model/LatencyModel.py`). For each iteration, server total time = server waiting time + server computation time.

## Acknowledgement
We thank authors of [MicroFedML](https://eprint.iacr.org/2022/714.pdf) for providing an example template of ABIDES framework.