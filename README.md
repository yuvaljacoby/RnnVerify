*** RnnVerify, June 2020 ***

This repository contains the proof-of-concept implementation of the
RnnVerify tool, as described in the paper:

   Y. Jacoby, C. Barrett and G. Katz. Verifying Recurrent Neural Networks using
   Invariant Inference


The paper may be found at:

    https://arxiv.org/abs/2004.02462

This file contains instructions for compiling RnnVerify and for running
the experiments described in the paper, and also some information on
the RnnVerify code and the various folders.

## Compilation Instructions

The implementation was run and tested on Ubuntu 16.04.


Installing Gurobi:
    [Get Gurobi license](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) (free academic license is available) 
    Install the Gurobi [Python Interface](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html)

Compiling Marabou:

      mkdir build
	  cd build
	  cmake ..
      cmake --build .

Get python dependencies (we do recommand using virtualenv): 

    pip install -r requirements.txt

Note that this will *not* install gurobipy (since it does not support pip), [here](https://support.gurobi.com/hc/en-us/community/posts/360046430451-installing-gurobi-in-python-requires-root-access) is one solution to make gurobipy work in virtualenv

## Running the experiments

The experiments files are in rnn_experiment folder and can be devided into two
categories:

  (i) algorithms_compare - Compression of RnnVerify to the Unrolling method used in
  [RNSVerify](https://www.aaai.org/ojs/index.php/AAAI/article/view/4555).

  (ii) self_compare - Using RnnVerify to prove local robustness for Speaker
  Recognition networks.

This repository contains the code for all described experiments. All experiments
are run using simple python scripts, we keep raw results in pickle's (and save
them in the 'pickles' folder).

To run the first experiment (comperssion with RNSverify), use:
    PYTHONPATH=. python3 rnn_experiment/algorithms_compare/experiment.py exp 25 

For the second experiment:
    PYTHONPATH=. python3 rnn_experiment/self_compare/experiment.py exp all 20

## Information regarding the RnnVerify code

The main folder of the tool are:

1. RNN :
    The added implementation on top of the feed forward verification tool.
    Contains integration with the Marabou framework (via python API), and the
    implementation of the suggested algorithms

2. src - Marabou implementation :
    Marabou is a Verification tool for Feed Forward Networks
    For more details see the [paper](http://aisafety.stanford.edu/marabou/MarabouCAV2019.pdf) or [Marabou repository](https://github.com/NeuralNetworkVerification/Marabou)


3. models:
    This folder contains tensorflow-keras trained networks for speaker
    recognition which were used for the different experiments.


### Tests
There are multiple test suites that demonstare a simple query to RnnVerify and
demonstare the method on more (random) networks.
To run all tests use:
    PYTHONPATH=. pytest RNN/*

