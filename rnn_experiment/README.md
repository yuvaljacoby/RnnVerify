Self compare experiments compare between different alpha search algorithms and for different amount of iterations

## Controlled Experiment
Is used to run comperasion between different algorithms for a given number of iterations.

For example running an experiment for the rnn4_fc32_epoch40.h5 the command is:
  
    PYTHONPATH=. python3 rnn_experiment/self_compare/experiment.py controlled model_20classes_rnn4_fc32_epoch40.h5 points_0_2_200.pkl 0 2>&1 | tee logs/rnn4_controlled_0.log

* file - rnn_experiment/self_compare/experiment.py   
args:  
* "controlled" - to indicated which type of experiment
* path_to_h5 - which rnn to use
* path_to_points - list of points to use (array where each cell is array of shape
40\)
* 0 - in which index to start running in the list points (in order to enable
parallel execution) 
* thsages to the staff's personal emails "tee" trick is to display the outputs and save them to the file.


## Iterations Experiment 
Is used to compare a single algorithm throughout time.  

#### How to run on cluster:
first we need to create the sbatch files, since we PYTHONPATH and stuff like
that we run:
    ./run_py.sh 'rnn_experiment/self_compare/create_sbatch_iterations_exp.py FMCAD_EXP/models/ FMCAD_EXP/sbatch/'


### Generating Points
To make sure all experiments runs with the same points we generate them ones and save in a pickle.

Running:

    PYTHONPATH=. python3 rnn_experiment/self_compare/generate_points.py
    
Will create the pickle (if one passes an argument it's the folder to put the pickle in, otherwise uses default) 


