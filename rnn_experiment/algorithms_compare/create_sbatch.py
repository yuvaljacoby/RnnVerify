from datetime import datetime
import os
import sys
import shutil
import time

BASE_FOLDER = "."
CLUSTER_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
if os.path.exists(CLUSTER_FOLDER):
    BASE_FOLDER = CLUSTER_FOLDER 

OUT_FOLDER = os.path.join(BASE_FOLDER, "out_rns/")
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)

def check_if_model_in_dir(model_name: str, output_folder: str):
    if not output_folder:
        return False
    for f in os.listdir(output_folder):
        if model_name in f or model_name[:model_name.rfind('.')] in f:
            return True
    return False


def write_one_sbatch(sbatch_folder, t):
    exp_time = str(datetime.now()).replace(" ", "-")
    with open(os.path.join(sbatch_folder, "exact_time" + str(t) + ".sh"), "w") as slurm_file:
        slurm_file.write('#!/bin/bash\n')
        slurm_file.write('#SBATCH --job-name=rns{}_{}\n'.format(t, exp_time))
        slurm_file.write('#SBATCH --cpus-per-task=6\n')
        slurm_file.write('#SBATCH --time=24:00:00\n')
        slurm_file.write('#SBATCH --output={}/rns_{}_{}.out\n'.format(OUT_FOLDER, t, time.strftime("%Y%m%d-%H%M%S")))
        slurm_file.write('#SBATCH --mem-per-cpu=500\n')
        slurm_file.write('#SBATCH -w, --nodelist=hm-68\n') # gurobi license problems, only this node has the acadamic license
        slurm_file.write('#SBATCH --mail-user=yuvalja@cs.huji.ac.il\n')
        slurm_file.write('export LD_LIBRARY_PATH={}\n'.format(BASE_FOLDER))
        slurm_file.write('export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou\n')
        slurm_file.write('python3 rnn_experiment/algorithms_compare/experiment.py {} {}\n'.format("exact", t))

def create_sbatch(sbatch_folder: str, times: int):
    print("*" * 100)
    print("creating sbatch")
    print("*" * 100)

    os.makedirs(sbatch_folder, exist_ok=1)
    for t in range(2, times):
        write_one_sbatch(sbatch_folder, t)

if __name__ == '__main__':
    sbatch_folder = sys.argv[1]
    if out_folder == 'help':
        print("USAGE: <SBATCH_FOLDER>, <T_MAX>")
        print("creates an sbatch file that compares
                RnnVerify and RNSVerify or each time in range(2,T_MAT_MAX)")

    if len(sys.argv) > 2:
        create_sbatch(sbatch_folder, int(sys.argv[2]))
