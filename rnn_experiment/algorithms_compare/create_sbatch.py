from datetime import datetime
import os
import sys
import shutil
import time

BASE_FOLDER = "/home/yuval/projects/Marabou/"
if os.path.exists("/cs/usr/yuvalja/projects/Marabou"):
    BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"

OUT_FOLDER = os.path.join(BASE_FOLDER, "ATVA_EXP/out_rns/")
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)

def check_if_model_in_dir(model_name: str, output_folder: str):
    if not output_folder:
        return False
    for f in os.listdir(output_folder):
        if model_name in f or model_name[:model_name.rfind('.')] in f:
            return True
    return False


def write_one_sbatch(output_folder, t):
    exp_time = str(datetime.now()).replace(" ", "-")
    with open(os.path.join(output_folder, "exact_time" + str(t) + ".sh"), "w") as slurm_file:
        slurm_file.write('#!/bin/bash\n')
        slurm_file.write('#SBATCH --job-name=rns{}_{}\n'.format(t, exp_time))
        slurm_file.write('#SBATCH --cpus-per-task=6\n')
        slurm_file.write('#SBATCH --time=24:00:00\n')
        slurm_file.write('#SBATCH --output={}/rns_{}_{}.out\n'.format(OUT_FOLDER, t, time.strftime("%Y%m%d-%H%M%S")))
        slurm_file.write('#SBATCH --mem-per-cpu=500\n')
        slurm_file.write('#SBATCH -w, --nodelist=hm-68\n')
        slurm_file.write('#SBATCH --mail-user=yuvalja@cs.huji.ac.il\n')
        slurm_file.write('export LD_LIBRARY_PATH=/cs/usr/yuvalja/projects/Marabou\n')
        slurm_file.write('export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou\n')
        slurm_file.write('python3 rnn_experiment/algorithms_compare/experiment.py {} {}\n'.format("exact", t))

def create_sbatch(output_folder: str, times: int):
    print("*" * 100)
    print("creating sbatch")
    print("*" * 100)

    os.makedirs(output_folder, exist_ok=1)
    for t in range(2, times):
        write_one_sbatch(output_folder, t)

if __name__ == '__main__':
    out_folder = sys.argv[1]
    if out_folder == 'help':
        print("out_folder, max_time\ncreates an sbatch file that runs only rns for each time in range(2,max_time)")
    if len(sys.argv) > 2:
        create_sbatch(out_folder, int(sys.argv[2]))
