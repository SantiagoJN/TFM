import subprocess
import os

# Fork a child process
processid = os.fork()


interpreter = "/mnt/cephfs/home/graphics/sjimenez/anaconda3/envs/env_train_37/bin/python3.7"
script = "/mnt/cephfs/home/graphics/sjimenez/disentangling-vae-master/main_optuna_parallel.py"
models_names = "test_run1"
num_trials = 4
coordination_file = "test_run_file"

if processid > 0 : # Parent~~		GPU 0
  cmd = f"{interpreter} {script} {models_names} -t {num_trials} -g 0 -p {coordination_file} --no-progress-bar > train1.txt"
  output = subprocess.check_output(cmd, shell=True)

else : # Child~~ 			GPU 1
  cmd = f"{interpreter} {script} {models_names} -t {num_trials} -g 1 -p {coordination_file} --no-progress-bar > train2.txt"
  output = subprocess.check_output(cmd, shell=True)
 
