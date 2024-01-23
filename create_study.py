import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, JournalFileStorage # For file-based storage!
import numpy as np

NAME = "test"
storage = JournalStorage(JournalFileStorage(f"HPO_{NAME}.log")) # ! [WARNING]Remember to keep this coherent with main_optuna_parallel:375

min_res = 100
max_res = 4001
red_factor = 4
num_brackets = np.floor(np.emath.logn(red_factor,(max_res/min_res))) + 1
print(f'[INFO] Using {num_brackets} brackets')

study = optuna.create_study(
        study_name=NAME,
        sampler=TPESampler(), 
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_res, max_resource=max_res, reduction_factor=red_factor #-> Slide 200 -> Lower it to 10 to let more trainings
        ),
        direction='maximize',
        storage=storage,
        )   #load_if_exists=True)

print(f'Created study with name {NAME}')
print(study.sampler.hyperopt_parameters())
