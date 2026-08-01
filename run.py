from run_loop import run_loop

import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore", category = UserWarning, message="resource_tracker")

function_name = str(sys.argv[1])
method_name= str(sys.argv[2])
num_al_iter = int(sys.argv[3])
n_init = int(sys.argv[4])
seed =int(sys.argv[5])
grad_q2 = False
if len(sys.argv) > 6:
    grad_q2 = sys.argv[6].lower() in {"1", "true", "yes", "y"}

results =run_loop(
    function_name = function_name,
    method_name =method_name,
    n_init = n_init,
    num_al_iter = num_al_iter,
    seed = seed,
    grad_q2= grad_q2,
)

output_dir= os.path.join("results", function_name)
os.makedirs(output_dir, exist_ok = True)
file_path = os.path.join(
    output_dir, f"{function_name}_{method_name}_{seed}.pickle"
)

with open(file_path, "wb") as f:
    pickle.dump(results, f)
