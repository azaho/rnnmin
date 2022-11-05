import config
import models
import tasks
import networks
import plots
import argparse
import json
import time
import random
import hashlib
import numpy as np
import torch

"""
python3 main.py --dim_recurrent 100 --index 0 --noise 0.1 --verbose --random OA --simple_input --simple_output --hold_zero
"""

parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--dim_recurrent', type=int,
                    help='dim_recurrent', default=49)
parser.add_argument('--index', type=int,
                    help='index of this trial', default=0)
parser.add_argument('--random', type=str,
                    help='init random to', default="XZ")
parser.add_argument('--noise', type=float,
                    help='how much noise to add?', default=0)
parser.add_argument('--simple_input', action="store_true",
                    help='just input sin/cos of orientation?')
parser.add_argument('--simple_output', action="store_true",
                    help='just output sin/cos of orientation?')
parser.add_argument("--verbose", action="store_true")

parser.add_argument('--reglam', type=float,
                    help='regularization lambda?', default=0)
parser.add_argument('--regnorm', type=int,
                    help='regulatization norm to use?', default=0)
parser.add_argument('--hold_zero', action="store_true",
                    help='hold outputs at zero?')
parser.add_argument('--s_n_o', action="store_true",
                    help='SCUFFED: normalize networks outputs?')
args = parser.parse_args()
dim_recurrent = args.dim_recurrent
index = args.index
init_random = abs(hash(args.random)) % 10**8
noise = args.noise
simple_input = args.simple_input
simple_output = args.simple_output
orientation_neurons = 32
reg_lam = args.reglam
reg_norm = args.regnorm
hold_zero = args.hold_zero

random.seed(init_random)
torch.manual_seed(init_random)
np.random.seed(init_random)

start_time = time.time()

hold_orientation_for = 50
hold_cue_for = 50
delay0_set = torch.arange(10, 51)
delay1_set = torch.arange(10, 51)
delay2_set = torch.arange(10, 51)


task = tasks.TWO_ORIENTATIONS_DOUBLE_OUTPUT(orientation_neurons, hold_orientation_for, hold_cue_for, delay0_set, delay1_set, delay2_set,
                                            simple_input=simple_input, simple_output=simple_output, hold_outputs_at_zero=hold_zero)
model = models.CTRNN(task=task, dim_recurrent=dim_recurrent, nonlinearity="retanh",
                     _SCUFFED_NORMALIZE_OUTPUTS=args.s_n_o)

directory = f"t{task.name}_m{model.name}_dr{dim_recurrent}"
if hold_zero:
    directory += "_hz"
if reg_norm > 0:
    directory += f"_l{reg_norm}_la{reg_lam}"
if not simple_input:
    directory += "_nsi"
if not simple_output:
    directory += "_nso"
if args.s_n_o:
    directory += "_sno"
directory += f"_n{noise}_r{args.random}"

result = networks.train_network(model, task, max_steps=100000,
                                evaluate_plateau_every=500,
                                batch_size=64,
                                silent=not args.verbose,
                                save_best_network=True,
                                set_note_parameters=[] if not args.verbose else None,
                                set_save_parameters=[10000, 30000, 100000],
                                dir_save_parameters="data/"+directory,
                                lr_max_steps=7,
                                lr_step_at_plateau=False,
                                learning_rate=1e-3,
                                start_at_best_network_after_lr_step=True,
                                start_evaluating_plateau_after=2000,
                                add_noise=(noise>0),
                                noise_amplitude=noise,
                                regularization_lambda=reg_lam,
                                regularization_norm=reg_norm if reg_norm>0 else None,
                                clip_gradients=True)#3e-3)
result["training_time"] = time.time() - start_time
result["error_store"] = result["error_store"].tolist()

steps = range(0, len(result["error_store"]), int(len(result["error_store"])/100))
result["error_store"] = {step: result["error_store"][step] for step in steps}

networks.save_network_dict(result["best_network_dict"], f"data/{directory}/i{index}.pth")
model.load_state_dict(result["best_network_dict"])
del result["best_network_dict"]

result["final_accuracy"] = task.assess_accuracy(model, batch_size=512)

with open(f"data/{directory}/i{index}.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

exit()

######################## PREANALYSIS CODE
hold_orientation_for, hold_cue_for = 50, 50
#delay0, delay1, delay2 = delay0_set[-1].item(), delay1_set[-1].item(), delay2_set[-1].item()
delay0, delay1, delay2 = torch.median(delay0_set).item(), torch.median(delay1_set).item(), torch.median(delay2_set).item()
total_time = hold_orientation_for*2+hold_cue_for+delay0+delay1+delay2

print("Carrying out pre-analysis...")

state_dict = torch.load(f"data/{directory}/i{index}.pth")["model_state_dict"]
model.load_state_dict(state_dict)

####################################

def generate_megabatch(task, delay0, delay1, delay2):
    batch = []
    batch_labels = []
    output_masks = []
    for orientation1 in range(180):
        for orientation2 in range(180):
            to_batch, to_batch_labels, to_mask = task._make_trial(orientation1, orientation2, delay0, delay1, delay2)
            batch.append(to_batch.unsqueeze(0))
            batch_labels.append(to_batch_labels.unsqueeze(0))
            output_masks.append(to_mask.unsqueeze(0))
    return torch.cat(batch).to(config.device), torch.cat(batch_labels).to(config.device), torch.cat(
        output_masks).to(config.device)
batch = generate_megabatch(task, delay0, delay1, delay2)
output = model(batch[0])

####################################

data_all = torch.zeros((total_time, dim_recurrent, 180, 180))
for orientation1 in range(180):
    for orientation2 in range(180):
        o = output[1][orientation1 * 180 + orientation2]
        data_all[:, :, orientation1, orientation2] = o

####################################

timestep = -1 + delay0 + hold_orientation_for + delay1 + hold_orientation_for + delay2 + hold_cue_for
timestep_description = "end of task"
sor = []
for i in range(dim_recurrent):
    data_in = data_all[timestep][i]
    ridge = 0.01
    var1 = torch.var(torch.sum(data_in, axis=1)).item()
    var2 = torch.var(torch.sum(data_in, axis=0)).item()
    var = ((var1+ridge)/(var2+ridge))
    #if var>10:
    sor.append({"id": i, "var1": var1, "var2": var2, "var": var, "pref": (1 if var1>var2 else 2)})
   # print(f"UNIT {i}: {var1/var2+var2/var1}")
sor = sorted(sor, reverse=True, key=lambda x: x["var"])
sor1 = [x["id"] for x in sor if x["pref"]==1]
sor2 = [x["id"] for x in sor if x["pref"]==2]
sor_i = [x["id"] for x in sor]

####################################

result = {}
result["sor_i"] = sor_i
result["sor"] = sor
result["hold_orientation_for"] = hold_orientation_for
result["hold_cue_for"] = hold_cue_for
result["delay0"] = delay0
result["delay1"] = delay1
result["delay2"] = delay2

#retrievedTensor = tf.tensor(saved.data, saved.shape)

torch.save(data_all, f"data/{directory}/i{index}_megabatch_tuningdata.pt")
torch.save(output, f"data/{directory}/i{index}_megabatch_output.pt")
torch.save(batch[0], f"data/{directory}/i{index}_megabatch_input.pt")
torch.save(batch[1], f"data/{directory}/i{index}_megabatch_target.pt")
torch.save(batch[2], f"data/{directory}/i{index}_megabatch_mask.pt")
with open(f"data/{directory}/i{index}_info.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
