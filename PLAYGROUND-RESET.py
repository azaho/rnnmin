#!/usr/bin/env python
# coding: utf-8

# ## Importing the model

# ##### Importing packages

# In[13]:


import config
import models
import tasks
import networks
import plots
import argparse
import json
import time
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
import numpy as np
import pathlib
import shutil

plt.ioff()
matplotlib.use('Agg')

# ##### Specifying the model

# In[14]:


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
parser.add_argument('--parameter_updates', type=int,
                    help='which network to analyze?', default=0)
parser.add_argument('--ori_res', type=int,
                    help='which network to analyze?', default=3)
parser.add_argument('--redo_preanalysis', action="store_true",
                    help='hold outputs at zero?')
args = parser.parse_args()
dim_recurrent = args.dim_recurrent
index = args.index
init_random = args.random #abs(hash(args.random)) % 10**8
noise = args.noise
simple_input = args.simple_input
simple_output = args.simple_output
orientation_neurons = 32
reg_lam = args.reglam
reg_norm = args.regnorm
hold_zero = args.hold_zero
parameter_updates = args.parameter_updates # 0 = best network overall
redo_preanalysis = None if not args.redo_preanalysis else True# None = redo only if necessary


# In[15]:


ORI_RES = args.ori_res
ORI_SET = np.arange(0, 180, ORI_RES)
ORI_SET_SIZE = ORI_SET.shape[0]


# ##### Loading the model

# In[16]:


task = tasks.TWO_ORIENTATIONS_DOUBLE_OUTPUT()
model = models.CTRNN(task=task, dim_recurrent=dim_recurrent)
delay0_set = torch.arange(10, 51)
delay1_set = torch.arange(10, 51)
delay2_set = torch.arange(10, 51)

directory = f"t{task.name}_m{model.name}_dr{dim_recurrent}"
if hold_zero:
    directory += "_hz"
if reg_norm > 0:
    directory += f"_l{reg_norm}_la{reg_lam}"
if not simple_input:
    directory += "_nsi"
if not simple_output:
    directory += "_nso"
directory += f"_n{noise}_r{init_random}"
directory = "data/" + directory

if parameter_updates > 0:
    model_filename = f"model_parameterupdate{parameter_updates}.pth"
else:
    model_filename = f"model_best.pth"


# In[17]:

index = model_filename.split(".")[0]
_path = pathlib.Path(f"{directory}/{index}/megabatch_tuningdata.pt")
_path.parent.mkdir(parents=True, exist_ok=True)

shutil.rmtree(f'{directory}/{index}', ignore_errors=True)
print(f'{directory}/{index}')

_path = pathlib.Path(f"{directory}/{index}/megabatch_tuningdata.pt")
_path.parent.mkdir(parents=True, exist_ok=True)

hold_orientation_for, hold_cue_for = 50, 50
#delay0, delay1, delay2 = delay0_set[-1].item(), delay1_set[-1].item(), delay2_set[-1].item()
#delay0, delay1, delay2 = torch.median(delay0_set).item(), torch.median(delay1_set).item(), torch.median(delay2_set).item()
delay0, delay1, delay2 = 50, 50, 50
total_time = hold_orientation_for*2+hold_cue_for+delay0+delay1+delay2

orientation_neurons = 32
task = tasks.TWO_ORIENTATIONS_DOUBLE_OUTPUT(orientation_neurons, hold_orientation_for, hold_cue_for, delay0_set, delay1_set, delay2_set,
                                        simple_input=simple_input, simple_output=simple_output)
model = models.CTRNN(task=task, dim_recurrent=dim_recurrent, nonlinearity="retanh")

print("Carrying out pre-analysis...")

state_dict = torch.load(f"{directory}/{model_filename}")["model_state_dict"]
model.load_state_dict(state_dict)

####################################
print("Generating megabatch...")

def generate_megabatch(task, delay0, delay1, delay2):
    batch = []
    batch_labels = []
    output_masks = []
    for orientation1 in ORI_SET:
        for orientation2 in ORI_SET:
            to_batch, to_batch_labels, to_mask = task._make_trial(orientation1, orientation2, delay0, delay1, delay2)
            batch.append(to_batch.unsqueeze(0))
            batch_labels.append(to_batch_labels.unsqueeze(0))
            output_masks.append(to_mask.unsqueeze(0))
    return torch.cat(batch).to(config.device), torch.cat(batch_labels).to(config.device), torch.cat(
        output_masks).to(config.device)
batch = generate_megabatch(task, delay0, delay1, delay2)
print("Running the model...")
output = model(batch[0])

####################################
print("Calculating data_all...")

data_all = torch.zeros((total_time, dim_recurrent, ORI_SET_SIZE, ORI_SET_SIZE))
for orientation1 in range(ORI_SET_SIZE):
    for orientation2 in range(ORI_SET_SIZE):
        o = output[1][orientation1 * ORI_SET_SIZE + orientation2]
        data_all[:, :, orientation1, orientation2] = o

####################################
print("Calculating tuning indices...")

tuning_indices = []
for timestep in range(total_time):
    sor = []
    for i in range(dim_recurrent):
        data_in = data_all[timestep][i]
        var1 = torch.var(torch.sum(data_in, axis=1))+0.01
        var2 = torch.var(torch.sum(data_in, axis=0))+0.01
        var = (var1/var2).item()
        #if var>10:
        sor.append({"id": i, "var": var, "pref": (1 if var1>var2 else 2)})
       # print(f"UNIT {i}: {var1/var2+var2/var1}")
    sor = sorted(sor, reverse=True, key=lambda x: x["var"])
    sor_i = [x["id"] for x in sor]
    tuning_indices.append(sor_i)
tuning_indices = torch.tensor(tuning_indices, dtype=int)

####################################
print("Saving...")

result = {}
#result["sor_i"] = sor_i
#result["sor"] = sor
result["hold_orientation_for"] = hold_orientation_for
result["hold_cue_for"] = hold_cue_for
result["delay0"] = delay0
result["delay1"] = delay1
result["delay2"] = delay2

#retrievedTensor = tf.tensor(saved.data, saved.shape)

index = model_filename.split(".")[0]
torch.save(data_all, f"{directory}/{index}/megabatch_tuningdata.pt")
torch.save(tuning_indices, f"{directory}/{index}/megabatch_tuningindices.pt")
torch.save(output, f"{directory}/{index}/megabatch_output.pt")
torch.save(batch[0], f"{directory}/{index}/megabatch_input.pt")
torch.save(batch[1], f"{directory}/{index}/megabatch_target.pt")
torch.save(batch[2], f"{directory}/{index}/megabatch_mask.pt")
with open(f"{directory}/{index}/info.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("Done...")


# In[18]:


index = model_filename.split(".")[0]
with open(f"{directory}/{index}/info.json", 'r', encoding='utf-8') as f:
    j = json.load(f)

hold_orientation_for = j["hold_orientation_for"]
hold_cue_for = j["hold_cue_for"]
delay0 = j["delay0"]
delay1 = j["delay1"]
delay2 = j["delay2"]
total_time = hold_orientation_for*2+hold_cue_for+delay0+delay1+delay2

orientation_neurons = 32
task = tasks.TWO_ORIENTATIONS_DOUBLE_OUTPUT(orientation_neurons, hold_orientation_for, hold_cue_for, delay0_set, delay1_set, delay2_set,
                                            simple_input=simple_input, simple_output=simple_output)
model = models.CTRNN(task=task, dim_recurrent=dim_recurrent, nonlinearity="retanh")

state_dict = torch.load(f"{directory}/{model_filename}")["model_state_dict"]
model.load_state_dict(state_dict)

megabatch_tuningdata = torch.load(f"{directory}/{index}/megabatch_tuningdata.pt")
megabatch_tuningindices = torch.load(f"{directory}/{index}/megabatch_tuningindices.pt")
megabatch_output = torch.load(f"{directory}/{index}/megabatch_output.pt")
megabatch_input = torch.load(f"{directory}/{index}/megabatch_input.pt")
megabatch_target = torch.load(f"{directory}/{index}/megabatch_target.pt")
megabatch_mask = torch.load(f"{directory}/{index}/megabatch_mask.pt")

###### useful functions
def annotate_task_on_plt(plt):
    # add ticks
    plt.xticks([0,
                delay0,
                delay0+hold_orientation_for,
                delay0+hold_orientation_for+delay1,
                delay0+hold_orientation_for*2+delay1,
                delay0+hold_orientation_for*2+delay1+delay2,
                delay0+hold_orientation_for*2+delay1+delay2+hold_cue_for])
    # add patches to visualize inputs
    plt.axvspan(delay0, delay0+hold_orientation_for, facecolor="r", alpha=0.1)
    plt.axvspan(delay0+hold_orientation_for+delay1, delay0+hold_orientation_for*2+delay1, facecolor="b", alpha=0.1)
    plt.axvspan(delay0+hold_orientation_for*2+delay1+delay2,
                delay0+hold_orientation_for*2+delay1+delay2+hold_cue_for, facecolor="k", alpha=0.1)
    # add patches to legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(mpatches.Patch(color='r', alpha=0.3, label='O1 presented'))
    handles.append(mpatches.Patch(color='b', alpha=0.3, label='O2 presented'))
    handles.append(mpatches.Patch(color='k', alpha=0.3, label='cue presented'))
    plt.legend(handles=handles)
def images_side_by_side(images, save_to=None):
    plt.close('all')
    n = len(images)
    sqt = math.ceil(n ** 0.5)
    if n==1: nrows, ncols=1, 1
    if n==2: nrows, ncols=1, 2
    if n==3: nrows, ncols=1, 3
    if n>3: nrows, ncols=sqt, sqt
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 6*nrows))
    if nrows == 1:
        if ncols == 1: ax = np.array([[ax]])
        else: ax = np.array([ax])
    else:
        if ncols == 1: ax = np.array([[ax[0]], [ax[1]]])
    for a in ax.ravel():
        a.title.set_visible(False)
        a.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
        a.axis('off')
    for i in range(n):
        a = ax[i // sqt][i % sqt]
        a.imshow(images[i])
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    #plt.show()
    #st = fig.suptitle(f"Tuning to orientations {timestep_description}", fontsize=20)
    #st.set_y(0.95)
    #fig.supxlabel('Orientation (0 to 180 deg), orientation1 (red) and orientation2 (black)', fontsize=14)
    #fig.supylabel('Average activation (0 to 1), with standard deviation', fontsize=14)
    #fig.subplots_adjust(top=0.95, left=0.04, bottom=0.04)
    #fig.show()
    #plt.close('all')
def plt_to_image(fig):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


###### picture of tunings
print("Picture of neuron tunings...")
def get_tuning_heatmaps(timestep, timestep_description, data_all=megabatch_tuningdata, sor_i=megabatch_tuningindices[-1]):
    plt.close('all')
    plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})

    if sor_i is None:
        sor_i = megabatch_tuningindices[timestep]

    n = len(sor_i)
    sqt = math.ceil(n ** 0.5)
    fig, ax = plt.subplots(nrows=math.ceil(n / sqt), ncols=sqt, figsize=(12, 12))
    for a in ax.ravel():
        a.title.set_visible(False)
        a.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
        # a.axis('off')
    for i in range(n):
        a = ax[i // sqt][i % sqt]
        a.imshow(data_all[timestep][sor_i][i].detach().numpy(), vmin=0, vmax=1)

        # uncomment to see which unit has which id
        # a.title.set_visible(True)
        # a.set_title(sor_i[i])
    plt.tight_layout()
    st = fig.suptitle(f"Tuning to orientations {timestep_description}", fontsize=20)
    fig.supxlabel('Orientation2 (0 to 180 deg)', fontsize=14)
    fig.supylabel('Orientation1 (0 to 180 deg)', fontsize=14)
    fig.subplots_adjust(top=0.95, left=0.04, bottom=0.04)
    return fig
def get_tuning_curves(timestep, timestep_description, data_all=megabatch_tuningdata, sor_i=megabatch_tuningindices[-1]):
    plt.close('all')
    plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})

    if sor_i is None:
        sor_i = megabatch_tuningindices[timestep]

    n = len(sor_i)
    sqt = math.ceil(n ** 0.5)
    fig, ax = plt.subplots(nrows=math.ceil(n / sqt), ncols=sqt, figsize=(12, 12))
    for a in ax.ravel():
        a.title.set_visible(False)
        a.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
        # a.axis('off')
    for i in range(n):
        a = ax[i // sqt][i % sqt]
        # a.plot(range(180), torch.mean(data_all[timestep][sor_i][i], axis=0).detach().numpy(), "k")

        x = ORI_SET
        y = torch.mean(data_all[timestep][sor_i][i], axis=1).detach().numpy()
        e = torch.std(data_all[timestep][sor_i][i], dim=1).detach().numpy()
        markers, caps, bars = a.errorbar(x, y, e, fmt="r", ecolor="r")
        [bar.set_alpha(0.05) for bar in bars]
        [cap.set_alpha(0) for cap in caps]

        x = ORI_SET
        y = torch.mean(data_all[timestep][sor_i][i], axis=0).detach().numpy()
        e = torch.std(data_all[timestep][sor_i][i], dim=0).detach().numpy()
        markers, caps, bars = a.errorbar(x, y, e, fmt="k", ecolor="k")
        [bar.set_alpha(0.05) for bar in bars]
        [cap.set_alpha(0) for cap in caps]

        a.set_ylim(0, 1)
    plt.tight_layout()
    st = fig.suptitle(f"Tuning to orientations {timestep_description}", fontsize=20)
    # st.set_y(0.95)
    fig.supxlabel('Orientation (0 to 180 deg), orientation1 (red) and orientation2 (black)', fontsize=14)
    fig.supylabel('Average activation (0 to 1), with standard deviation', fontsize=14)
    fig.subplots_adjust(top=0.95, left=0.04, bottom=0.04)
    return fig
images_side_by_side((plt_to_image(get_tuning_heatmaps(total_time - 1 - hold_cue_for, "before go cue")),
                         plt_to_image(get_tuning_heatmaps(total_time - 1, "at end of task"))),
                    save_to=f"{directory}/{index}/tuning_heatmaps.pdf")

###### video
print("Making video...")
def get_title(timestep):
    if timestep >= hold_orientation_for*2+hold_cue_for+delay0+delay1+delay2:
        return "cue2 presented"
    if timestep >= hold_orientation_for*2+delay0+delay1+delay2:
        return "cue1 presented"
    if timestep >= hold_orientation_for*2+delay0+delay1:
        return "delay2"
    if timestep >= hold_orientation_for+delay0+delay1:
        return "O2 presented"
    if timestep >= hold_orientation_for+delay0:
        return "delay1"
    if timestep >= delay0:
        return "O1 presented"
    return "delay0"
dirname = "activity_video"
import pathlib
_path = pathlib.Path(f"{directory}/{index}/{dirname}/file.png")
_path.parent.mkdir(parents=True, exist_ok=True)
start_from = delay0-5
go_to = total_time
for t in range(start_from, go_to):
    plt.close('all')
    sqt = round(dim_recurrent ** 0.5)
    fig, ax = plt.subplots(nrows=sqt, ncols=sqt, figsize=(12, 12), facecolor='white')
    for i in range(dim_recurrent):
        a = ax[i // sqt][i % sqt]
        a.imshow(megabatch_tuningdata[t][megabatch_tuningindices[-1]][i].detach().numpy(), vmin=0, vmax=1)
        a.title.set_visible(False)
        a.axis('off')
        a.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
    plt.tight_layout()
    st = fig.suptitle(f"Tuning at t={t:03} ({get_title(t)})", fontsize=20)
    #st.set_y(0.95)
    fig.supxlabel('Orientation2 (0 to 180 deg)', fontsize=14)
    fig.supylabel('Orientation1 (0 to 180 deg)', fontsize=14)
    fig.subplots_adjust(top=0.95, left=0.04, bottom=0.04)
    if t % 10 == 0:
        print(f"{t}({(t-start_from)/(go_to-start_from)*100:.2f}%)", end=" ")
    if t % 50 == 0:
        print("") # for some reason printing shows only after \n has been printed
    plt.savefig(f"{directory}/{index}/{dirname}/fig{t:03}.png", bbox_inches='tight')
os.system(f"ffmpeg -framerate 10 -pattern_type glob -i '{directory}/{index}/{dirname}/*.png' -c:v libx264 -pix_fmt yuv420p {directory}/{index}/{dirname}.mp4")
print("")


##### Connectivity


exit()