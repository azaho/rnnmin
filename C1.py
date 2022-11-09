#!/usr/bin/env python
# coding: utf-8

# ## Loading the model

# In[266]:

import argparse

import config
import models
import tasks
import networks
import plots
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
from sklearn.manifold import TSNE
import shutil


plt.ioff()
matplotlib.use('Agg')

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

ORI_RES = 3


ORI_SET = np.arange(0, 180, ORI_RES)
ORI_SET_SIZE = ORI_SET.shape[0]

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

######################## PREANALYSIS CODE
index = model_filename.split(".")[0]
_path = pathlib.Path(f"{directory}/{index}/megabatch_tuningdata.pt")
_path.parent.mkdir(parents=True, exist_ok=True)

shutil.rmtree(f'{directory}/{index}', ignore_errors=True)
print(f'{directory}/{index}')

_path = pathlib.Path(f"{directory}/{index}/megabatch_tuningdata.pt")
_path.parent.mkdir(parents=True, exist_ok=True)

hold_orientation_for, hold_cue_for = 50, 50
# delay0, delay1, delay2 = delay0_set[-1].item(), delay1_set[-1].item(), delay2_set[-1].item()
# delay0, delay1, delay2 = torch.median(delay0_set).item(), torch.median(delay1_set).item(), torch.median(delay2_set).item()
delay0, delay1, delay2 = 50, 50, 50
total_time = hold_orientation_for * 2 + hold_cue_for + delay0 + delay1 + delay2

orientation_neurons = 32
task = tasks.TWO_ORIENTATIONS_DOUBLE_OUTPUT(orientation_neurons, hold_orientation_for, hold_cue_for, delay0_set,
                                            delay1_set, delay2_set,
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
            to_batch, to_batch_labels, to_mask = task._make_trial(orientation1, orientation2, delay0, delay1,
                                                                  delay2)
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
        var1 = torch.var(torch.sum(data_in, axis=1)) + 0.01
        var2 = torch.var(torch.sum(data_in, axis=0)) + 0.01
        var = (var1 / var2).item()
        # if var>10:
        sor.append({"id": i, "var": var, "pref": (1 if var1 > var2 else 2)})
    # print(f"UNIT {i}: {var1/var2+var2/var1}")
    sor = sorted(sor, reverse=True, key=lambda x: x["var"])
    sor_i = [x["id"] for x in sor]
    tuning_indices.append(sor_i)
tuning_indices = torch.tensor(tuning_indices, dtype=int)

####################################
print("Saving...")

result = {}
# result["sor_i"] = sor_i
result["sor"] = sor
result["hold_orientation_for"] = hold_orientation_for
result["hold_cue_for"] = hold_cue_for
result["delay0"] = delay0
result["delay1"] = delay1
result["delay2"] = delay2

# retrievedTensor = tf.tensor(saved.data, saved.shape)

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

index = model_filename.split(".")[0]
with open(f"{directory}/{index}/info.json", 'r', encoding='utf-8') as f:
    j = json.load(f)

hold_orientation_for = j["hold_orientation_for"]
hold_cue_for = j["hold_cue_for"]
delay0 = j["delay0"]
delay1 = j["delay1"]
delay2 = j["delay2"]
sor = j["sor"]
total_time = hold_orientation_for * 2 + hold_cue_for + delay0 + delay1 + delay2

orientation_neurons = 32
task = tasks.TWO_ORIENTATIONS_DOUBLE_OUTPUT(orientation_neurons, hold_orientation_for, hold_cue_for, delay0_set,
                                            delay1_set, delay2_set,
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
data_all = megabatch_output
tuning_indices = megabatch_tuningindices


def make_saving_path(filename):
    return f"{directory}/{index}/{filename}"


def annotate_task_on_plt(plt):
    # add ticks
    plt.xticks([0,
                delay0,
                delay0 + hold_orientation_for,
                delay0 + hold_orientation_for + delay1,
                delay0 + hold_orientation_for * 2 + delay1,
                delay0 + hold_orientation_for * 2 + delay1 + delay2,
                delay0 + hold_orientation_for * 2 + delay1 + delay2 + hold_cue_for])
    # add patches to visualize inputs
    plt.axvspan(delay0, delay0 + hold_orientation_for, facecolor="r", alpha=0.1)
    plt.axvspan(delay0 + hold_orientation_for + delay1, delay0 + hold_orientation_for * 2 + delay1, facecolor="b",
                alpha=0.1)
    plt.axvspan(delay0 + hold_orientation_for * 2 + delay1 + delay2,
                delay0 + hold_orientation_for * 2 + delay1 + delay2 + hold_cue_for, facecolor="k", alpha=0.1)
    # add patches to legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(mpatches.Patch(color='r', alpha=0.3, label='O1 presented'))
    handles.append(mpatches.Patch(color='b', alpha=0.3, label='O2 presented'))
    handles.append(mpatches.Patch(color='k', alpha=0.3, label='cue presented'))
    plt.legend(handles=handles)


def images_side_by_side(images, save_to=None, vert_pref=False, figsize=None, title=None):
    plt.close('all')
    n = len(images)
    sqt = math.ceil(n ** 0.5)
    if n == 1: nrows, ncols = 1, 1
    if n == 2: nrows, ncols = (2, 1) if vert_pref else (1, 2)
    if n == 3: nrows, ncols = (3, 1) if vert_pref else (1, 3)
    if n > 3: nrows, ncols = sqt, sqt
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows) if figsize is None else figsize)
    if nrows == 1:
        if ncols == 1:
            ax = np.array([[ax]])
        else:
            ax = np.array([ax])
    else:
        if ncols == 1: ax = np.array([[ax[0]], [ax[1]]]).T
    for a in ax.ravel():
        a.title.set_visible(False)
        a.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
        a.axis('off')
    for i in range(n):
        a = ax[i // sqt][i % sqt]
        a.imshow(images[i])
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    if title is not None:
        st = fig.suptitle(title, fontsize=20)
        # st.set_y(0.95)
        # fig.supxlabel('Orientation (0 to 180 deg), orientation1 (red) and orientation2 (black)', fontsize=14)
        # fig.supylabel('Average activation (0 to 1), with standard deviation', fontsize=14)
        fig.subplots_adjust(top=0.95, left=0.04, bottom=0.04)
        # fig.show()
        # plt.close('all')


def show_images_side_by_side(images, vert_pref=False, figsize=None, title=None):
    images_side_by_side(images, vert_pref=vert_pref, figsize=figsize, title=title)


def plt_to_image(fig):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_tuning_heatmaps(timestep, timestep_description, data_all=None, sor_i=megabatch_tuningindices[-1]):
    plt.close('all')

    if data_all is None:
        data_all = megabatch_tuningdata
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


def get_tuning_curves(timestep, timestep_description, data_all=None, sor_i=megabatch_tuningindices[-1]):
    plt.close('all')

    if data_all is None:
        data_all = megabatch_tuningdata
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


def get_connplot_graph(timestep, cc_smoothing=True):
    R1_i = R1_indices[timestep]
    R2_i = R2_indices[timestep]
    R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1) * ORI_RES
    R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1) * ORI_RES
    W = model.fc_h2ah.weight
    sm = 10  # size of smoothing
    distances_weights = {}
    for i in range(len(R1_i)):
        for j in range(len(R1_i)):
            if j == i: continue
            diff = (R1_pref[i] - R1_pref[j]).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing:
                diff = (diff // sm) * sm
            w_ij = W[R1_i[j], R1_i[i]]  # weight from i to j
            for c in (range(-sm // 2, sm // 2 + 1) if cc_smoothing else [0]):
                if diff + c not in distances_weights: distances_weights[diff + c] = []
                distances_weights[diff + c].append(w_ij.item())
    r1_distances = np.array(sorted(distances_weights.keys()))
    r1_weights = [sum(distances_weights[diff]) / len(distances_weights[diff]) for diff in r1_distances]
    r1_weights_std = [np.std(distances_weights[diff]) for diff in r1_distances]
    distances_weights = {}
    for i in range(len(R2_i)):
        for j in range(len(R2_i)):
            if j == i: continue
            diff = (R2_pref[i] - R2_pref[j]).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing:
                diff = (diff // sm) * sm
            w_ij = W[R2_i[j], R2_i[i]]  # weight from i to j
            for c in (range(-sm // 2, sm // 2 + 1) if cc_smoothing else [0]):
                if diff + c not in distances_weights: distances_weights[diff + c] = []
                distances_weights[diff + c].append(w_ij.item())
    r2_distances = np.array(sorted(distances_weights.keys()))
    r2_weights = [sum(distances_weights[diff]) / len(distances_weights[diff]) for diff in r2_distances]
    r2_weights_std = [np.std(distances_weights[diff]) for diff in r2_distances]
    return r1_distances, r1_weights, r1_weights_std, r2_distances, r2_weights, r2_weights_std


# In[267]:


t1, t1d = -1+delay0, "before O1 presented"
t1_5, t1_5d = -1+delay0+hold_orientation_for//2, "amid 01 presentation"
t2, t2d = -1+delay0+hold_orientation_for, "after O1 presented"
t3, t3d = -1+delay0+hold_orientation_for+delay1, "before O2 presented"
t4, t4d = -1+delay0+hold_orientation_for+delay1+hold_orientation_for, "after O2 presented"
t5, t5d = -1+delay0+hold_orientation_for+delay1+hold_orientation_for+delay2, "before go cue"
t6, t6d = -1+total_time, "end of task"


# ## Tunings of neurons

# In[1]:


# s['var'] is (variance in O1 + 0.01)/(variance in O2 + 0.01)
# thus, for highly tuned units, s['var']+1/s['var'] is big. 
# for DT units, it's small. 10 is an arbitrary threshold here.
DT_i = torch.tensor([i for i, s in enumerate(sor) if s['var']+1/s['var']<2.5])
R1_ends_at_i = min(DT_i)
R2_starts_from_i = max(DT_i)+1
R1_num = R1_ends_at_i
DT_num = len(DT_i)
R2_num = dim_recurrent - R1_num - DT_num


# In[2]:


images_side_by_side((
    plt_to_image(get_tuning_heatmaps(t2, t2d, sor_i=megabatch_tuningindices[-1])),
    plt_to_image(get_tuning_heatmaps(t3, t3d, sor_i=megabatch_tuningindices[-1])),
    plt_to_image(get_tuning_heatmaps(t4, t4d, sor_i=megabatch_tuningindices[-1])),
    plt_to_image(get_tuning_heatmaps(t5, t5d, sor_i=megabatch_tuningindices[-1])),
), save_to=make_saving_path("tunings.pdf"), title=f"{R1_num} R1 - {DT_num} DT - {R2_num} R2")


# ## Ring->Output

# In[292]:


if simple_output:
    timestep = t5
    timestep_description = t5d
    R1_i = megabatch_tuningindices[timestep][:R1_ends_at_i]
    R2_i = megabatch_tuningindices[timestep][R2_starts_from_i:]
    R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
    #sort units within R1 and R2 by preferred angle
    R1_i = R1_i.clone()[torch.argsort(R1_pref)]
    R1_pref = R1_pref.clone()[torch.argsort(R1_pref)]
    R2_i = R2_i.clone()[torch.argsort(R2_pref)]
    R2_pref = R2_pref.clone()[torch.argsort(R2_pref)]

    # find closest sin and cos
    min_k_sin, min_k_cos, min_err_sin, min_err_cos = -1, -1, 1e10, 1e10
    for k in range(20):
        err = torch.sum((model.fc_h2y.weight[:, R1_i][0]-torch.sin(R1_pref/180*3.14*2)/k)**2)
        if err<min_err_sin: min_k_sin=k;min_err_sin=err
        err = torch.sum((model.fc_h2y.weight[:, R1_i][1]-torch.cos(R1_pref/180*3.14*2)/k)**2)
        if err<min_err_cos: min_k_cos=k;min_err_cos=err
    plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})
    fig = plt.figure(figsize=(12, 6))
    plt.plot(R1_pref.cpu().detach().numpy(), torch.sin(R1_pref/180*3.14*2).cpu().detach().numpy()/min_k_sin, 
             "k-", linewidth=2.5, label=f"sin(2x)/{min_k_sin}")
    plt.plot(R1_pref.cpu().detach().numpy(), torch.cos(R1_pref/180*3.14*2).cpu().detach().numpy()/min_k_cos, 
             "k--", linewidth=2.5, label=f"cos(2x)/{min_k_cos}")
    plt.plot(R1_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R1_i][0].cpu().detach().numpy(), 
             "r-", linewidth=3.5, label="R1 to sin(O1)")
    plt.plot(R1_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R1_i][1].cpu().detach().numpy(), 
             "g-", linewidth=3.5, label="R1 to cos(O1)", alpha=1)
    plt.plot(R1_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R1_i][2].cpu().detach().numpy(), 
             "b-", linewidth=3.5, label="R1 to sin(O2)", alpha=0.5)
    plt.plot(R1_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R1_i][3].cpu().detach().numpy(), 
             "b--", linewidth=3.5, label="R1 to cos(O2)", alpha=0.5)

    #plt.axhline(y=O1_error_no_reset, color='r', linestyle='--', label="R1 error, no reset")
    #plt.axhline(y=O2_error_no_reset, color='b', linestyle='--', label="R2 error, no reset")
    plt.xlabel('Preferred orientation (deg)')
    plt.ylabel('Weight from unit to output')
    plt.title(f'R1 to output connectivity ({timestep_description})')
    plt.legend()
    im1 = plt_to_image(fig)

    # find closest sin and cos
    min_k_sin, min_k_cos, min_err_sin, min_err_cos = -1, -1, 1e10, 1e10
    for k in range(20):
        err = torch.sum((model.fc_h2y.weight[:, R2_i][2]-torch.sin(R2_pref/180*3.14*2)/k)**2)
        if err<min_err_sin: min_k_sin=k;min_err_sin=err
        err = torch.sum((model.fc_h2y.weight[:, R2_i][3]-torch.cos(R2_pref/180*3.14*2)/k)**2)
        if err<min_err_cos: min_k_cos=k;min_err_cos=err
    fig = plt.figure(figsize=(12, 6))
    plt.plot(R2_pref.cpu().detach().numpy(), torch.sin(R2_pref/180*3.14*2).cpu().detach().numpy()/min_k_sin, 
             "k-", linewidth=2.5, label=f"sin(2x)/{min_k_sin}")
    plt.plot(R2_pref.cpu().detach().numpy(), torch.cos(R2_pref/180*3.14*2).cpu().detach().numpy()/min_k_cos, 
             "k--", linewidth=2.5, label=f"cos(2x)/{min_k_cos}")
    plt.plot(R2_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R2_i][0].cpu().detach().numpy(), 
             "r-", linewidth=3.5, label="R2 to sin(O1)", alpha=0.5)
    plt.plot(R2_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R2_i][1].cpu().detach().numpy(), 
             "r--", linewidth=3.5, label="R2 to cos(O1)", alpha=0.5)
    plt.plot(R2_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R2_i][2].cpu().detach().numpy(), 
             "b-", linewidth=3.5, label="R2 to sin(O2)")
    plt.plot(R2_pref.cpu().detach().numpy(), model.fc_h2y.weight[:, R2_i][3].cpu().detach().numpy(), 
             "g--", linewidth=3.5, label="R2 to cos(O2)")
    #plt.axhline(y=O1_error_no_reset, color='r', linestyle='--', label="R1 error, no reset")
    #plt.axhline(y=O2_error_no_reset, color='b', linestyle='--', label="R2 error, no reset")
    plt.xlabel('Preferred orientation (deg)')
    plt.ylabel('Weight from unit to output')
    plt.title(f'R2 to output connectivity ({timestep_description})')
    plt.legend()
    im2 = plt_to_image(fig)

    images_side_by_side((im1, im2), vert_pref=True, figsize=(12, 12), save_to=make_saving_path("ring_output_connectivity.pdf"))


# ## Clustering

# In[273]:


t_from, t_from_d = t1, t1d
t_to, t_to_d = t5, t5d
arr = megabatch_output[1].clone()[:, t_from:t_to, :].reshape(-1, dim_recurrent)
arr = arr[:, torch.cat((R1_i, DT_i, R2_i))]
arr = arr.T.cpu().detach().numpy()
corr = np.corrcoef(arr)
corr[corr!=corr]=1 # get rid of null values
#corr = np.abs(corr)
fig = plt.figure(figsize=(6, 6))
plt.imshow(corr)
plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
#plt.axis('off')
plt.xlabel("unit")
plt.ylabel("unit")
plt.title(f"correlation of unit activities\n({t_from_d} to {t_to_d})")
plt.xticks([0, 99])
plt.yticks([0, 99])
im1 = plt_to_image(fig)

tsne = TSNE(2, metric='precomputed', init='random', learning_rate='auto')
tsne_result = tsne.fit_transform(1-(corr))
#tsne_result.shape
fig = plt.figure()
#ax = fig.add_subplot()
R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
plt.scatter(tsne_result[:len(R1_i), 0], tsne_result[:len(R1_i), 1], R1_pref, color='r', label="R1 units")
plt.scatter(tsne_result[-len(R2_i):, 0], tsne_result[-len(R2_i):, 1], R2_pref, color='b', label="R2 units")
plt.scatter(tsne_result[len(R1_i):len(R1_i)+len(DT_i), 0], tsne_result[len(R1_i):len(R1_i)+len(DT_i), 1], color='g', label="DT units")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title(f"tSNE of unit activities, metric=1-corr")
plt.legend()
im2 = plt_to_image(fig)

t_from, t_from_d = t3, t3d
t_to, t_to_d = t6, t6d
arr = megabatch_output[1].clone()[:, t_from:t_to, :].reshape(-1, dim_recurrent)
arr = arr[:, torch.cat((R1_i, DT_i, R2_i))]
arr = arr.T.cpu().detach().numpy()
corr = np.corrcoef(arr)
corr[corr!=corr]=1 # get rid of null values
#corr = np.abs(corr)
fig = plt.figure(figsize=(6, 6))
plt.imshow(corr)
plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
#plt.axis('off')
plt.xlabel("unit")
plt.ylabel("unit")
plt.title(f"correlation of unit activities\n({t_from_d} to {t_to_d})")
plt.xticks([0, 99])
plt.yticks([0, 99])
im3 = plt_to_image(fig)

tsne = TSNE(2, metric='precomputed', init='random', learning_rate='auto')
tsne_result = tsne.fit_transform(1-(corr))
#tsne_result.shape
fig = plt.figure()
#ax = fig.add_subplot()
R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
plt.scatter(tsne_result[:len(R1_i), 0], tsne_result[:len(R1_i), 1], R1_pref, color='r', label="R1 units")
plt.scatter(tsne_result[-len(R2_i):, 0], tsne_result[-len(R2_i):, 1], R2_pref, color='b', label="R2 units")
plt.scatter(tsne_result[len(R1_i):len(R1_i)+len(DT_i), 0], tsne_result[len(R1_i):len(R1_i)+len(DT_i), 1], color='g', label="DT units")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title(f"tSNE of unit activities, metric=1-corr")
plt.legend()
im4 = plt_to_image(fig)

images_side_by_side((
    im1, im2, im3, im4
), save_to=make_saving_path("clustering.pdf"), title=f"Unsupervised clustering of units")


# In[274]:


t_from, t_from_d = t1, t1d
t_to, t_to_d = t5, t5d
arr = megabatch_output[1].clone()[:, t_from:t_to, :].reshape(-1, dim_recurrent)
arr = arr[:, torch.cat((R1_i, DT_i, R2_i))]
arr = arr.T.cpu().detach().numpy()
corr = np.corrcoef(arr)
corr[corr!=corr]=1 # get rid of null values
corr = np.abs(corr)
fig = plt.figure(figsize=(6, 6))
plt.imshow(corr)
plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
#plt.axis('off')
plt.xlabel("unit")
plt.ylabel("unit")
plt.title(f"|correlation| of unit activities\n({t_from_d} to {t_to_d})")
plt.xticks([0, 99])
plt.yticks([0, 99])
im1 = plt_to_image(fig)

tsne = TSNE(2, metric='precomputed', init='random', learning_rate='auto')
tsne_result = tsne.fit_transform(1-(corr))
#tsne_result.shape
fig = plt.figure()
#ax = fig.add_subplot()
R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
plt.scatter(tsne_result[:len(R1_i), 0], tsne_result[:len(R1_i), 1], R1_pref, color='r', label="R1 units")
plt.scatter(tsne_result[-len(R2_i):, 0], tsne_result[-len(R2_i):, 1], R2_pref, color='b', label="R2 units")
plt.scatter(tsne_result[len(R1_i):len(R1_i)+len(DT_i), 0], tsne_result[len(R1_i):len(R1_i)+len(DT_i), 1], color='g', label="DT units")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title(f"tSNE of unit activities, metric=1-|corr|")
plt.legend()
im2 = plt_to_image(fig)

t_from, t_from_d = t3, t3d
t_to, t_to_d = t6, t6d
arr = megabatch_output[1].clone()[:, t_from:t_to, :].reshape(-1, dim_recurrent)
arr = arr[:, torch.cat((R1_i, DT_i, R2_i))]
arr = arr.T.cpu().detach().numpy()
corr = np.corrcoef(arr)
corr[corr!=corr]=1 # get rid of null values
corr = np.abs(corr)
fig = plt.figure(figsize=(6, 6))
plt.imshow(corr)
plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
#plt.axis('off')
plt.xlabel("unit")
plt.ylabel("unit")
plt.title(f"|correlation| of unit activities\n({t_from_d} to {t_to_d})")
plt.xticks([0, 99])
plt.yticks([0, 99])
im3 = plt_to_image(fig)

tsne = TSNE(2, metric='precomputed', init='random', learning_rate='auto')
tsne_result = tsne.fit_transform(1-(corr))
#tsne_result.shape
fig = plt.figure()
#ax = fig.add_subplot()
R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
plt.scatter(tsne_result[:len(R1_i), 0], tsne_result[:len(R1_i), 1], R1_pref, color='r', label="R1 units")
plt.scatter(tsne_result[-len(R2_i):, 0], tsne_result[-len(R2_i):, 1], R2_pref, color='b', label="R2 units")
plt.scatter(tsne_result[len(R1_i):len(R1_i)+len(DT_i), 0], tsne_result[len(R1_i):len(R1_i)+len(DT_i), 1], color='g', label="DT units")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title(f"tSNE of unit activities, metric=1-|corr|")
plt.legend()
im4 = plt_to_image(fig)

images_side_by_side((
    im1, im2, im3, im4
), save_to=make_saving_path("clustering_abs.pdf"), title=f"Unsupervised clustering of units")


# ## Ring->Ring

# In[275]:


def get_connplot_graph(timestep, R1_i=None, R2_i=None, R1_pref=None, R2_pref=None, cc_smoothing=True):
    if R1_i is None:
        R1_i = megabatch_tuningindices[timestep][:R1_ends_at_i]
    if R2_i is None:
        R2_i = megabatch_tuningindices[timestep][R2_starts_from_i:]
    if R1_pref is None:
        R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    if R2_pref is None:
        R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
    W = model.fc_h2ah.weight
    sm = 10 # size of smoothing
    distances_weights = {}
    for i in range(len(R1_i)):
        for j in range(len(R1_i)):
            if j == i: continue
            diff = (R1_pref[i]-R1_pref[j]).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing: 
                diff = (diff // sm) * sm
            w_ij = W[R1_i[j], R1_i[i]] # weight from i to j 
            for c in (range(-sm//2, sm//2+1) if cc_smoothing else [0]):
                if diff+c not in distances_weights: distances_weights[diff+c] = []
                distances_weights[diff+c].append(w_ij.item())
    r1_distances = np.array(sorted(distances_weights.keys()))
    r1_weights = [sum(distances_weights[diff])/len(distances_weights[diff]) for diff in r1_distances]
    r1_weights_std = [np.std(distances_weights[diff]) for diff in r1_distances]
    distances_weights = {}
    for i in range(len(R2_i)):
        for j in range(len(R2_i)):
            if j == i: continue
            diff = (R2_pref[i]-R2_pref[j]).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing: 
                diff = (diff // sm) * sm
            w_ij = W[R2_i[j], R2_i[i]] # weight from i to j 
            for c in (range(-sm//2, sm//2+1) if cc_smoothing else [0]):
                if diff+c not in distances_weights: distances_weights[diff+c] = []
                distances_weights[diff+c].append(w_ij.item())
    r2_distances = np.array(sorted(distances_weights.keys()))
    r2_weights = [sum(distances_weights[diff])/len(distances_weights[diff]) for diff in r2_distances] 
    r2_weights_std = [np.std(distances_weights[diff]) for diff in r2_distances]
    return r1_distances, r1_weights, r1_weights_std, r2_distances, r2_weights, r2_weights_std


# In[276]:


timestep, timestep_desc = t5, t5d
r1_distances, r1_weights, r1_weights_std, r2_distances, r2_weights, r2_weights_std = get_connplot_graph(timestep)

fig = plt.figure(figsize=(10, 5))
markers, caps, bars = plt.errorbar(r1_distances, r1_weights, r1_weights_std, color='k', markersize=10, label="R1->R1 weights", capsize=0)
[bar.set_alpha(0.4) for bar in bars]
[cap.set_alpha(0.4) for cap in caps]
markers, caps, bars = plt.errorbar(r2_distances, r2_weights, r2_weights_std, color='r', markersize=10, label="R2->R2 weights", capsize=0)
[bar.set_alpha(0.4) for bar in bars]
[cap.set_alpha(0.4) for cap in caps]
plt.axhline(y=0.0, color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Difference in unit preferred orientation (deg)')
plt.ylabel('Average weight')
plt.title(f'R1 and R2 connectivity compared ({timestep_desc})')
plt.legend()
plt.xlim(-90, 90)
images_side_by_side((plt_to_image(fig),), figsize=(10, 5), 
                    save_to=make_saving_path("ring_connectivity.pdf"))


# In[277]:


def get_connplot_r1r2_graph(timestep, R1_i=None, R2_i=None, R1_pref=None, R2_pref=None, cc_smoothing=True):
    if R1_i is None:
        R1_i = megabatch_tuningindices[timestep][:R1_ends_at_i]
    if R2_i is None:
        R2_i = megabatch_tuningindices[timestep][R2_starts_from_i:]
    if R1_pref is None:
        R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    if R2_pref is None:
        R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES
    W = model.fc_h2ah.weight
    sm = 10 # size of smoothing
    distances_weights = {}
    for i in range(len(R1_i)):
        for j in range(len(R2_i)):
            #if j == i: continue
            diff = (R1_pref[i]-R2_pref[j]).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing: 
                diff = (diff // sm) * sm
            w_ij = W[R2_i[j], R1_i[i]] # weight from i to j 
            for c in (range(-sm//2, sm//2+1) if cc_smoothing else [0]):
                if diff+c not in distances_weights: distances_weights[diff+c] = []
                distances_weights[diff+c].append(w_ij.item())
    r1_distances = np.array(sorted(distances_weights.keys()))
    r1_weights = [sum(distances_weights[diff])/len(distances_weights[diff]) for diff in r1_distances]
    r1_weights_std = [np.std(distances_weights[diff]) for diff in r1_distances]
    distances_weights = {}
    for i in range(len(R2_i)):
        for j in range(len(R1_i)):
            #if j == i: continue
            diff = (R2_pref[i]-R1_pref[j]).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing: 
                diff = (diff // sm) * sm
            w_ij = W[R1_i[j], R2_i[i]] # weight from i to j 
            for c in (range(-sm//2, sm//2+1) if cc_smoothing else [0]):
                if diff+c not in distances_weights: distances_weights[diff+c] = []
                distances_weights[diff+c].append(w_ij.item())
    r2_distances = np.array(sorted(distances_weights.keys()))
    r2_weights = [sum(distances_weights[diff])/len(distances_weights[diff]) for diff in r2_distances] 
    r2_weights_std = [np.std(distances_weights[diff]) for diff in r2_distances]
    return r1_distances, r1_weights, r1_weights_std, r2_distances, r2_weights, r2_weights_std


# In[278]:


R1_i = megabatch_tuningindices[t5][:R1_ends_at_i]
R2_i = megabatch_tuningindices[t5][R2_starts_from_i:]
images = []
for i, x in enumerate([(t1_5, t1_5d), (t3, t3d), (t4, t4d), (t5, t5d)]):
    timestep, timestep_desc = x
    R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES

    r1_distances, r1_weights, r1_weights_std, r2_distances, r2_weights, r2_weights_std = get_connplot_r1r2_graph(timestep, R1_i, R2_i, R1_pref, R2_pref)

    fig = plt.figure(figsize=(10, 5))
    markers, caps, bars = plt.errorbar(r1_distances, r1_weights, r1_weights_std, color='k', markersize=10, label="R1->R2", capsize=0)
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]
    markers, caps, bars = plt.errorbar(r2_distances, r2_weights, r2_weights_std, color='r', markersize=10, label="R2->R1", capsize=0)
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]
    plt.axhline(y=0.0, color='k', linestyle='-', linewidth=0.3)
    plt.xlabel('Difference in unit preferred orientation (deg)')
    plt.ylabel('Average weight')
    plt.title(f'R1 and R2 connectivity ({timestep_desc})')
    plt.legend()
    plt.xlim(-90, 90)
    images.append(plt_to_image(fig))
images_side_by_side(images, 
                    save_to=make_saving_path("r1_r2_connectivity.pdf"), 
                    title="R1 and R2 determined at before go cue; What changes is their preferred orientation",
                    figsize=(12, 7))


# ## Input->Ring

# In[63]:


def get_connplot_iu_graph(units_id, unit_pref, timestep, cc_smoothing=True):
    sm = 10 # size of smoothing
    
    distances_weights = {}
    for i in range(len(units_id)):
        for j in range(orientation_neurons):
            if j == i: continue
            diff = (unit_pref[i]-round(180*j/orientation_neurons)).item()
            diff = (diff + 180 + 90) % 180 - 90
            if not cc_smoothing: 
                diff = (diff // sm) * sm
            w_ij = model.fc_x2ah.weight[units_id[i], j] # weight from i to j 
            for c in (range(-sm//2, sm//2+1) if cc_smoothing else [0]):
                if diff+c not in distances_weights: distances_weights[diff+c] = []
                distances_weights[diff+c].append(w_ij.item())
                
    o1_distances = np.array(sorted(distances_weights.keys()))
    o1_weights = [sum(distances_weights[diff])/len(distances_weights[diff]) for diff in o1_distances]
    o1_weights_std = [np.std(distances_weights[diff]) for diff in o1_distances]
    return o1_distances, o1_weights, o1_weights_std


# In[64]:


R1_i = megabatch_tuningindices[t5][:R1_ends_at_i]
R2_i = megabatch_tuningindices[t5][R2_starts_from_i:]
imagess = []
for i, x in enumerate([(t1_5, t1_5d), (t3, t3d), (t4, t4d), (t5, t5d)]):
    timestep, timestep_desc = x
    R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES

    o1_distances, o1_weights, o1_weights_std = get_connplot_iu_graph(R1_i, R1_pref, timestep)
    o2_distances, o2_weights, o2_weights_std = get_connplot_iu_graph(R2_i, R2_pref, timestep)
    fig = plt.figure(figsize=(10, 5))
    markers, caps, bars = plt.errorbar(o1_distances, o1_weights, o1_weights_std, color='k', markersize=10, label="IN->R1", capsize=0)
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]
    if timestep >= t4:
        markers, caps, bars = plt.errorbar(o2_distances, o2_weights, o2_weights_std, color='r', markersize=10, label="IN->R2", capsize=0)
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]
    plt.axhline(y=0.0, color='k', linestyle='-', linewidth=0.3)
    plt.xlabel('Difference in unit preferred orientation (deg)')
    plt.ylabel('Average weight')
    plt.title(f'Input to R1 and R2 connectivity ({timestep_desc})')
    plt.legend()
    plt.xlim(-90, 90)
    imagess.append(plt_to_image(fig))
images_side_by_side(imagess, 
                    save_to=make_saving_path("ring_input_connectivity_1.pdf"), 
                    title="R1 and R2 determined at before go cue; What changes is their preferred orientation",
                    figsize=(12, 7))


# In[65]:


R1_i = megabatch_tuningindices[t5][:R1_ends_at_i]
R2_i = megabatch_tuningindices[t5][R2_starts_from_i:]
imagess = []
for i, x in enumerate([(t1_5, t1_5d), (t3, t3d), (t4, t4d), (t5, t5d)]):
    timestep, timestep_desc = x
    R1_i = megabatch_tuningindices[timestep][:R1_ends_at_i]
    R2_i = megabatch_tuningindices[timestep][R2_starts_from_i:]
    R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES

    o1_distances, o1_weights, o1_weights_std = get_connplot_iu_graph(R1_i, R1_pref, timestep)
    o2_distances, o2_weights, o2_weights_std = get_connplot_iu_graph(R2_i, R2_pref, timestep)
    fig = plt.figure(figsize=(10, 5))
    markers, caps, bars = plt.errorbar(o1_distances, o1_weights, o1_weights_std, color='k', markersize=10, label="IN->R1", capsize=0)
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]
    if timestep >= t4:
        markers, caps, bars = plt.errorbar(o2_distances, o2_weights, o2_weights_std, color='r', markersize=10, label="IN->R2", capsize=0)
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]
    plt.axhline(y=0.0, color='k', linestyle='-', linewidth=0.3)
    plt.xlabel('Difference in unit preferred orientation (deg)')
    plt.ylabel('Average weight')
    plt.title(f'Input to R1 and R2 connectivity ({timestep_desc})')
    plt.legend()
    plt.xlim(-90, 90)
    imagess.append(plt_to_image(fig))
images_side_by_side(imagess, 
                    save_to=make_saving_path("ring_input_connectivity_2.pdf"), 
                    title="R1 and R2 determined at timestep",
                    figsize=(12, 7))


# In[66]:


R1_i = megabatch_tuningindices[t5][:R1_ends_at_i]
R2_i = megabatch_tuningindices[t5][R2_starts_from_i:]
imagess = []
for i, x in enumerate([(t1_5, t1_5d), (t3, t3d), (t4, t4d), (t5, t5d)]):
    timestep, timestep_desc = x
    R1_i = megabatch_tuningindices[timestep][:R1_ends_at_i]
    R2_i = megabatch_tuningindices[timestep][R2_starts_from_i:]
    R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R2_i], dim=1), dim=1)*ORI_RES

    if timestep < t4:
        R1_i = megabatch_tuningindices[timestep]
        R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[timestep][R1_i], dim=2), dim=1)*ORI_RES
    
    o1_distances, o1_weights, o1_weights_std = get_connplot_iu_graph(R1_i, R1_pref, timestep)
    o2_distances, o2_weights, o2_weights_std = get_connplot_iu_graph(R2_i, R2_pref, timestep)
    fig = plt.figure(figsize=(10, 5))
    markers, caps, bars = plt.errorbar(o1_distances, o1_weights, o1_weights_std, color='k', markersize=10, label="IN->R1", capsize=0)
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]
    if timestep >= t4:
        markers, caps, bars = plt.errorbar(o2_distances, o2_weights, o2_weights_std, color='r', markersize=10, label="IN->R2", capsize=0)
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]
    plt.axhline(y=0.0, color='k', linestyle='-', linewidth=0.3)
    plt.xlabel('Difference in unit preferred orientation (deg)')
    plt.ylabel('Average weight')
    plt.title(f'Input to R1 and R2 connectivity ({timestep_desc})')
    plt.legend()
    plt.xlim(-90, 90)
    imagess.append(plt_to_image(fig))
images_side_by_side(imagess, 
                    save_to=make_saving_path("ring_input_connectivity_3.pdf"), 
                    title="R1 and R2 determined at timestep, R1 is ALL units in first 2 figures",
                    figsize=(12, 7))


# ## Overlap with R1 and R2

# In[279]:


R1_indices = megabatch_tuningindices[:, :R1_ends_at_i]
DT_indices = megabatch_tuningindices[:, R1_ends_at_i:R2_starts_from_i]
R2_indices = megabatch_tuningindices[:, R2_starts_from_i:]
intersection_size_tr1_r1 = [np.intersect1d(R1_indices[-1], R1_indices[t]).shape[0]/R1_indices[-1].shape[0] for t in range(total_time)]
intersection_size_tr1_dt = [np.intersect1d(DT_indices[-1], R1_indices[t]).shape[0]/R1_indices[-1].shape[0] for t in range(total_time)]
intersection_size_tr1_r2 = [np.intersect1d(R2_indices[-1], R1_indices[t]).shape[0]/R1_indices[-1].shape[0] for t in range(total_time)]
intersection_size_r2 = [np.intersect1d(R2_indices[-1], R2_indices[t]).shape[0]/R2_indices[-1].shape[0] for t in range(total_time)]
time = range(1, total_time+1)

font = {'family' : 'Ariel',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

plt.figure(figsize=(8, 4))
plt.plot(time, intersection_size_tr1_r1, "r-", linewidth=3.5, markersize=10, label="tR1 to R1")
plt.plot(time, intersection_size_tr1_dt, "g-", linewidth=3.5, markersize=10, label="tR1 to DT")
plt.plot(time, intersection_size_tr1_r2, "b-", linewidth=3.5, markersize=10, label="tR1 to R2")
#plt.plot(time, intersection_size_r2, "b-", linewidth=3.5, markersize=10, label="R2 units")
plt.xlabel('timestep')
plt.ylabel('Overlap with last-timestep tuned units')
plt.title(f'How similar are ring units to last-timestep ring units?')
plt.legend()
annotate_task_on_plt(plt)
plt.savefig(make_saving_path("R1_similarity.pdf"), bbox_inches='tight')


# ## PCA

# In[280]:


from sklearn.decomposition import PCA

R1_i = megabatch_tuningindices[-1][:R1_ends_at_i]
DT_i = megabatch_tuningindices[-1][R1_ends_at_i:R2_starts_from_i]
R2_i = megabatch_tuningindices[-1][R2_starts_from_i:]
R1_pref = torch.argmax(torch.sum(megabatch_tuningdata[-1][R1_i], dim=2), dim=1)*ORI_RES
R2_pref = torch.argmax(torch.sum(megabatch_tuningdata[-1][R2_i], dim=1), dim=1)*ORI_RES

######################### R1

# In[293]:


t_from, t_from_d = t1, t1d
t_to, t_to_d = t5, t5d

# timestep_from, timestep_to = 0, 300

indices = torch.cat((R1_i,))
activity_of = "R1"

arr = megabatch_output[1].clone()[:, t_from:t_to, :].reshape(-1, dim_recurrent)
# arr = arr[:, torch.cat((R1_i, DT_i))]#R2_indices[-1]]
arr = arr[:, indices]  # R2_indices[-1]]
arr = arr.cpu().detach().numpy()

pca = PCA(n_components=10, svd_solver='full')
arr_pca = pca.fit_transform(arr)

fig = plt.figure(figsize=(6, 3))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.title(f"Explained variance in {activity_of} activity by PCA\n({t_from_d} to {t_to_d})")
plt.xlabel("component #")
plt.ylabel("ratio of explained variance")
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.savefig(make_saving_path(f"pca_{activity_of}_{t_from}to{t_to}_explainedvariance.pdf"), bbox_inches='tight')

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
# res = res[:, :, :]
c = [t for o1 in range(res.shape[0]) for o2 in range(res.shape[1]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot()
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by timestep\n({t_from_d} to {t_to_d})")
im1 = plt_to_image(fig)

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
res = res[:, :, :]
c = [o1 for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot()
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n({t_from_d} to {t_to_d})")
im2 = plt_to_image(fig)

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
res = res[:, :, :]
c = [t for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot(projection='3d')
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 2], res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC3")
ax.set_ylabel("PC1")
ax.set_zlabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by timestep\n({t_from_d} to {t_to_d})")
im3 = plt_to_image(fig)

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
res = res[:, :, :]
c = [o1 for o1 in range(res.shape[0]) for o2 in range(res.shape[1]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot(projection='3d')
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 2], res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC3")
ax.set_ylabel("PC1")
ax.set_zlabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n({t_from_d} to {t_to_d})")
im4 = plt_to_image(fig)

images_side_by_side((im1, im2, im3, im4), save_to=make_saving_path(f"pca_{activity_of}_{t_from}to{t_to}_plots.pdf"))


# ### Videos of PCA

# In[294]:


def get_title(timestep):
    if timestep >= hold_orientation_for * 2 + hold_cue_for + delay0 + delay1 + delay2:
        return "cue2 presented"
    if timestep >= hold_orientation_for * 2 + delay0 + delay1 + delay2:
        return "cue1 presented"
    if timestep >= hold_orientation_for * 2 + delay0 + delay1:
        return "delay2"
    if timestep >= hold_orientation_for + delay0 + delay1:
        return "orientation2 presented"
    if timestep >= hold_orientation_for + delay0:
        return "delay1"
    if timestep >= delay0:
        return "orientation1 presented"
    return "delay0"

def _pca_3d_r1(ax, timestep):
    a = timestep
    t = -t1+a
    res = arr_pca.reshape(180//ORI_RES, 180//ORI_RES, t_to-t_from, 10)
    res = res[:, :, t:t+1, :]
    c = [o1 for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
    res = res.reshape(-1, 10)
    #ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
    #ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
    #ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
    ax.scatter(res[:, 1], res[:, 0], res[:, 2], c=c, s=10)
    ax.set_xlim(np.min(arr_pca, axis=0)[1], np.max(arr_pca, axis=0)[1])
    ax.set_ylim(np.min(arr_pca, axis=0)[0], np.max(arr_pca, axis=0)[0])
    ax.set_zlim(np.min(arr_pca, axis=0)[2], np.max(arr_pca, axis=0)[2])
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC3")
    ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n(PCA {t_from_d} to {t_to_d})"+
                 f"\n(after {get_title(a)})")
def _pca_2d_r1(ax, timestep):
    a = timestep
    t = -t1+a
    res = arr_pca.reshape(180//ORI_RES, 180//ORI_RES, t_to-t_from, 10)
    res = res[:, :, t:t+1, :]
    c = [o1 for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
    res = res.reshape(-1, 10)
    #ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
    #ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
    #ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
    ax.scatter(res[:, 0], res[:, 1], c=c, s=10)
    ax.set_xlim(np.min(arr_pca, axis=0)[0], np.max(arr_pca, axis=0)[0])
    ax.set_ylim(np.min(arr_pca, axis=0)[1], np.max(arr_pca, axis=0)[1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n(PCA {t_from_d} to {t_to_d})"+
                 f"\n(after {get_title(a)})")

def _pca_3d_r2(ax, timestep):
    a = timestep
    t = -t1+a
    res = arr_pca.reshape(180//ORI_RES, 180//ORI_RES, t_to-t_from, 10)
    res = res[:, :, t:t+1, :]
    c = [o2 for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
    res = res.reshape(-1, 10)
    #ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
    #ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
    #ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
    ax.scatter(res[:, 1], res[:, 0], res[:, 2], c=c, s=10)
    ax.set_xlim(np.min(arr_pca, axis=0)[1], np.max(arr_pca, axis=0)[1])
    ax.set_ylim(np.min(arr_pca, axis=0)[0], np.max(arr_pca, axis=0)[0])
    ax.set_zlim(np.min(arr_pca, axis=0)[2], np.max(arr_pca, axis=0)[2])
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC3")
    ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n(PCA {t_from_d} to {t_to_d})"+
                 f"\n(after {get_title(a)})")
def _pca_2d_r2(ax, timestep):
    a = timestep
    t = -t1+a
    res = arr_pca.reshape(180//ORI_RES, 180//ORI_RES, t_to-t_from, 10)
    res = res[:, :, t:t+1, :]
    c = [o2 for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
    res = res.reshape(-1, 10)
    #ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
    #ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
    #ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
    ax.scatter(res[:, 0], res[:, 1], c=c, s=10)
    ax.set_xlim(np.min(arr_pca, axis=0)[0], np.max(arr_pca, axis=0)[0])
    ax.set_ylim(np.min(arr_pca, axis=0)[1], np.max(arr_pca, axis=0)[1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n(PCA {t_from_d} to {t_to_d})"+
                 f"\n(after {get_title(a)})")

# In[297]:

plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})
plt.close('all')
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(2, 2, 1, projection='3d')
_pca_3d_r1(ax, t2)
ax = fig.add_subplot(2, 2, 2, projection='3d')
_pca_3d_r1(ax, t3)
ax = fig.add_subplot(2, 2, 3, projection='3d')
_pca_3d_r1(ax, t4)
ax = fig.add_subplot(2, 2, 4, projection='3d')
_pca_3d_r1(ax, t5-1)
plt.savefig(make_saving_path(f"pca_{activity_of}_3d.pdf"), bbox_inches='tight')
plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})

plt.close('all')
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(2, 2, 1)
_pca_3d_r1(ax, t2)
ax = fig.add_subplot(2, 2, 2)
_pca_3d_r1(ax, t3)
ax = fig.add_subplot(2, 2, 3)
_pca_3d_r1(ax, t4)
ax = fig.add_subplot(2, 2, 4)
_pca_3d_r1(ax, t5-1)
plt.savefig(make_saving_path(f"pca_{activity_of}_2d.pdf"), bbox_inches='tight')

# In[293]:


t_from, t_from_d = t1, t1d
t_to, t_to_d = t5, t5d

# timestep_from, timestep_to = 0, 300

indices = torch.cat((R2_i,))
activity_of = "R2"

arr = megabatch_output[1].clone()[:, t_from:t_to, :].reshape(-1, dim_recurrent)
# arr = arr[:, torch.cat((R1_i, DT_i))]#R2_indices[-1]]
arr = arr[:, indices]  # R2_indices[-1]]
arr = arr.cpu().detach().numpy()

pca = PCA(n_components=10, svd_solver='full')
arr_pca = pca.fit_transform(arr)

fig = plt.figure(figsize=(6, 3))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.title(f"Explained variance in {activity_of} activity by PCA\n({t_from_d} to {t_to_d})")
plt.xlabel("component #")
plt.ylabel("ratio of explained variance")
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.savefig(make_saving_path(f"pca_{activity_of}_{t_from}to{t_to}_explainedvariance.pdf"), bbox_inches='tight')

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
# res = res[:, :, :]
c = [t for o1 in range(res.shape[0]) for o2 in range(res.shape[1]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot()
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by timestep\n({t_from_d} to {t_to_d})")
im1 = plt_to_image(fig)

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
res = res[:, :, :]
c = [o1 for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot()
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n({t_from_d} to {t_to_d})")
im2 = plt_to_image(fig)

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
res = res[:, :, :]
c = [t for o1 in range(res.shape[1]) for o2 in range(res.shape[0]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot(projection='3d')
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 2], res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC3")
ax.set_ylabel("PC1")
ax.set_zlabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by timestep\n({t_from_d} to {t_to_d})")
im3 = plt_to_image(fig)

arr_pca.shape
res = arr_pca.reshape(180 // ORI_RES, 180 // ORI_RES, t_to - t_from, 10)
res = res[:, :, :]
c = [o1 for o1 in range(res.shape[0]) for o2 in range(res.shape[1]) for t in range(res.shape[2])]
res = res.reshape(-1, 10)
fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot()
ax = fig.add_subplot(projection='3d')
# ax.scatter(tsne_result[:len(R1_indices[-1]), 0], tsne_result[:len(R1_indices[-1]), 1], color='r', label="R1 units")
# ax.scatter(tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 0], tsne_result[len(R1_indices[-1]):len(R1_indices[-1])+len(DT_indices[-1]), 1], color='g', label="DT units")
# ax.scatter(tsne_result[-len(R2_indices[-1]):, 0], tsne_result[-len(R2_indices[-1]):, 1], color='b', label="R2 units")
ax.scatter(res[:, 2], res[:, 0], res[:, 1], c=c, s=1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-4.5, 5)
# ax.legend()
ax.set_xlabel("PC3")
ax.set_ylabel("PC1")
ax.set_zlabel("PC2")
ax.set_title(f"PCA on {activity_of} activity, colored by orientation1\n({t_from_d} to {t_to_d})")
im4 = plt_to_image(fig)

images_side_by_side((im1, im2, im3, im4), save_to=make_saving_path(f"pca_{activity_of}_{t_from}to{t_to}_plots.pdf"))


# ### Videos of PCA

# In[294]:


def get_title(timestep):
    if timestep >= hold_orientation_for * 2 + hold_cue_for + delay0 + delay1 + delay2:
        return "cue2 presented"
    if timestep >= hold_orientation_for * 2 + delay0 + delay1 + delay2:
        return "cue1 presented"
    if timestep >= hold_orientation_for * 2 + delay0 + delay1:
        return "delay2"
    if timestep >= hold_orientation_for + delay0 + delay1:
        return "orientation2 presented"
    if timestep >= hold_orientation_for + delay0:
        return "delay1"
    if timestep >= delay0:
        return "orientation1 presented"
    return "delay0"


# In[297]:

# In[297]:

plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})
plt.close('all')
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(2, 2, 1, projection='3d')
_pca_3d_r2(ax, t2)
ax = fig.add_subplot(2, 2, 2, projection='3d')
_pca_3d_r2(ax, t3)
ax = fig.add_subplot(2, 2, 3, projection='3d')
_pca_3d_r2(ax, t4)
ax = fig.add_subplot(2, 2, 4, projection='3d')
_pca_3d_r2(ax, t5-1)
plt.savefig(make_saving_path(f"pca_{activity_of}_3d.pdf"), bbox_inches='tight')
plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14})

plt.close('all')
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(2, 2, 1)
_pca_3d_r2(ax, t2)
ax = fig.add_subplot(2, 2, 2)
_pca_3d_r2(ax, t3)
ax = fig.add_subplot(2, 2, 3)
_pca_3d_r2(ax, t4)
ax = fig.add_subplot(2, 2, 4)
_pca_3d_r2(ax, t5-1)
plt.savefig(make_saving_path(f"pca_{activity_of}_2d.pdf"), bbox_inches='tight')


# In[ ]:

from PyPDF2 import PdfMerger

files = [f for f in os.listdir(make_saving_path("")) if f.endswith('.pdf')]
files.sort(key=lambda x: os.path.getmtime(make_saving_path(x)))
pdfs = [make_saving_path(i) for i in files]

merger = PdfMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write(f"{directory}_{model_filename}.pdf")
