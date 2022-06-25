import config
import os
import numpy as np  # https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib.pyplot as plt
import pathlib


def plot_eigenvalues(model, title, path):
    _path = pathlib.Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    modelname = model.name
    W = None
    if modelname == 'Elman RNN': W = model.fc_h2h.weight.detach().numpy();
    if modelname == 'CTRNN': W = model.fc_h2ah.weight.detach().numpy();
    assert(W is not None), f"{modelname} is not supported in plot_eigenvalues (plots.py)"
    eigVal = np.linalg.eigvals(W)

    plt.figure()  # initial eigenvalue spectrum of recurrent weight matrix
    plt.plot(eigVal.real, eigVal.imag, 'k.', markersize=10)
    plt.xlabel('real(eig(W))')
    plt.ylabel('imag(eig(W))')
    plt.title(f'{title}')
    plt.axis('equal')  # plt.axis('scaled')
    plt.savefig(path, bbox_inches='tight')  # add bbox_inches='tight' to keep title from being cutoff


def plot_trainingerror(model, error_store, title, path,
                       ylabel='Error during training',
                       semilogy=False):
    _path = pathlib.Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    numparameterupdates = len(error_store)
    plt.figure()  # training error vs number of parameter updates
    if semilogy:
        plt.semilogy(np.arange(0, numparameterupdates), error_store, 'k-', linewidth=1, label=f"{model.name}")
    else:
        plt.plot(np.arange(0, numparameterupdates), error_store, 'k-', linewidth=1, label=f"{model.name}")
             #label=f'{model.name} {error_store[numparameterupdates]:.4g}')
    plt.xlabel('Number of parameter updates')
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.xlim(left=0)
    plt.ylim(top=max(error_store))
    if not semilogy: plt.ylim(bottom=0)
    plt.savefig(path, bbox_inches='tight')  # add bbox_inches='tight' to keep title from being cutoff


"""
fontsize = 14
T = np.arange(0, numTtest)  # (numTtest,)

plt.figure()  # firing of all hidden units on a single trial
# for itrial in range(numtrialstest):
for itrial in range(1):
    plt.clf()
    plt.plot(T, h[itrial, :, :])
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.ylabel(f'Firing of {dim_recurrent} hidden units', fontsize=fontsize)
    # plt.legend()
    plt.title(
        f'{modelname}, trial {itrial}\n{numtrialstest} test trials, {numTtest} timesteps in simulation\n{numparameterupdatesmodel} parameter updates, normalized error = {normalizederror:.6g}%',
        fontsize=fontsize)
    plt.xlim(left=0)
    # plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
    plt.savefig('%s/testtrial%g_numTtest%g_%gparameterupdates_h_%s%s.pdf' % (
        figdir, itrial, numTtest, numparameterupdatesmodel, modelname.replace(" ", ""), figuresuffix),
                bbox_inches='tight')  # add bbox_inches='tight' to keep title from being cutoff

plt.figure()  # firing of a single hidden unit across all trials
# for iunit in range(dim_recurrent):
for iunit in range(1):
    plt.clf()
    plt.plot(T, h[:, :, iunit].transpose())
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.ylabel(f'Firing rate of unit {iunit}\nduring {numtrialstest} test trials', fontsize=fontsize)
    # plt.legend()
    plt.title(
        f'{modelname}, unit {iunit}\n{numtrialstest} test trials, {numTtest} timesteps in simulation\n{numparameterupdatesmodel} parameter updates, normalized error = {normalizederror:.6g}%',
        fontsize=fontsize)
    plt.xlim(left=0)
    # plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
    plt.savefig('%s/unit%g_numTtest%g_%gparameterupdates_h_%s%s.pdf' % (
        figdir, iunit, numTtest, numparameterupdatesmodel, modelname.replace(" ", ""), figuresuffix),
                bbox_inches='tight')  # add bbox_inches='tight' to keep title from being cutoff
"""