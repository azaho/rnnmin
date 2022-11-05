import config
import os
import numpy as np  # https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib.pyplot as plt
import pathlib


def lr_multiplier(lr_step):
    return 1 / lr_step ** 2


# TODO: move noise_amplitude to model class
# TODO: add save_parameters_every, save_best

# outputs: error_store, gradient_norm_store
def train_network(model, task, max_steps, batch_size=64,
                  optimizer="Adam", learning_rate=1e-3,
                  add_noise=False, noise_amplitude=0.1,
                  clip_gradients=False, max_gradient_norm=10,
                  set_note_parameters=None, set_save_parameters=None,
                  save_best_network=True,
                  dir_save_parameters=None, silent=False,
                  pregenerate_data=False, dataset_to_use=None,  # dataset to use for training, if provided
                  store_errors=True, store_gradient_norms=False,
                  evaluate_plateau_every=100,  # if min training error for last N steps is more than previous N steps
                  plateau_tolerance=0.01,  # has to be within plateau_tolerance of last error
                  start_evaluating_plateau_after=1000,
                  lr_step_at_plateau=True,  # change learning rate when plateau reached
                  lr_max_steps=5,  # only do so many learning rate changes
                  lr_step_multiplier_function=lr_multiplier,  # function to change learning rate
                  start_at_best_network_after_lr_step=True,
                  stop_at_last_plateau=True,
                  regularization_lambda=0.1,
                  regularization_norm=None):  # stop after final learning rate step if plateau reached again
    if optimizer == "Adam": optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # lr = 1e-3 default
    assert (type(optimizer) is not str), f"{optimizer} is not supported by train_network (train.py)"

    error_store = -700 * np.ones(max_steps + 1)
    # error_store[0] is the error before any parameter updates have been made,
    # error_store[j] is the error after j parameter updates
    gradient_norm_store = -700 * np.ones(max_steps + 1)
    # gradient_norm[0] is norm of the gradient before any parameter updates have been made,
    # gradient_norm[j] is the norm of the gradient after j parameter updates

    if set_note_parameters is None:
        num_partitions = max(20, int(max_steps/200))
        set_note_parameters = np.unique(np.concatenate(
            (np.arange(0, min(6, max_steps)), np.round(np.linspace(0, max_steps, num=num_partitions, endpoint=True))))).astype(int)
    if set_save_parameters is None:
        set_save_parameters = set_note_parameters
    if len(set_save_parameters) > 0:
        assert (dir_save_parameters is not None), "dir_save_parameters is not given for train_network"
        if not dir_save_parameters.endswith("/"):
            dir_save_parameters = dir_save_parameters + "/"

    # Generate the whole dataset at once
    if pregenerate_data:
        inputs, targets, output_masks = task.generate_dataset(max_steps + 1, batch_size)

    best_network_dict = None
    best_network_error = None

    lr_step = 0
    for p in range(max_steps + 1):
        if pregenerate_data:
            input, target, output_mask = inputs[p], targets[p], output_masks[p]
        elif dataset_to_use is not None:
            i = p % dataset_to_use[0].shape[0]
            input, target, output_mask = dataset_to_use[0][i], dataset_to_use[1][i], dataset_to_use[2][i]
        else:
            input, target, output_mask = task.generate_batch(batch_size=batch_size)

        if not add_noise: noise_amplitude = 0
        output, h = model(input, noise_amplitude=noise_amplitude)

        # TODO: add criterion
        error = torch.sum((output[output_mask == 1] - target[output_mask == 1]) ** 2) / torch.sum(
            output_mask == 1)
        if regularization_norm == 1:
            for param in model.parameters():
                if param.requires_grad is True:
                    error += regularization_lambda * torch.sum(torch.abs(param))
        if regularization_norm == 2:
            for param in model.parameters():
                if param.requires_grad is True:
                    error += regularization_lambda * torch.sum(param ** 2)
        # output_mask: batch_size x numT x dim_output tensor, elements
        # 0(timepoint does not contribute to this term in the error function),
        # 1(timepoint contributes to this term in the error function)
        error_store[p] = error.item()

        # don't train on step 0, just store error
        if p == 0:
            best_network_dict = model.state_dict()
            best_network_error = error.item()
            if save_best_network:
                save_network(model, dir_save_parameters + f'model_best.pth')
            continue

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the error with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        error.backward()

        # clip the norm of the gradient
        if clip_gradients:
            max_gradient_norm = 10
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        if store_gradient_norms:
            gradient = []  # store all gradients
            for param in model.parameters():  # model.parameters include those defined in __init__ even if they are not used in forward pass
                if param.requires_grad is True:  # model.parameters include those defined in __init__ even if param.requires_grad is False (in this case param.grad is None)
                    gradient.append(param.grad.detach().flatten().numpy())
            gradient = np.concatenate(gradient)  # gradient = torch.cat(gradient)
            assert np.allclose(gradient.size, model.num_parameters), \
                "size of gradient and number of learned parameters don't match!"
            gradient_norm_store[p] = np.sqrt(np.sum(gradient ** 2))

        if np.isin(p, set_note_parameters):
            if not silent:
                error_wo_reg = torch.sum((output[output_mask == 1] - target[output_mask == 1]) ** 2) / torch.sum(output_mask == 1)
                print(f'{p} parameter updates: error = {error.item():.4g}, w/o reg {error_wo_reg.item():.4g}')
        if np.isin(p, set_save_parameters):
            save_network(model, dir_save_parameters + f'model_parameterupdate{p}.pth')
        if error.item() < best_network_error:
            best_network_dict = model.state_dict()
            best_network_error = error.item()
            if save_best_network:
                save_network(model, dir_save_parameters + f'model_best.pth')

        if lr_step_at_plateau and \
                p > start_evaluating_plateau_after and \
                p % evaluate_plateau_every == 0:
            min_now = min(error_store[:p + 1])
            min_before = min(error_store[:p - evaluate_plateau_every + 1])
            if min_now/min_before > 1 - plateau_tolerance:
                # reached plateau
                if not silent:
                    print(f'Reached plateau {lr_step}/{lr_max_steps} at {p} steps ({min_now:.5f} vs {min_before:.5f})')
                if lr_step == lr_max_steps and stop_at_last_plateau:
                    break
                lr_step += 1
                # TODO: do this better
                # probably better to reinitialize optimizer to reset momentum etc
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate * lr_step_multiplier_function(lr_step)
                if start_at_best_network_after_lr_step:
                    model.load_state_dict(best_network_dict)

    # get rid of the buffer (if training for less than max_steps)
    error_store = error_store[error_store != -700]
    gradient_norm_store = gradient_norm_store[gradient_norm_store != -700]


    #TODO: output network with lowest error?
    result = {
        "parameter_updates": p,
        "final_error": error.item(),
        "best_network_error": best_network_error,
        "best_network_dict": best_network_dict
    }
    if store_errors:
        result["error_store"] = error_store
    if store_gradient_norms:
        result["gradient_norm_store"] = gradient_norm_store

    return result


# save the trained model’s learned parameters
def save_network(model, path):
    _path = pathlib.Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, path)

def save_network_dict(model_dict, path):
    _path = pathlib.Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model_dict}, path)
