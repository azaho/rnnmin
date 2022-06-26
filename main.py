import config
import models
import tasks
import networks
import plots


MIN_ACCEPTABLE_ACCURACY = 1.0
N_CARDS_FROM = 3
N_CARDS_TO = 10
DIM_RECURRENT_FROM = 0
DIM_RECURRENT_TO = 100
REPEAT_EVERY_TRY = 10
TRAIN_FOR_STEPS = 3000


# Try REPEAT_EVERY_TRY times to find a network with dim_recurrent recurrent units to solve the task
def find_acceptable_network(task, dim_recurrent, silent=False):
    dataset = task.generate_dataset(TRAIN_FOR_STEPS+1, batch_size=64)
    for i in range(REPEAT_EVERY_TRY):
        model = models.CTRNN(task=task, dim_recurrent=dim_recurrent).to(config.device)

        networks.train_network(model, task, max_steps=TRAIN_FOR_STEPS, silent=True, dataset_to_use=dataset)
        accuracy = task.assess_accuracy(model)

        if not silent: print(f"== TRY {i+1}: accuracy = {accuracy}")

        if accuracy >= MIN_ACCEPTABLE_ACCURACY:
            return accuracy, model
    return None


# Use binary search to find the smallest network capable of solving the task
def find_min_acceptable_network(task, silent=False):
    low = DIM_RECURRENT_FROM
    high = DIM_RECURRENT_TO

    if not silent: print(f"\nFOR {task.n_cards} CARDS")

    mid = 0
    last_acceptable_units = None
    last_acceptable_model = None
    while low <= high:
        mid = (high + low) // 2
        if not silent: print(f"= FOR {mid} RECURRENT UNITS")

        res = find_acceptable_network(task, mid, silent=silent)
        if res is None:
            low = mid + 1
        else:
            high = mid - 1
            last_acceptable_units = mid
            last_acceptable_model = res[1]

    if last_acceptable_units is None:
        if not silent: print(f"MIN NETWORK HAS >{DIM_RECURRENT_TO} UNITS")
        return None
    else:
        if not silent: print(f"MIN NETWORK HAS {last_acceptable_units} UNITS")
        return last_acceptable_units, last_acceptable_model

"""
if __name__=="__main__":
    print(f"Working device: {config.device}")
    for n_cards in range(N_CARDS_FROM, N_CARDS_TO + 1):
        task = tasks.CARDS_WITH_CUES(n_cards=n_cards, hold_card_for=3, wait_period=5, ask_card_for=3)
        res = find_min_acceptable_network(task)

        if res is None:
            break
        networks.save_network(res[1], f"{n_cards}.pth")
"""

"""
Simply train on one task and plot
"""

print(f"Using {config.device}")
task = tasks.CARDS_WITH_CUES(n_cards=20, hold_card_for=3, wait_period=5, ask_card_for=3)
model = models.CTRNN(task=task, dim_recurrent=500)

plots.plot_eigenvalues(model, "Eigenvalues before training", "data/CARDS4/eig_before.pdf")

result = networks.train_network(model, task, max_steps=100000,
                                                          dir_save_parameters="data/CARDS4/parameters/",
                                                          evaluate_plateau_every=500б
                                                        batch_size=128)
error_store = result["error_store"]

plots.plot_eigenvalues(model, "Eigenvalues after training", "data/CARDS4/eig_after.pdf")
plots.plot_trainingerror(model, error_store, "Error during training", "data/CARDS4/error.pdf")
plots.plot_trainingerror(model, error_store, "Error during training", "data/CARDS4/error_log.pdf", semilogy=True)

