import config
import models
import tasks
import networks
import plots
import argparse
import json
import time

parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--dim_recurrent', type=int,
                    help='dim_recurrent')
parser.add_argument('--index', type=int,
                    help='index of this trial')
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
dim_recurrent = args.dim_recurrent
index = args.index


start_time = time.time()

task = tasks.CARDS_WITH_CUES(n_cards=10, hold_card_for=3, wait_period=5, ask_card_for=3)
model = models.CTRNN(task=task, dim_recurrent=dim_recurrent)
result = networks.train_network(model, task, max_steps=30000,
                                evaluate_plateau_every=500,
                                batch_size=64,
                                silent=not args.verbose,
                                save_best_network=False,
                                set_note_parameters=[] if not args.verbose else None,
                                set_save_parameters=[])
result["training_time"] = time.time() - start_time
result["error_store"] = result["error_store"].tolist()

steps = range(0, len(result["error_store"]), int(len(result["error_store"])/100))
result["error_store"] = {step: result["error_store"][step] for step in steps}

networks.save_network_dict(result["best_network_dict"], f"data/dim_recurrent{dim_recurrent}_index{index}.pth")
model.load_state_dict(result["best_network_dict"])
del result["best_network_dict"]

result["final_accuracy"] = task.assess_accuracy(model, batch_size=512)

with open(f"data/dim_recurrent{dim_recurrent}_index{index}.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
