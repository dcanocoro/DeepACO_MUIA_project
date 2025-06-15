import time
import csv
import torch
from torch.distributions import Categorical, kl

from net import Net
from aco import ACO
from utils import *

torch.manual_seed(1234)

EPS = 1e-10
device = 'cuda:0'

def infer_instance(model, demands, distances, n_ants, t_aco_diff):
    if model:
        model.eval()
        pyg_data = gen_pyg_data(demands, distances, device)
        heu_vec = model(pyg_data)
        heu_mat = heu_vec.reshape((n_node+1, n_node+1)) + EPS
        aco = ACO(
            distances=distances,
            demand=demands,
            n_ants=n_ants,
            heuristic=heu_mat,
            device=device,
            adaptive=True
        )
    else:
        aco = ACO(
            distances=distances,
            demand=demands,
            n_ants=n_ants,
            device=device,
            adaptive=True
        )
        
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        best_cost = aco.run(t)
        results[i] = best_cost
    return results
    
@torch.no_grad()
def test(dataset, model, n_ants, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for demands, distances in dataset:
        results = infer_instance(model, demands, distances, n_ants, t_aco_diff)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start


# Output
output_csv = "test_results.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["n_node", "T", "Average Objective", "Model"])

    n_ants = 5
    t_aco = [1, 10, 20, 30]

    for n_node in [20, 100]:
        test_list = load_test_dataset(n_node, device)

        # DeepACO
        net = Net().to(device)
        net.load_state_dict(torch.load(f'./pretrained/cvrp/cvrp{n_node}.pt', map_location=device))
        avg_aco_best, duration = test(test_list, net, n_ants, t_aco)
        print('DeepACO - total duration: ', duration)
        for i, t in enumerate(t_aco):
            avg = avg_aco_best[i].item()
            print("T={}, average obj. is {}.".format(t, avg))
            writer.writerow([n_node, t, avg, "DeepACO"])

        # Vanilla ACO
        avg_aco_best, duration = test(test_list, None, n_ants, t_aco)
        print('VanillaACO - total duration: ', duration)
        for i, t in enumerate(t_aco):
            avg = avg_aco_best[i].item()
            print("T={}, average obj. is {}.".format(t, avg))
            writer.writerow([n_node, t, avg, "VanillaACO"])

        print()
   