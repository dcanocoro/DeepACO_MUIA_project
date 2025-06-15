from aco import ACO
from net import Net
from utils import load_test_dataset, gen_pyg_data
import torch, time, itertools, csv

device      = 'cuda:0'
n_ants      = 5
iterations  = 3
variants    = [("AS", False, False),
               ("Elitist", True, False),
               ("MMAS", True, True)]

def run_one_instance(dem, dist, heu_mat, elitist, min_max):
    aco = ACO(distances=dist, demand=dem,
              n_ants=n_ants, heuristic=heu_mat,
              elitist=elitist, min_max=min_max, device=device)
    best = aco.run(iterations)
    return best

def main(size):
    # load data and network
    data  = load_test_dataset(size, device)
    net   = Net().to(device)
    net.load_state_dict(torch.load(f'./pretrained/cvrp/cvrp{size}.pt',
                                   map_location=device))
    net.eval()

    # iterate over instances
    results = []          # rows: (instance_id, variant, learned?, cost)
    for idx,(dem,dist) in enumerate(data):
        heu_vec = net(gen_pyg_data(dem, dist, device))
        heu_mat = heu_vec.reshape(size+1, size+1) + 1e-10   # learned η
        for name,elit,min_max in variants:
            for learned, heuristic in [("learned", heu_mat),
                                       ("classic", None)]:   # 1/distance
                c = run_one_instance(dem, dist, heuristic, elit, min_max)
                results.append((idx, name, learned, float(c)))

    # save to csv for plotting/statistics
    with open(f'variants_{size}.csv', 'w', newline='') as f:
        csv.writer(f).writerows([("id","variant","heuristic","cost")] + results)

if __name__ == "__main__":
    for size in [20, 100]:
        main(size)
