from __future__ import annotations

import itertools, os, pathlib, sys, time, warnings, logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from net import Net
from aco import ACO as _BaseACO
from utils import load_test_dataset, gen_pyg_data

# ───────────────────────────── logging ────────────────────────────────────
try:
    from rich.logging import RichHandler
    _handler = RichHandler(markup=True, rich_tracebacks=True)
except ImportError:
    _handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_handler],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# ───────────────────────────── ACO wrapper ───────────────────────────
class SafeACO(_BaseACO):
    """ACO variant that guards against NaNs/Infs at sampling time."""
    @torch.no_grad()
    def pick_move(self, prev, visit_mask, capacity_mask, require_prob):
        pher = torch.nan_to_num(self.pheromone[prev], nan=1e-12, posinf=1e4, neginf=1e-12)
        heu  = torch.nan_to_num(self.heuristic[prev], nan=1e-12, posinf=1e4, neginf=1e-12)

        dist = (pher.clamp_min(1e-12) ** self.alpha) * (heu.clamp_min(1e-12) ** self.beta)
        dist = dist * visit_mask * capacity_mask
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

        row_sum = dist.sum(dim=1, keepdim=True)
        needs_fallback = (row_sum == 0).squeeze(-1)
        if needs_fallback.any():
            dist[needs_fallback, 0] = 1.0

        probs = dist
        actions = torch.distributions.Categorical(probs).sample()
        logp = torch.distributions.Categorical(probs).log_prob(actions) if require_prob else None
        return actions, logp

# ─────────────────────────── experiment configuration ────────────────────────────

DEVICE = "cuda:0"
if not torch.cuda.is_available():
    logger.error("CUDA requested but not available. Aborting.")
    sys.exit(1)

EPS        = 1e-10
ACO_ITERS  = 3
N_ANTS     = 5
SIZES      = [100]
GRIDS      = {"alpha": [0.5,1,2,4], "beta": [0.5,2,4], "decay": [0.3,0.5,0.7,0.9]}
OUT_DIR    = pathlib.Path("results"); OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────── helper functions ─────────────────────────────

@torch.no_grad()
def eval_instance(demands: torch.Tensor, distances: torch.Tensor, model: Net | None,
                  alpha: float, beta: float, decay: float) -> float:
    demands, distances = demands.to(DEVICE), distances.to(DEVICE)
    heu_mat = None
    if model is not None:
        pyg = gen_pyg_data(demands, distances, DEVICE).to(DEVICE)
        heu_vec = model(pyg) + EPS
        heu_mat = heu_vec.reshape((demands.size(0),) * 2)

    aco = SafeACO(distances=distances,
                  demand=demands,
                  n_ants=N_ANTS,
                  alpha=alpha,
                  beta=beta,
                  decay=decay,
                  heuristic=heu_mat,
                  device=DEVICE)
    aco.run(ACO_ITERS)
    return float(aco.lowest_cost)


def bench_one_size(n_nodes: int) -> pd.DataFrame:
    logger.info(f"[{n_nodes}] dataset & model loading…")
    dataset = load_test_dataset(n_nodes, DEVICE)
    net = Net().to(DEVICE)
    net.load_state_dict(torch.load(f"./pretrained/cvrp/cvrp{n_nodes}.pt", map_location=DEVICE))
    net.eval()

    rows: List[Dict[str, float]] = []
    combos = list(itertools.product(*GRIDS.values()))
    logger.info(f"[{n_nodes}] {len(combos)} combos × {len(dataset)} instances")

    for alpha, beta, decay in tqdm(combos, desc=f"{n_nodes}-node grid"):
        t0, deep_costs, van_costs = time.time(), [], []
        for dem, dist in dataset:
            deep_costs.append(eval_instance(dem, dist, net,   alpha, beta, decay))
            van_costs.append(eval_instance(dem, dist, None,  alpha, beta, decay))

        rows.append({
            "size": n_nodes, "alpha": alpha, "beta": beta, "decay": decay,
            "deepaco_mean": float(np.mean(deep_costs)),
            "vanilla_mean": float(np.mean(van_costs)),
        })
        logger.info("✓ α=%s β=%s δ=%s in %.1fs", alpha, beta, decay, time.time()-t0)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"sensitivity_{n_nodes}.csv", index=False)
    return df

# ──────────────────────────── main ────────────────────────────────────────

def main():
    logger.info(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    big = pd.concat([bench_one_size(sz) for sz in SIZES], ignore_index=True)
    big.to_csv(OUT_DIR / "sensitivity_all.csv", index=False)
    logger.info("All sizes done")

if __name__ == "__main__":
    main()
