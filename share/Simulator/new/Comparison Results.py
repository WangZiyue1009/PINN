import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch
from scipy.optimize import linprog

if __package__ in (None, ""):
    workspace_root = Path(__file__).resolve().parents[2]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet
from Simulator.cases.aggregation_case import (
    build_case_from_physical_config,
    sampled_models_test,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CONFIG = {
    "n_samples": 40,
    "ev_range": (70, 90),
    "train_ev_range": (60, 100),
    "tcl_count": 80,
    "pv_count": 20,
    "bl_count": 0,
    "wind_count": 0,
    "seed_base": 0,
    "discrete_rate": 0.0,
    "T": 24,
    "n_cal": 10,
}

PLOT_CONFIG = {
    "use_real_price": True,
    "real_price_path": r"D:\important\share\data1\Random_Price.csv",
    "price_col": "total_lmp_da",
    "normalize_real_price": False,
    "price_seed": 20260506,
    "num_price_scenarios": 10,
    "price_sample_limit": None,
    "price_low": 0.0,
    "price_high": 1.0,
}

PRETRAIN_WEIGHT_NAME = "pretrainnet_weights_20260504_161742.pth"
FULLNET_WEIGHT_NAME = "fullnet_weights_20260504_201739.pth"
SUPERVISED_WEIGHT_NAME = "supervised_weights_20260902_202606.pth"

PRICE_NORM_EPS = 1e-9
DECOMP_ABS_TOL = 1e-6
DECOMP_REL_TOL = 1e-6

NORM_VAR_BOUNDS: Optional[Tuple[float, float]] = (0.0, 1.0)
CHECK_NORM_SOLUTION = True

METHOD_ORDER = ["fullnet", "supervised", "cube", "power_energy"]
METHOD_META = {
    "fullnet": {
        "label": "Learned Polytope Approx. (FullNet)",
        "short_label": "Learned Polytope",
        "color": "#2F6FB0",
    },
    "supervised": {
        "label": "Supervised Regression (FullNet)",
        "short_label": "Supervised",
        "color": "#8E44AD",
    },
    "cube": {
        "label": "Inner Approx. (Cube)",
        "short_label": "Inner Approx.",
        "color": "#C75B1C",
    },
    "power_energy": {
        "label": "Outer Approx. (Power-Energy)",
        "short_label": "Outer Approx.",
        "color": "#2F9D77",
    },
}


def method_label(method_id: str, short: bool = False):
    key = "short_label" if short else "label"
    return METHOD_META[method_id][key]


def resolve_weight_path(case_name: str, filename: str):
    candidate_paths = [
        os.path.join(PROJECT_ROOT, "results", case_name, filename),
        os.path.join(os.path.dirname(PROJECT_ROOT), "results", case_name, filename),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Weight file not found: {filename}. Checked: {candidate_paths}"
    )


def extract_delta_theta(batch_data, target_device):
    features = []
    for key in batch_data.keys():
        tensor = batch_data[key]
        flattened = tensor.view(tensor.size(0), -1)
        features.append(flattened)
    return torch.cat(features, dim=1).to(target_device)

def load_fullnet_from_case(case):
    case_name = case["casename"]
    pretrain_weight = resolve_weight_path(case_name, PRETRAIN_WEIGHT_NAME)
    fullnet_weight = resolve_weight_path(case_name, FULLNET_WEIGHT_NAME)

    pretrainnet = PreTrainNet(case["A_hat"], case["b_hat"], device=device)
    pretrainnet.load_state_dict(torch.load(pretrain_weight, map_location=device))
    A_pretrained, b_pretrained = pretrainnet()
    A_pretrained = A_pretrained[0].detach().cpu().numpy()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()

    fullnet = FullNet(
        dim_theta=case["params"]["count"],
        A_init=A_pretrained,
        b_init=b_pretrained,
        device=device,
    )
    fullnet.load_state_dict(torch.load(fullnet_weight, map_location=device))
    fullnet.to(device)
    fullnet.eval()

    return pretrain_weight, fullnet_weight, fullnet


def load_supervised_from_case(case):
    case_name = case["casename"]
    pretrain_weight = resolve_weight_path(case_name, "pretrainnet_weights_20260830_191651.pth")
    try:
        supervised_weight = resolve_weight_path(case_name, SUPERVISED_WEIGHT_NAME)
    except FileNotFoundError:
        print(
            f"[warn] 未找到监督权重 ({SUPERVISED_WEIGHT_NAME})，跳过 supervised 方法。"
        )
        return None

    pretrainnet = PreTrainNet(case["A_hat"], case["b_hat"], device=device)
    pretrainnet.load_state_dict(torch.load(pretrain_weight, map_location=device))
    A_pretrained, b_pretrained = pretrainnet()
    A_pretrained = A_pretrained[0].detach().cpu().numpy()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()

    supervised = FullNet(
        dim_theta=case["params"]["count"],
        A_init=A_pretrained,
        b_init=b_pretrained,
        device=device,
    )
    supervised.load_state_dict(torch.load(supervised_weight, map_location=device))
    supervised.to(device)
    supervised.eval()
    return supervised


def evaluate_fullnet(case, fullnet, n_repeats: int = 5):
    batch_data = next(iter(case["params"]["dataloader"]))

    delta_theta = extract_delta_theta(batch_data, device)
    with torch.no_grad():
        _ = fullnet(delta_theta)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_repeats):
        start_t = time.perf_counter()
        delta_theta = extract_delta_theta(batch_data, device)
        with torch.no_grad():
            A_pred, b_pred = fullnet(delta_theta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start_t) * 1000.0)

    elapsed_ms = float(np.median(times_ms))
    A_pred_np = A_pred[0].detach().cpu().numpy()
    b_pred_np = b_pred[0].detach().cpu().numpy()

    return A_pred_np, b_pred_np, elapsed_ms

def evaluate_cube(agg):
    (A_cube, b_cube), comp_time = agg.build_cube_approximation()
    return A_cube, b_cube, comp_time * 1000.0


def evaluate_power_energy(agg):
    (A_pe, b_pe), comp_time = agg.build_power_energy_approximation(
        include_base_load=False,
        solver_name="gurobi",
        eps=1e-8,
    )
    return A_pe, b_pe, comp_time * 1000.0


def build_agg_for_sample(sample):

    config = sample["config"]

    agg, _ = build_case_from_physical_config(
        config={
            "ev_count": int(config["ev_count"]),
            "tcl_count": int(config["tcl_count"]),
            "pv_count": int(config["pv_count"]),
        },
        norm_bounds=sample["norm_bounds"],
        seed=TEST_CONFIG["seed_base"] + int(config["model_id"]),
        discrete_rate=TEST_CONFIG["discrete_rate"],
        T=TEST_CONFIG["T"],
        temp_idx=int(config["tcl_idx"]),
        pv_idx=int(config["pv_idx"]),
    )

    return agg


def summarize_method(runtime_rows):
    df = pd.DataFrame(runtime_rows)
    summaries = []
    for method_id, group_df in df.groupby("method_id", sort=False):
        runtime_ms = group_df["runtime_ms"].to_numpy(dtype=float)
        summaries.append({
            "method_id": method_id,
            "method": method_label(method_id),
            "n_samples": int(len(group_df)),
            "runtime_mean_ms": float(np.mean(runtime_ms)),
            "runtime_median_ms": float(np.median(runtime_ms)),
            "runtime_max_ms": float(np.max(runtime_ms)),
        })
    return pd.DataFrame(summaries)


def build_random_price_curves(
        T: int,
        n_scenarios: int,
        seed: int,
        low: float = -1.0,
        high: float = 1.0,
):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_scenarios, T))


def load_real_price_curves(
        csv_path: str,
        T: int = 24,
        col_name: str = "total_lmp_da",
        normalize: bool = False,
        low: float = 0.0,
        high: float = 1.0,
        n_scenarios: Optional[int] = None,
        divide_by_1000: bool = True,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到电价 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if col_name not in df.columns:
        raise ValueError(f"列 '{col_name}' 不存在于 {csv_path} 中。可用列: {list(df.columns)}")

    prices_all = df[col_name].values.astype(float)
    if divide_by_1000:
        prices_all = prices_all / 1000.0

    total_rows = len(prices_all)
    if total_rows % T != 0:
        n_full = total_rows // T
        prices_all = prices_all[:n_full * T]
        print(f"[Warning] 总行数 {total_rows} 已截断为 {n_full * T} 行")

    n_scenes = len(prices_all) // T
    price_curves = prices_all.reshape(n_scenes, T)

    if normalize:
        p_min = price_curves.min()
        p_max = price_curves.max()
        if p_max > p_min:
            price_curves = low + (price_curves - p_min) / (p_max - p_min) * (high - low)

    if n_scenarios is not None and len(price_curves) > n_scenarios:
        price_curves = price_curves[:n_scenarios]

    print(f"[Info] 成功加载 {len(price_curves)} 条电价场景")
    return price_curves


def build_physical_power_scalers(agg):
    p_max = np.zeros(agg.T, dtype=float)
    p_min = np.zeros(agg.T, dtype=float)

    for pv_data in agg.PV_list:
        p_min -= pv_data["max_power_curve"]
    for tcl_data in agg.TCL_list:
        p_max += tcl_data["Qmax"] / tcl_data["COP"]
    for ev_data in agg.EV_list:
        ta_idx = int(np.floor(ev_data["ta"] / agg.deltaT))
        td_idx = int(np.ceil(ev_data["td"] / agg.deltaT))
        p_max[ta_idx:td_idx] += ev_data["P_chg"]

    scale = p_max - p_min
    if np.any(scale <= 0):
        bad_idx = np.where(scale <= 0)[0].tolist()
        raise ValueError(f"Non-positive physical power scale at time indices: {bad_idx}")

    return p_min, p_max


def get_linprog_bounds(dim: int):
    if NORM_VAR_BOUNDS is None:
        return [(None, None)] * dim
    lb, ub = NORM_VAR_BOUNDS
    return [(lb, ub)] * dim


def validate_normalized_solution(x_norm, method_id: str, sample_id: int, price_id: int):
    if not CHECK_NORM_SOLUTION or NORM_VAR_BOUNDS is None:
        return
    lb, ub = NORM_VAR_BOUNDS
    tol = 1e-6
    if np.any(x_norm < lb - tol) or np.any(x_norm > ub + tol):
        print(
            f"Warning: {method_id} solution outside normalized bounds "
            f"for sample={sample_id}, price={price_id}."
        )


def optimize_cost_on_polytope(
        A_hat,
        b_hat,
        price_curve,
        p_min,
        p_max,
        delta_t,
        method_id: str,
        sample_id: int,
        price_id: int,
):
    scale = p_max - p_min
    c_norm = price_curve * scale * delta_t
    result = linprog(
        c_norm,
        A_ub=A_hat,
        b_ub=b_hat,
        bounds=get_linprog_bounds(len(price_curve)),
        method="highs",
    )
    if not result.success:
        return None, None, None
    x_norm = result.x
    validate_normalized_solution(x_norm, method_id, sample_id, price_id)
    p_phys = x_norm * scale + p_min
    total_cost = float(np.sum(price_curve * p_phys * delta_t))
    return total_cost, p_phys, x_norm


def optimize_cost_on_true_model(case, price_curve, p_min, p_max, delta_t):
    model = case["model"].clone()
    if hasattr(model, "ray_obj"):
        model.del_component(model.ray_obj)
    if hasattr(model, "cost_obj"):
        model.del_component(model.cost_obj)

    scale = p_max - p_min
    model.cost_obj = pyo.Objective(
        expr=sum(
            price_curve[t] * (model.var_proj[t] * scale[t] + p_min[t]) * delta_t
            for t in range(len(price_curve))
        ),
        sense=pyo.minimize,
    )
    solver = pyo.SolverFactory("gurobi")
    res = solver.solve(model, tee=False)
    if res.solver.termination_condition not in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ]:
        raise RuntimeError(f"True-model cost optimization failed: {res.solver.termination_condition}")

    x_norm = np.array([pyo.value(model.var_proj[t]) for t in model.T], dtype=float)
    p_phys = x_norm * scale + p_min
    total_cost = float(np.sum(price_curve * p_phys * delta_t))
    return total_cost, p_phys, x_norm


def compute_price_guided_decomposition_error(
        case,
        target_p_phys: np.ndarray,
        delta_t: float,
):
    model = case["model"].clone()

    for obj_name in ["ray_obj", "cost_obj", "decomp_obj"]:
        if hasattr(model, obj_name):
            model.del_component(getattr(model, obj_name))

    for comp_name in ["u_plus", "u_minus", "track_constraint"]:
        if hasattr(model, comp_name):
            model.del_component(getattr(model, comp_name))

    target_p_phys = np.asarray(target_p_phys, dtype=float)
    T = len(target_p_phys)

    p_min, p_max = build_physical_power_scalers_from_model_case(case)
    scale = p_max - p_min

    model.u_plus = pyo.Var(range(T), within=pyo.NonNegativeReals)
    model.u_minus = pyo.Var(range(T), within=pyo.NonNegativeReals)

    def track_rule(m, t):
        return (
                m.var_proj[t] * float(scale[t])
                + float(p_min[t])
                + m.u_plus[t]
                - m.u_minus[t]
                == float(target_p_phys[t])
        )

    model.track_constraint = pyo.Constraint(range(T), rule=track_rule)
    model.decomp_obj = pyo.Objective(
        expr=sum((model.u_plus[t] + model.u_minus[t]) * delta_t for t in range(T)),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("gurobi")
    res = solver.solve(model, tee=False)
    if res.solver.termination_condition not in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ]:
        raise RuntimeError(
            f"Decomposition check failed: {res.solver.termination_condition}"
        )

    abs_error = float(pyo.value(model.decomp_obj))
    denom = max(float(np.sum(np.abs(target_p_phys) * delta_t)), PRICE_NORM_EPS)
    rel_error = float(abs_error / denom)
    return abs_error, rel_error


def build_physical_power_scalers_from_model_case(case):
    if isinstance(case, dict) and "_p_min_for_decomp" in case and "_p_max_for_decomp" in case:
        return case["_p_min_for_decomp"], case["_p_max_for_decomp"]
    raise ValueError("Physical power scalers were not attached to case.")


def direction_cost_scale(price_curve, p_min, p_max, delta_t):
    return float(np.sum(np.abs(price_curve) * (p_max - p_min) * delta_t))


def evaluate_price_guided_costs_for_sample(
        sample_id: int,
        case,
        agg,
        method_polytopes: Dict[str, Tuple[np.ndarray, np.ndarray]],
        price_curves: np.ndarray,
):
    p_min, p_max = build_physical_power_scalers(agg)
    case["_p_min_for_decomp"] = p_min
    case["_p_max_for_decomp"] = p_max
    delta_t = agg.deltaT
    rows = []

    for price_id, price_curve in enumerate(price_curves):
        ref_scale = max(direction_cost_scale(price_curve, p_min, p_max, delta_t), PRICE_NORM_EPS)
        true_cost, true_p, true_x = optimize_cost_on_true_model(
            case=case,
            price_curve=price_curve,
            p_min=p_min,
            p_max=p_max,
            delta_t=delta_t,
        )

        for method_id in METHOD_ORDER:
            if method_id not in method_polytopes:
                continue
            A_hat, b_hat = method_polytopes[method_id]
            method_cost, method_p, method_x = optimize_cost_on_polytope(
                A_hat=A_hat,
                b_hat=b_hat,
                price_curve=price_curve,
                p_min=p_min,
                p_max=p_max,
                delta_t=delta_t,
                method_id=method_id,
                sample_id=sample_id,
                price_id=price_id,
            )
            if method_cost is None:
                rows.append({
                    "sample_id": sample_id,
                    "price_id": price_id,
                    "method_id": method_id,
                    "method": method_label(method_id),
                    "true_cost": true_cost,
                    "method_cost": np.nan,
                    "cost_diff": np.nan,
                    "normalized_cost_diff": np.nan,
                    "abs_decomp_error": np.nan,
                    "rel_decomp_error": np.nan,
                    "is_decomp_failed": True,
                })
                continue
            abs_decomp_error, rel_decomp_error = compute_price_guided_decomposition_error(
                case=case,
                target_p_phys=method_p,
                delta_t=delta_t,
            )

            cost_diff = float(method_cost - true_cost)
            norm_cost_diff = float(cost_diff / ref_scale)

            rows.append(
                {
                    "sample_id": sample_id,
                    "price_id": price_id,
                    "method_id": method_id,
                    "method": method_label(method_id),
                    "true_cost": true_cost,
                    "method_cost": method_cost,
                    "cost_diff": cost_diff,
                    "normalized_cost_diff": norm_cost_diff,
                    "abs_decomp_error": abs_decomp_error,
                    "rel_decomp_error": rel_decomp_error,
                    "is_decomp_failed": bool(
                        (abs_decomp_error > DECOMP_ABS_TOL) and (rel_decomp_error > DECOMP_REL_TOL)
                    ),
                }
            )

    return rows


def summarize_price_costs(price_detail_df):
    summaries = []
    for method_id, group_df in price_detail_df.groupby("method_id", sort=False):
        cost_diff = group_df["cost_diff"].to_numpy(dtype=float)
        norm_diff = group_df["normalized_cost_diff"].to_numpy(dtype=float)
        abs_decomp = group_df["abs_decomp_error"].to_numpy(dtype=float)
        rel_decomp = group_df["rel_decomp_error"].to_numpy(dtype=float)
        summaries.append(
            {
                "method_id": method_id,
                "method": method_label(method_id),
                "n_cases": int(len(group_df)),
                "cost_diff_mean": float(np.mean(cost_diff)),
                "cost_diff_median": float(np.median(cost_diff)),
                "cost_diff_p10": float(np.quantile(cost_diff, 0.10)),
                "cost_diff_p90": float(np.quantile(cost_diff, 0.90)),
                "normalized_cost_diff_mean": float(np.mean(norm_diff)),
                "normalized_cost_diff_median": float(np.median(norm_diff)),
                "normalized_cost_diff_p10": float(np.quantile(norm_diff, 0.10)),
                "normalized_cost_diff_p90": float(np.quantile(norm_diff, 0.90)),
                "abs_decomp_error_mean": float(np.mean(abs_decomp)),
                "abs_decomp_error_median": float(np.median(abs_decomp)),
                "rel_decomp_error_mean": float(np.mean(rel_decomp)),
                "rel_decomp_error_median": float(np.median(rel_decomp)),
                "rel_decomp_error_p90": float(np.quantile(rel_decomp, 0.90)),
                "decomp_failure_rate": float(np.mean(group_df["is_decomp_failed"].to_numpy(dtype=bool))),
            }
        )
    return pd.DataFrame(summaries)


def save_price_curves(price_curves, save_path):
    df = pd.DataFrame(price_curves, columns=[f"t{idx + 1}" for idx in range(price_curves.shape[1])])
    df.insert(0, "price_id", np.arange(price_curves.shape[0]))
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def main():

    samples = sampled_models_test(
        n_samples=TEST_CONFIG["n_samples"],
        train_ev_range=TEST_CONFIG["train_ev_range"],
        tcl_count=TEST_CONFIG["tcl_count"],
        pv_count=TEST_CONFIG["pv_count"],
        seed_base=TEST_CONFIG["seed_base"],
        discrete_rate=TEST_CONFIG["discrete_rate"],
        T=TEST_CONFIG["T"],
        test_mode=TEST_CONFIG.get("test_mode", "interpolation"),
    )

    if not samples:
        raise RuntimeError("sampled_models_test() 未返回任何测试样本。")

    pretrain_weight, fullnet_weight, fullnet = load_fullnet_from_case(samples[0]["case"])
    supervised_net = load_supervised_from_case(samples[0]["case"])

    save_dir = r"D:\important\share\results\aggregation\figures_pictures"
    os.makedirs(save_dir, exist_ok=True)

    runtime_rows = []
    price_rows = []

    if PLOT_CONFIG.get("use_real_price", False):
        price_curves = load_real_price_curves(
            csv_path=PLOT_CONFIG["real_price_path"],
            T=TEST_CONFIG["T"],
            col_name=PLOT_CONFIG.get("price_col", "total_lmp_da"),
            normalize=PLOT_CONFIG.get("normalize_real_price", False),
            low=PLOT_CONFIG["price_low"],
            high=PLOT_CONFIG["price_high"],
            n_scenarios=PLOT_CONFIG["num_price_scenarios"],
        )
    else:
        price_curves = build_random_price_curves(
            T=TEST_CONFIG["T"],
            n_scenarios=PLOT_CONFIG["num_price_scenarios"],
            seed=PLOT_CONFIG["price_seed"],
            low=PLOT_CONFIG["price_low"],
            high=PLOT_CONFIG["price_high"],
        )

    price_sample_limit = PLOT_CONFIG.get("price_sample_limit", None)
    if price_sample_limit is None:
        price_eval_sample_ids = set(range(len(samples)))
    else:
        price_eval_sample_ids = set(range(min(int(price_sample_limit), len(samples))))

    for sample_id, sample in enumerate(samples):
        case = sample["case"]
        config = sample["config"]

        agg = build_agg_for_sample(sample)

        print(
            f"\nSample {sample_id + 1}/{len(samples)} "
            f"({config.get('model_id', sample_id)}): "
            f"EV={config['ev_count']}, "
            f"T={config.get('tcl_temp_ambient_avg', '?')}, "
            f"PV={config.get('pv_energy', '?')}"
        )

        A_fullnet, b_fullnet, fullnet_time_ms = evaluate_fullnet(case, fullnet)
        runtime_rows.append({
            "sample_id": sample_id,
            "method_id": "fullnet",
            "method": method_label("fullnet"),
            "runtime_ms": fullnet_time_ms,
        })

        A_cube, b_cube, cube_time_ms = evaluate_cube(agg)
        runtime_rows.append({
            "sample_id": sample_id,
            "method_id": "cube",
            "method": method_label("cube"),
            "runtime_ms": cube_time_ms,
        })

        A_pe, b_pe, pe_time_ms = evaluate_power_energy(agg)
        runtime_rows.append({
            "sample_id": sample_id,
            "method_id": "power_energy",
            "method": method_label("power_energy"),
            "runtime_ms": pe_time_ms,
        })

        A_sup = b_sup = None
        if supervised_net is not None:
            A_sup, b_sup, sup_time_ms = evaluate_fullnet(case, supervised_net)
            runtime_rows.append({
                "sample_id": sample_id,
                "method_id": "supervised",
                "method": method_label("supervised"),
                "runtime_ms": sup_time_ms,
            })

        if sample_id in price_eval_sample_ids:
            method_polytopes = {
                "fullnet": (A_fullnet, b_fullnet),
                "cube": (A_cube, b_cube),
                "power_energy": (A_pe, b_pe),
            }
            if supervised_net is not None:
                method_polytopes["supervised"] = (A_sup, b_sup)

            sample_price_rows = evaluate_price_guided_costs_for_sample(
                sample_id=sample_id,
                case=case,
                agg=agg,
                method_polytopes=method_polytopes,
                price_curves=price_curves,
            )
            price_rows.extend(sample_price_rows)

        print(f"Finished sample {sample_id + 1}/{len(samples)}")

    runtime_summary_df = summarize_method(runtime_rows)
    price_detail_df = pd.DataFrame(price_rows)
    if not price_detail_df.empty:
        price_summary_df = summarize_price_costs(price_detail_df)
    else:
        price_summary_df = pd.DataFrame()

    price_curves_path = os.path.join(save_dir, "price_guided_random_price_curves_sl8.csv")
    price_detail_path = os.path.join(save_dir, "price_guided_cost_details_sl8.csv")
    price_summary_path = os.path.join(save_dir, "price_guided_cost_summary_sl8.csv")
    runtime_summary_path = os.path.join(save_dir, "runtime_summary_sl8.csv")

    save_price_curves(price_curves, price_curves_path)
    runtime_summary_df[["method_id", "method", "runtime_mean_ms", "runtime_max_ms"]].to_csv(
        runtime_summary_path, index=False, encoding="utf-8-sig"
    )

    if not price_detail_df.empty:
        price_detail_df.to_csv(price_detail_path, index=False, encoding="utf-8-sig")
        price_summary_df.to_csv(price_summary_path, index=False, encoding="utf-8-sig")

    print("pretrain weight:", pretrain_weight)
    print("fullnet weight:", fullnet_weight)
    print("\n=== Runtime Summary ===")
    print(runtime_summary_df.to_string(index=False))

    if not price_summary_df.empty:
        print("\n=== Price-Guided Cost Summary ===")
        print(price_summary_df.to_string(index=False))

    print("\nprice curves saved:", price_curves_path)
    print("runtime summary saved:", runtime_summary_path)
    if not price_detail_df.empty:
        print("price-guided cost details saved:", price_detail_path)
        print("price-guided cost summary saved:", price_summary_path)


if __name__ == "__main__":
    main()