import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import rcParams
from scipy.optimize import linprog

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet
from Simulator.cases.aggregation_case import (
    Aggregator,
    bind_training_norm_bounds_to_agg,
    build_profile_reference_tables,
    build_train_test_profile_pools,
    get_train_split_norm_bounds,
)


PRETRAIN_WEIGHTS = r"D:\important\share\results\aggregation\pretrainnet_weights_20260504_161742.pth"
FULLNET_WEIGHTS = r"D:\important\share\results\aggregation\fullnet_weights_20260504_201739.pth"

# =========================
# 电价配置（真实电价 / 随机电价 开关）
# =========================
USE_REAL_PRICE = True  # True: 使用真实电价 CSV 文件; False: 使用均匀分布随机采样电价
REAL_PRICE_PATH = r"D:\important\share\Simulator\pictures\data1\Extreme_High_Price.csv"  # 真实电价 CSV 路径
PRICE_COL = "total_lmp_da"  # 电价列名
NORMALIZE_REAL_PRICE = False  # 是否将真实电价归一化至 [PRICE_LOW, PRICE_HIGH]

PRICE_SEED = 20260506
N_PRICE_CURVES = 10  # 使用的电价场景数量（若 USE_REAL_PRICE=True 且设为 None，则加载 CSV 中的全部场景）
PRICE_LOW = 0
PRICE_HIGH = 1.0

T_DEFAULT = 24
TRAINING_EV_RANGE = (60, 100)

EV_COUNTS_FOR_TEST = (60, 75, 90, 100)

# 严格控制变量实验的参数水平
# 温度：固定同一条日温度曲线形状，仅整体平移平均温度
TEMP_MEANS_FOR_TEST = (-8.9, -7.4, -5.5, -2.7)

# 光伏：固定同一条日光伏曲线形状，仅按比例缩放出力。
PV_ENERGY_LEVELS = (5.3, 5.6, 5.8, 6.0)

FIXED_EV_COUNT = 80
FIXED_TCL_COUNT = 80
FIXED_PV_COUNT = 20

DEFAULT_SEED = 0
DEFAULT_DISCRETE_RATE = 0.0


# Matplotlib 样式设置
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 1000
rcParams["font.size"] = 12
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 12
rcParams["legend.fontsize"] = 12
rcParams["axes.linewidth"] = 0.9
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def style_paper_axis(ax):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.24)
    ax.grid(False, axis="x")

    # 保留左、下、上、右边框
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(True)

    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["top"].set_linewidth(0.9)

    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        labelsize=12,
        width=0.8,
        length=3.5,
        direction="out",
        top=False,
        right=False,
    )

    ax.margins(x=0.08)


def format_y_axis_million(ax):
    """将成本纵轴从 1e6 科学计数法改为 10^6 格式，视觉上更清楚。"""
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / 1e6:.2f}"))
    ax.text(
        -0.08,
        1.02,
        r"$\times 10^6$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
    )


def build_training_norm_bounds(ev_range=TRAINING_EV_RANGE, T=T_DEFAULT):
    """构造训练集归一化边界、历史曲线参考表和训练/测试曲线池。"""
    reference_tables = build_profile_reference_tables(T=T)
    profile_pools = build_train_test_profile_pools(reference_tables)
    norm_bounds = get_train_split_norm_bounds(
        ev_range=ev_range,
        reference_tables=reference_tables,
        profile_pools=profile_pools,
    )
    return norm_bounds, reference_tables, profile_pools


def middle_value(values):
    """从一组离散编号中选取中间位置编号。"""
    values = np.asarray(values, dtype=int)
    values = np.sort(values)
    return int(values[len(values) // 2])


def get_profile_ids(profile_pools, profile_type):
    """从整个候选数据集中读取温度或光伏曲线编号。"""
    if profile_type == "temp":
        keys = ["temp_train_ids", "temp_test_ids"]
    elif profile_type == "pv":
        keys = ["pv_train_ids", "pv_test_ids"]
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")

    ids = []
    for key in keys:
        if key in profile_pools:
            ids.extend(list(profile_pools[key]))

    ids = np.unique(np.asarray(ids, dtype=int))
    ids = np.sort(ids)

    if len(ids) == 0:
        raise ValueError(f"No profile ids found for profile_type={profile_type}")

    return ids


def generate_resources_with_fixed_randomness(agg, ev_count, tcl_count, pv_count, seed):
    """严格控制变量：分别固定 EV、TCL、PV 的随机种子。"""
    agg.rng = np.random.default_rng(seed + 101)
    agg.gen_EV(int(ev_count))

    agg.rng = np.random.default_rng(seed + 202)
    agg.gen_TCL(int(tcl_count))

    agg.rng = np.random.default_rng(seed + 303)
    agg.gen_PV(int(pv_count))


def build_center_profile_indices(profile_pools):
    """选取固定的中间温度曲线和中间光伏曲线，作为控制变量基准曲线。"""
    temp_ids = get_profile_ids(profile_pools, "temp")
    pv_ids = get_profile_ids(profile_pools, "pv")
    return {
        "temp_idx": middle_value(temp_ids),
        "pv_idx": middle_value(pv_ids),
    }


def shift_temperature_profile_to_mean(agg, target_mean):
    """固定温度日曲线形状，仅通过整体平移改变平均温度。"""
    current_mean = float(np.mean(agg.theta_amb))
    agg.theta_amb = np.asarray(agg.theta_amb, dtype=float) + (float(target_mean) - current_mean)


def get_base_pv_energy(agg):
    """计算当前光伏基准曲线的总可用出力。"""
    return float(np.sum(np.asarray(agg.pv_curve, dtype=float)) * agg.deltaT)


def scale_pv_profile(agg, pv_scale):
    """固定光伏日曲线形状，仅按比例缩放出力水平。"""
    agg.pv_curve = np.asarray(agg.pv_curve, dtype=float) * float(pv_scale)


# =========================
# 电价生成 / 加载函数
# =========================

def build_random_price_curves(
    T=T_DEFAULT,
    n_curves=N_PRICE_CURVES,
    seed=PRICE_SEED,
    low=PRICE_LOW,
    high=PRICE_HIGH,
):
    """生成 K 条随机采样电价曲线（每个时段在 [low, high] 上均匀独立采样）。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_curves, T))


def load_real_price_curves(
    csv_path: str,
    T: int = 24,
    col_name: str = "total_lmp_da",
    normalize: bool = False,
    low: float = 0.0,
    high: float = 1.0,
    n_scenarios=None,
    divide_by_1000: bool = True,
):
    """
    从 CSV 中按顺序每 T 行提取一条真实电价曲线（T 默认 24，即每天）。
    价格默认除以 1000（转为千美元/MWh）。
    """
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
        prices_all = prices_all[: n_full * T]
        print(f"[Warning] 总行数 {total_rows} 不是 {T} 的倍数，已截断为 {n_full * T} 行，得到 {n_full} 个场景")

    n_scenes = len(prices_all) // T
    price_curves = prices_all.reshape(n_scenes, T)

    if normalize:
        p_min = price_curves.min()
        p_max = price_curves.max()
        if p_max > p_min:
            price_curves = low + (price_curves - p_min) / (p_max - p_min) * (high - low)

    if n_scenarios is not None and len(price_curves) > n_scenarios:
        price_curves = price_curves[:n_scenarios]

    print(f"[Info] 成功加载 {len(price_curves)} 条真实电价场景（每条 {T} 小时，单位: 千美元/MWh）")
    return price_curves


# =========================
# 场景构造函数
# =========================

def build_case_for_ev_count(
    ev_count,
    norm_bounds,
    profile_indices,
    seed=DEFAULT_SEED,
    T=T_DEFAULT,
    discrete_rate=DEFAULT_DISCRETE_RATE,
    tcl_count=FIXED_TCL_COUNT,
    pv_count=FIXED_PV_COUNT,
):
    agg = Aggregator(
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        temp_idx=profile_indices["temp_idx"],
        pv_idx=profile_indices["pv_idx"],
    )
    bind_training_norm_bounds_to_agg(agg, norm_bounds)
    generate_resources_with_fixed_randomness(
        agg=agg,
        ev_count=ev_count,
        tcl_count=tcl_count,
        pv_count=pv_count,
        seed=seed,
    )
    return agg, agg.case_aggregator(model_type="fullnet")


def build_case_for_temp_mean(
    target_temp_mean,
    norm_bounds,
    fixed_profile_indices,
    seed=DEFAULT_SEED,
    T=T_DEFAULT,
    discrete_rate=DEFAULT_DISCRETE_RATE,
    ev_count=FIXED_EV_COUNT,
    tcl_count=FIXED_TCL_COUNT,
    pv_count=FIXED_PV_COUNT,
):
    agg = Aggregator(
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        temp_idx=fixed_profile_indices["temp_idx"],
        pv_idx=fixed_profile_indices["pv_idx"],
    )
    shift_temperature_profile_to_mean(agg, target_temp_mean)
    bind_training_norm_bounds_to_agg(agg, norm_bounds)
    generate_resources_with_fixed_randomness(
        agg=agg,
        ev_count=ev_count,
        tcl_count=tcl_count,
        pv_count=pv_count,
        seed=seed,
    )
    return agg, agg.case_aggregator(model_type="fullnet")


def build_case_for_pv_scale(
    pv_scale,
    norm_bounds,
    fixed_profile_indices,
    seed=DEFAULT_SEED,
    T=T_DEFAULT,
    discrete_rate=DEFAULT_DISCRETE_RATE,
    ev_count=FIXED_EV_COUNT,
    tcl_count=FIXED_TCL_COUNT,
    pv_count=FIXED_PV_COUNT,
):
    agg = Aggregator(
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        temp_idx=fixed_profile_indices["temp_idx"],
        pv_idx=fixed_profile_indices["pv_idx"],
    )
    scale_pv_profile(agg, pv_scale)
    bind_training_norm_bounds_to_agg(agg, norm_bounds)
    generate_resources_with_fixed_randomness(
        agg=agg,
        ev_count=ev_count,
        tcl_count=tcl_count,
        pv_count=pv_count,
        seed=seed,
    )
    return agg, agg.case_aggregator(model_type="fullnet")


def load_torch_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fullnet(case):
    """加载 PreTrainNet 和 FullNet 权重，并返回 FullNet。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain = PreTrainNet(case["A_hat"], case["b_hat"], device=device).to(device)
    pretrain.load_state_dict(load_torch_weights(PRETRAIN_WEIGHTS, device))
    pretrain.eval()

    with torch.no_grad():
        A_init, b_init = pretrain()

    fullnet = FullNet(
        case["params"]["count"],
        A_init[0].detach().cpu().numpy(),
        b_init[0].detach().cpu().numpy(),
        device=device,
    ).to(device)
    fullnet.load_state_dict(load_torch_weights(FULLNET_WEIGHTS, device))
    fullnet.eval()

    return fullnet, device


def predict_polytope(case, fullnet, device):
    """由 FullNet 输出近似多面体 A_pred, b_pred。"""
    params_dict = case["params"]["params_dict"]
    theta = np.array([params_dict[key][0] for key in params_dict], dtype=np.float32)
    theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        A_pred, b_pred = fullnet(theta_tensor)

    return (
        A_pred[0].detach().cpu().numpy(),
        b_pred[0].detach().cpu().numpy(),
    )


def build_physical_power_scalers(agg):
    """根据底层资源参数构造功率反归一化上下界（EV、TCL 为正，PV 为负）。"""
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

    return p_min, p_max


def denormalize_power(x_norm, p_min, p_max):
    """将归一化功率变量反归一化为物理功率。"""
    return x_norm * (p_max - p_min) + p_min


def optimize_cost_on_polytope(A_pred, b_pred, p_min, p_max, price_curve, dt):
    """在 FullNet 输出的近似多面体上求解最小电价成本。"""
    scale = p_max - p_min
    c_norm = price_curve * scale * dt
    objective_scale = max(1.0, float(np.max(np.abs(c_norm))))
    c_norm_scaled = c_norm / objective_scale

    bounds = [(0.0, 1.0)] * len(c_norm)

    result = linprog(
        c_norm_scaled,
        A_ub=A_pred,
        b_ub=b_pred,
        bounds=bounds,
    )

    if not result.success and result.status == 4:
        result = linprog(
            c_norm_scaled,
            A_ub=A_pred,
            b_ub=b_pred,
            bounds=bounds,
            options={"method": "ipm"},
        )

    if not result.success:
        raise RuntimeError(f"Cost optimization failed: {result.message}")

    x_norm = result.x
    p_opt = denormalize_power(x_norm, p_min, p_max)
    total_cost = float(np.sum(price_curve * p_opt * dt))

    return total_cost, p_opt


def evaluate_cost_over_price_curves(A_pred, b_pred, p_min, p_max, price_curves, dt):
    """对同一个近似运行域，在 K 条电价曲线下分别求解最小成本。"""
    cost_samples = []
    power_samples = []

    for price_curve in price_curves:
        total_cost, p_opt = optimize_cost_on_polytope(
            A_pred=A_pred,
            b_pred=b_pred,
            p_min=p_min,
            p_max=p_max,
            price_curve=price_curve,
            dt=dt,
        )
        cost_samples.append(total_cost)
        power_samples.append(p_opt)

    return np.array(cost_samples, dtype=float), np.mean(power_samples, axis=0)


def summarize_cost_result(label, x_value, cost_samples, optimized_power_mean, price_curves):
    return {
        "label": label,
        "x_value": float(x_value),
        "total_cost": float(np.mean(cost_samples)),
        "mean_cost": float(np.mean(cost_samples)),
        "std_cost": float(np.std(cost_samples, ddof=1)) if len(cost_samples) > 1 else 0.0,
        "cost_samples": cost_samples,
        "optimized_power_mean": optimized_power_mean,
        "price_curves": price_curves.copy(),
    }


# =========================
# 三类关键参数趋势计算
# =========================

def collect_ev_trend_results(
    ev_counts=EV_COUNTS_FOR_TEST,
    training_ev_range=TRAINING_EV_RANGE,
    model_anchor_ev_count=60,
    T=T_DEFAULT,
    seed=DEFAULT_SEED,
    discrete_rate=DEFAULT_DISCRETE_RATE,
    tcl_count=FIXED_TCL_COUNT,
    pv_count=FIXED_PV_COUNT,
    price_curves=None,
):
    norm_bounds, _, profile_pools = build_training_norm_bounds(ev_range=training_ev_range, T=T)
    profile_indices = build_center_profile_indices(profile_pools)

    _, anchor_case = build_case_for_ev_count(
        model_anchor_ev_count,
        norm_bounds,
        profile_indices,
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        tcl_count=tcl_count,
        pv_count=pv_count,
    )
    fullnet, device = load_fullnet(anchor_case)

    results = []
    for ev_count in ev_counts:
        agg, case = build_case_for_ev_count(
            ev_count,
            norm_bounds,
            profile_indices,
            seed=seed,
            T=T,
            discrete_rate=discrete_rate,
            tcl_count=tcl_count,
            pv_count=pv_count,
        )

        p_min, p_max = build_physical_power_scalers(agg)
        A_pred, b_pred = predict_polytope(case, fullnet, device)

        cost_samples, optimized_power_mean = evaluate_cost_over_price_curves(
            A_pred=A_pred,
            b_pred=b_pred,
            p_min=p_min,
            p_max=p_max,
            price_curves=price_curves,
            dt=agg.deltaT,
        )

        results.append(summarize_cost_result(
            label=f"EV={ev_count}",
            x_value=ev_count,
            cost_samples=cost_samples,
            optimized_power_mean=optimized_power_mean,
            price_curves=price_curves,
        ))

    return results


def collect_temp_trend_results(
    training_ev_range=TRAINING_EV_RANGE,
    T=T_DEFAULT,
    seed=DEFAULT_SEED,
    discrete_rate=DEFAULT_DISCRETE_RATE,
    ev_count=FIXED_EV_COUNT,
    tcl_count=FIXED_TCL_COUNT,
    pv_count=FIXED_PV_COUNT,
    n_levels=4,
    price_curves=None,
):
    norm_bounds, _, profile_pools = build_training_norm_bounds(ev_range=training_ev_range, T=T)
    fixed_profile_indices = build_center_profile_indices(profile_pools)

    if n_levels == len(TEMP_MEANS_FOR_TEST):
        temp_means = list(TEMP_MEANS_FOR_TEST)
    else:
        temp_means = np.linspace(min(TEMP_MEANS_FOR_TEST), max(TEMP_MEANS_FOR_TEST), n_levels)

    _, anchor_case = build_case_for_temp_mean(
        temp_means[0],
        norm_bounds,
        fixed_profile_indices,
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        ev_count=ev_count,
        tcl_count=tcl_count,
        pv_count=pv_count,
    )
    fullnet, device = load_fullnet(anchor_case)

    results = []
    for target_temp in temp_means:
        agg, case = build_case_for_temp_mean(
            target_temp,
            norm_bounds,
            fixed_profile_indices,
            seed=seed,
            T=T,
            discrete_rate=discrete_rate,
            ev_count=ev_count,
            tcl_count=tcl_count,
            pv_count=pv_count,
        )

        p_min, p_max = build_physical_power_scalers(agg)
        A_pred, b_pred = predict_polytope(case, fullnet, device)

        cost_samples, optimized_power_mean = evaluate_cost_over_price_curves(
            A_pred=A_pred,
            b_pred=b_pred,
            p_min=p_min,
            p_max=p_max,
            price_curves=price_curves,
            dt=agg.deltaT,
        )

        results.append(summarize_cost_result(
            label=f"Temp={float(target_temp):.1f} ℃",
            x_value=float(target_temp),
            cost_samples=cost_samples,
            optimized_power_mean=optimized_power_mean,
            price_curves=price_curves,
        ))

    return results


def collect_pv_trend_results(
    training_ev_range=TRAINING_EV_RANGE,
    T=T_DEFAULT,
    seed=DEFAULT_SEED,
    discrete_rate=DEFAULT_DISCRETE_RATE,
    ev_count=FIXED_EV_COUNT,
    tcl_count=FIXED_TCL_COUNT,
    pv_count=FIXED_PV_COUNT,
    n_levels=len(PV_ENERGY_LEVELS),
    price_curves=None,
):
    norm_bounds, _, profile_pools = build_training_norm_bounds(ev_range=training_ev_range, T=T)
    fixed_profile_indices = build_center_profile_indices(profile_pools)

    base_agg = Aggregator(
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        temp_idx=fixed_profile_indices["temp_idx"],
        pv_idx=fixed_profile_indices["pv_idx"],
    )
    base_pv_energy = get_base_pv_energy(base_agg)

    if n_levels == len(PV_ENERGY_LEVELS):
        pv_energy_levels = list(PV_ENERGY_LEVELS)
    else:
        pv_energy_levels = np.linspace(min(PV_ENERGY_LEVELS), max(PV_ENERGY_LEVELS), n_levels)
    pv_scales = [float(e) / float(base_pv_energy) for e in pv_energy_levels]

    anchor_agg, anchor_case = build_case_for_pv_scale(
        pv_scales[0],
        norm_bounds,
        fixed_profile_indices,
        seed=seed,
        T=T,
        discrete_rate=discrete_rate,
        ev_count=ev_count,
        tcl_count=tcl_count,
        pv_count=pv_count,
    )
    fullnet, device = load_fullnet(anchor_case)

    results = []
    for pv_energy, pv_scale in zip(pv_energy_levels, pv_scales):
        agg, case = build_case_for_pv_scale(
            pv_scale,
            norm_bounds,
            fixed_profile_indices,
            seed=seed,
            T=T,
            discrete_rate=discrete_rate,
            ev_count=ev_count,
            tcl_count=tcl_count,
            pv_count=pv_count,
        )

        p_min, p_max = build_physical_power_scalers(agg)
        A_pred, b_pred = predict_polytope(case, fullnet, device)

        cost_samples, optimized_power_mean = evaluate_cost_over_price_curves(
            A_pred=A_pred,
            b_pred=b_pred,
            p_min=p_min,
            p_max=p_max,
            price_curves=price_curves,
            dt=agg.deltaT,
        )

        results.append(summarize_cost_result(
            label=f"PV energy={float(pv_energy):.3f}",
            x_value=float(pv_energy),
            cost_samples=cost_samples,
            optimized_power_mean=optimized_power_mean,
            price_curves=price_curves,
        ))

    return results


# =========================
# 绘图函数：箱线图版本
# =========================

def get_cost_samples(results):
    return [np.asarray(item["cost_samples"], dtype=float) for item in results]


def set_nice_ylim_for_box(ax, samples):
    all_values = np.concatenate(samples)
    lower = np.min(all_values)
    upper = np.max(all_values)
    span = upper - lower

    if span <= 0:
        span = max(abs(upper), 1.0) * 0.05

    ax.set_ylim(lower - 0.12 * span, upper + 0.16 * span)


def plot_box_with_mean(
    ax,
    samples,
    positions,
    labels,
    color,
    title,
    xlabel,
    ylabel="最优调度成本 / 元",
):
    box = ax.boxplot(
        samples,
        positions=positions,
        widths=0.45 if len(positions) <= 5 else 0.35,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": color, "linewidth": 1.4},
        boxprops={
            "facecolor": color,
            "alpha": 0.18,
            "edgecolor": color,
            "linewidth": 1.0,
        },
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )

    means = np.array([np.mean(s) for s in samples], dtype=float)

    ax.plot(
        positions,
        means,
        "o",
        color=color,
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label="均值",
        zorder=3,
    )

    ax.set_xlabel(f"{xlabel}\n{title}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    format_y_axis_million(ax)
    set_nice_ylim_for_box(ax, samples)
    style_paper_axis(ax)

    ax.legend(frameon=False, loc="upper right", fontsize=12, handlelength=1.4)
    return box


def plot_ev_trend_lines(results, save_path=None, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        created_fig = True
    else:
        fig = ax.figure

    x_values = np.array([item["x_value"] for item in results], dtype=float)
    samples = get_cost_samples(results)

    positions = np.arange(1, len(results) + 1)
    labels = [f"{int(v)}" for v in x_values]

    plot_box_with_mean(
        ax=ax,
        samples=samples,
        positions=positions,
        labels=labels,
        color="#2F6FB0",
        title="(a) EV 数量影响",
        xlabel="EV 数量",
        ylabel="最优调度成本 / 元",
    )

    if save_path and created_fig:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"EV influence figure saved to: {save_path}")

    if created_fig:
        plt.show()

    return fig, ax


def plot_temp_trend_lines(results, save_path=None, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        created_fig = True
    else:
        fig = ax.figure

    x_values = np.array([item["x_value"] for item in results], dtype=float)
    samples = get_cost_samples(results)

    positions = np.arange(1, len(results) + 1)
    labels = [f"{v:.1f}" for v in x_values]

    plot_box_with_mean(
        ax=ax,
        samples=samples,
        positions=positions,
        labels=labels,
        color="#B03A48",
        title="(b) 平均环境温度影响",
        xlabel="平均环境温度 / ℃",
        ylabel="最优调度成本 / 元",
    )

    if save_path and created_fig:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Temperature influence figure saved to: {save_path}")

    if created_fig:
        plt.show()

    return fig, ax


def plot_pv_trend_lines(results, save_path=None, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        created_fig = True
    else:
        fig = ax.figure

    x_values = np.array([item["x_value"] for item in results], dtype=float)
    samples = get_cost_samples(results)

    positions = np.arange(1, len(results) + 1)
    labels = [f"{v:.3f}" for v in x_values]

    plot_box_with_mean(
        ax=ax,
        samples=samples,
        positions=positions,
        labels=labels,
        color="#C75B1C",
        title="(c) 光伏曲线累计能量影响",
        xlabel="光伏曲线累计出力（标幺值）",
        ylabel="最优调度成本 / 元",
    )

    if save_path and created_fig:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"PV influence figure saved to: {save_path}")

    if created_fig:
        plt.show()

    return fig, ax


# =========================
# CSV 导出函数
# =========================

def _flatten_trend_results(results, parameter_type, parameter_name):
    rows = []
    for item in results:
        x_value = float(item["x_value"])
        label = item["label"]
        cost_samples = np.asarray(item["cost_samples"], dtype=float).ravel()
        for curve_idx, cost in enumerate(cost_samples):
            rows.append({
                "parameter_type": parameter_type,
                "parameter_name": parameter_name,
                "x_value": x_value,
                "label": label,
                "price_curve_idx": int(curve_idx),
                "cost": float(cost),
            })
    return rows


def _summarize_trend_results(results, parameter_type, parameter_name):
    summaries = []
    for item in results:
        cost_samples = np.asarray(item["cost_samples"], dtype=float).ravel()
        summaries.append({
            "parameter_type": parameter_type,
            "parameter_name": parameter_name,
            "x_value": float(item["x_value"]),
            "label": item["label"],
            "n_price_curves": int(len(cost_samples)),
            "mean_cost": float(np.mean(cost_samples)),
            "std_cost": float(np.std(cost_samples, ddof=1)) if len(cost_samples) > 1 else 0.0,
            "median_cost": float(np.median(cost_samples)),
            "q1_cost": float(np.quantile(cost_samples, 0.25)),
            "q3_cost": float(np.quantile(cost_samples, 0.75)),
            "min_cost": float(np.min(cost_samples)),
            "max_cost": float(np.max(cost_samples)),
        })
    return summaries


def save_parameter_trend_csvs(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    trend_specs = [
        (results_dict["ev_results"], "ev_count", "EV 数量"),
        (results_dict["temp_results"], "temp_mean", "平均环境温度 / ℃"),
        (results_dict["pv_results"], "pv_energy", "光伏曲线累计出力（标幺值）"),
    ]

    detail_rows = []
    summary_rows = []
    for results, parameter_type, parameter_name in trend_specs:
        detail_rows.extend(_flatten_trend_results(results, parameter_type, parameter_name))
        summary_rows.extend(_summarize_trend_results(results, parameter_type, parameter_name))

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = os.path.join(save_dir, "parameter_trend_cost_details333.csv")
    summary_path = os.path.join(save_dir, "parameter_trend_cost_summary333.csv")
    price_curves_path = os.path.join(save_dir, "parameter_trend_price_curves333.csv")

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    price_curves = np.asarray(results_dict["price_curves"], dtype=float)
    price_df = pd.DataFrame(
        price_curves,
        columns=[f"t{idx + 1}" for idx in range(price_curves.shape[1])],
    )
    price_df.insert(0, "price_curve_idx", np.arange(price_curves.shape[0]))
    price_df.to_csv(price_curves_path, index=False, encoding="utf-8-sig")

    print(f"参数影响成本明细已保存到: {detail_path}")
    print(f"参数影响成本汇总已保存到: {summary_path}")
    print(f"实验电价曲线已保存到: {price_curves_path}")

    return detail_path, summary_path, price_curves_path


# =========================
# 总图绘制与主流程
# =========================

def plot_parameter_trends(
    save_path=None,
    T=T_DEFAULT,
    n_price_curves=N_PRICE_CURVES,
    price_seed=PRICE_SEED,
    csv_dir=None,
):
    """
    绘制 1x3 关键参数影响总图：箱线图版本。

    支持选择真实电价 CSV 或随机均匀分布采样电价。
    """
    # 根据开关条件判定加载真实电价还是生成随机电价
    if USE_REAL_PRICE:
        price_curves = load_real_price_curves(
            csv_path=REAL_PRICE_PATH,
            T=T,
            col_name=PRICE_COL,
            normalize=NORMALIZE_REAL_PRICE,
            low=PRICE_LOW,
            high=PRICE_HIGH,
            n_scenarios=n_price_curves,
        )
    else:
        price_curves = build_random_price_curves(
            T=T,
            n_curves=n_price_curves,
            seed=price_seed,
            low=PRICE_LOW,
            high=PRICE_HIGH,
        )

    print(f"使用 {len(price_curves)} 条电价曲线计算最优调度成本分布。")
    print(f"第一条曲线平均价格约为 {np.mean(price_curves[0]):.4f}。")

    ev_results = collect_ev_trend_results(
        ev_counts=EV_COUNTS_FOR_TEST,
        T=T,
        price_curves=price_curves,
    )

    temp_results = collect_temp_trend_results(
        T=T,
        price_curves=price_curves,
    )

    pv_results = collect_pv_trend_results(
        T=T,
        price_curves=price_curves,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 14.0),
        constrained_layout=False,
    )

    plot_ev_trend_lines(ev_results, ax=axes[0])
    plot_temp_trend_lines(temp_results, ax=axes[1])
    plot_pv_trend_lines(pv_results, ax=axes[2])

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        bottom=0.04,
        top=0.97,
        hspace=0.35,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"参数影响箱线图已保存到: {save_path}")

    plt.show()

    results = {
        "ev_results": ev_results,
        "temp_results": temp_results,
        "pv_results": pv_results,
        "price_curves": price_curves,
    }

    if csv_dir:
        save_parameter_trend_csvs(results, csv_dir)

    return results


if __name__ == "__main__":
    figures_dir = os.path.join(PROJECT_ROOT, "results", "aggregation", "figures_pictures")
    save_path = os.path.join(figures_dir, "parameter_cost_trends.png")

    plot_parameter_trends(
        save_path=save_path,
        T=T_DEFAULT,
        n_price_curves=N_PRICE_CURVES,
        price_seed=PRICE_SEED,
        csv_dir=figures_dir,
    )