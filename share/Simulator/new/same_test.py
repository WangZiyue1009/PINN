import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from Simulator.Approximator import PreTrainNet, FullNet, ErrorCalculator
from Simulator.cases.aggregation_case import sampled_models_test, sampled_models_same_theta_test, inspect_xi_lists
from Simulator import PROJECT_ROOT
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv
import random
import time
from pathlib import Path
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ===================== 配置 =====================
SEED = 0
N_SAMPLES = 40
TCL_COUNT = 80
PV_COUNT = 20
TRAIN_EV_RANGE = (60, 100)
SEED_BASE = 666

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything(SEED)

all_mode_results = {}

print("\n" + "=" * 60)
print(" Same theta - Different xi robustness test")
print("=" * 60)
t0 = time.time()

# ==============================
# Same theta interpolation
# ==============================

same_theta_interpolation_samples = sampled_models_same_theta_test(
    n_samples=40,
    test_mode='interpolation',
    # 固定 theta
    ev_count=80,
    tcl_count=TCL_COUNT,
    pv_count=PV_COUNT,
    seed_base=5000,
    train_ev_range=TRAIN_EV_RANGE
)
print(
    f"Interpolation same theta samples: "
    f"{len(same_theta_interpolation_samples)}"
)


inspect_xi_lists(
    same_theta_interpolation_samples,
    save_dir=
    f"{PROJECT_ROOT}/results/aggregation/figures_pictures/interpolation"
)

init_sample = same_theta_interpolation_samples[0]  # 仅用于加载网络
case = init_sample['case']
params = init_sample['params']

pretrainnet = PreTrainNet(
    case['A_hat'],
    case['b_hat'],
    device=device
)

pretrainnet.load_state_dict(
    torch.load(
        f'{PROJECT_ROOT}\\results\\{case["casename"]}'
        '\\pretrainnet_weights_20260504_161742.pth',
        map_location=device,
    )
)
pretrainnet.eval()
A_pretrained, b_pretrained = pretrainnet()
A_pretrained = (A_pretrained[0].detach().cpu().numpy())
b_pretrained = (b_pretrained[0].detach().cpu().numpy())

fullnet = FullNet(
    dim_theta=case['params']['count'],
    A_init=A_pretrained,
    b_init=b_pretrained,
    device=device,
)

fullnet.load_state_dict(
    torch.load(
        f'{PROJECT_ROOT}\\results\\{case["casename"]}'
        '\\fullnet_weights_20260504_201739.pth',
        map_location=device,
    )
)

fullnet.to(device)
fullnet.eval()

def get_theta_from_sample(sample):
    """从单个 sample 中提取 dataloader 的第一个 batch，拼接成 theta 向量"""
    params = sample['params']
    dataloader = params['dataloader']
    batch_data = next(iter(dataloader))
    features = []
    for key in batch_data.keys():
        tensor = batch_data[key]
        features.append(
            tensor.view(tensor.size(0), -1)
        )
    theta = torch.cat(features, dim=1).to(device)
    return theta


# --- interpolation ---
theta_inter = get_theta_from_sample(same_theta_interpolation_samples[0])
with torch.no_grad():
    A_inter, b_inter = fullnet(theta_inter)
A_inter_np = A_inter[0].detach().cpu().numpy()
b_inter_np = b_inter[0].detach().cpu().numpy()
print("Interpolation prediction finished.")


def evaluate_same_theta_samples(
        samples,
        A_pred_np,
        b_pred_np,
        mode_name):
    """
    对一组 samples（固定 θ，不同 ξ）计算误差。
    """
    mode_results = []

    for i, sample in enumerate(samples):
        case = sample['case']
        config = sample['config']
        error_calculator = case['errorcalculator']

        print(
            f"{mode_name} "
            f"Testing xi {i}: "
            f"EV={config['ev_count']}, "
            f"PV={config['pv_energy']}"
            f"TCL={config['tcl_temp_ambient_avg']}"
        )

        error_calculator.configure(
            feas_tol=1e-8,
            opt_tol=1e-8
        )

        error_calculator.update_polytope(
            A_hat=A_pred_np,
            b_hat=b_pred_np
        )

        feas_results, opt_results = error_calculator.calculate(
            n_cal=10,
            cal_feas=True,
            cal_opt=True
        )

        feas_errors = [r['error'] for r in feas_results]
        opt_errors = [r['error'] for r in opt_results]

        # 5 个方向平均值，仅用于打印
        feas_error_mean = np.mean(feas_errors)
        opt_error_mean = np.mean(opt_errors)

        print(
            f"    FeasErr mean={feas_error_mean:.2e}, "
            f"OptErr mean={opt_error_mean:.2e}"
        )

        print(
            f"    Feas directions: "
            f"{[f'{x:.2e}' for x in feas_errors]}"
        )

        print(
            f"    Opt directions:  "
            f"{[f'{x:.2e}' for x in opt_errors]}"
        )

        mode_results.append({
            "model_id": i,
            "xi_id": sample['xi_id'],
            "ev_count": config['ev_count'],
            "pv_energy": config['pv_energy'],
            "error_feas": feas_errors,
            "error_opt": opt_errors,
        })

    print(f"{mode_name} finished: {len(mode_results)} samples")
    return mode_results



interpolation_results = evaluate_same_theta_samples(
    samples=same_theta_interpolation_samples,
    A_pred_np=A_inter_np,
    b_pred_np=b_inter_np,
    mode_name="Interpolation"
)


print(f"Total evaluation time: {time.time() - t0:.1f}s")



results_dict = {
    "Interpolation": {
        "feas": np.array([r["error_feas"] for r in interpolation_results]),
        "opt": np.array([r["error_opt"] for r in interpolation_results]),
    },

}

print("\n========== Summary ==========")

for mode in results_dict:
    print(f"\n{mode}:")
    for measure in ["feas", "opt"]:
        arr = results_dict[mode][measure]
        print(
            f"  {measure:6s}  "
            f"mean={np.mean(arr):.3e}  "
            f"std={np.std(arr):.3e}  "
            f"min={np.min(arr):.3e}  "
            f"max={np.max(arr):.3e}  "
            f"n={arr.size}"
        )


def save_errors_to_csv(mode_results, mode_name, save_dir):


    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    detail_path = save_dir / f"same_theta_{mode_name}_detail.csv"

    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)

        writer.writerow([
            "xi_index",
            "xi_id",
            "ev_count",
            "pv_energy",
            "cal_index",
            "error_feas",
            "error_opt",
        ])

        for idx, r in enumerate(mode_results):

            feas_errors = r["error_feas"]
            opt_errors = r["error_opt"]

            # 5 个 cal 方向逐行保存
            for cal_idx, (feas_error, opt_error) in enumerate(
                zip(feas_errors, opt_errors), start=1
            ):

                writer.writerow([
                    idx,
                    r.get("xi_id", ""),
                    r.get("ev_count", ""),
                    r.get("pv_energy", ""),
                    cal_idx,
                    feas_error,
                    opt_error,
                ])

    print(f"[INFO] Saved CSV: {detail_path}")

save_dir = f"{PROJECT_ROOT}/results/aggregation/figures_pictures/same_theta"

save_errors_to_csv(
    mode_results=interpolation_results,
    mode_name="interpolation",
    save_dir=save_dir,
)