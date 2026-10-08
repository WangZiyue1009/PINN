
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from Simulator.Approximator import PreTrainNet, FullNet, ErrorCalculator
from Simulator.cases.aggregation_case55555 import sampled_models_test
from Simulator import PROJECT_ROOT
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import time
from pathlib import Path


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
SEED_BASE = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_MODES = [
              # 'conservative',
              'interpolation',
              # 'extrapolation'
]

# ===================== 运行测试 =====================
all_mode_results = {}

for mode in TEST_MODES:
    print(f"\n{'='*60}")
    print(f"  Testing mode: {mode}")
    print(f"{'='*60}")

    seed_everything(SEED)
    t0 = time.time()

    sampled_models_list = sampled_models_test(
        n_samples=N_SAMPLES,
        train_ev_range=TRAIN_EV_RANGE,
        tcl_count=TCL_COUNT,
        pv_count=PV_COUNT,
        seed_base=SEED_BASE,
        test_mode=mode,
    )

    mode_results = []
    for i, sample in enumerate(sampled_models_list):
        case = sample['case']
        config = sample['config']
        params = sample['params']
        error_calculator = case['errorcalculator']

        print(f"  [{mode}] sample {config['model_id']}: "
              f"EV={params['params_dict']['ev_count']}, "
              f"T={params['params_dict'].get('tcl_temp_ambient_avg', '?')}, "
              f"PV={params['params_dict'].get('pv_energy', '?')}")

        error_calculator.configure(feas_tol=1e-8, opt_tol=1e-8)

        # 加载 PreTrainNet
        pretrainnet = PreTrainNet(case['A_hat'], case['b_hat'], device=device)
        pretrainnet.load_state_dict(
            torch.load(
                f'{PROJECT_ROOT}\\results\\{case["casename"]}\\pretrainnet_weights_20260830_122408.pth',
                map_location=device,
            )
        )
        A_pretrained, b_pretrained = pretrainnet()
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        b_pretrained = b_pretrained[0].detach().cpu().numpy()

        # 加载 FullNet
        fullnet = FullNet(
            dim_theta=case['params']['count'],
            A_init=A_pretrained, b_init=b_pretrained,
            device=device,
        )
        fullnet.load_state_dict(
            torch.load(
                f'{PROJECT_ROOT}\\results\\{case["casename"]}\\fullnet_weights_20260830_164253.pth',
                map_location=device,
            )
        )
        fullnet.to(device)

        # 前向传播
        dataloader = params['dataloader']
        batch_data = next(iter(dataloader))
        if batch_data:
            features = []
            for key in batch_data.keys():
                tensor = batch_data[key]
                features.append(tensor.view(tensor.size(0), -1))
            delta_theta = torch.cat(features, dim=1).to(device)
        else:
            continue

        with torch.no_grad():
            A_pred, b_pred = fullnet(delta_theta)
            A_pred_np = A_pred[0].detach().cpu().numpy()
            b_pred_np = b_pred[0].detach().cpu().numpy()

        error_calculator.update_polytope(A_hat=A_pred_np, b_hat=b_pred_np)

        # error_calculator.calculate(n_cal=5, cal_feas=True, cal_opt=True)
        # feas_error = error_calculator.training_history['feas'][-1]
        # opt_error = error_calculator.training_history['opt'][-1]
        feas_results, opt_results = error_calculator.calculate(
            n_cal=10,
            cal_feas=True,
            cal_opt=True,
        )

        # 保存5个cal方向的原始误差
        feas_errors = [
            r['error'] for r in feas_results
        ]

        opt_errors = [
            r['error'] for r in opt_results
        ]

        # 仍然计算5个方向的平均值，作为该模型的整体误差
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

        # len_his = len(error_calculator.training_history['feas'])
        # print(f"    FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
        #       f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        mode_results.append({
            'model_id': config['model_id'],
            'ev_count': config['ev_count'],
            'error_feas': feas_errors,
            'error_opt': opt_errors,
            'delta_theta': delta_theta[0].detach().cpu().numpy(),
            'A_pred': A_pred_np,
            'b_pred': b_pred_np,
        })

    all_mode_results[mode] = mode_results
    print(f"  [{mode}] done in {time.time() - t0:.1f}s, {len(mode_results)} samples")

# ===================== 汇总数据 =====================
results_dict = {}
for mode in TEST_MODES:
    res = all_mode_results[mode]
    results_dict[mode.capitalize()] = {
        'feas': np.array([r['error_feas'] for r in res]),
        'opt': np.array([r['error_opt'] for r in res]),
    }

# 打印统计摘要
print(f"\n{'='*60}")
print("  Summary")
print(f"{'='*60}")
for mode_name, data in results_dict.items():
    print(f"  {mode_name:15s}  "
          f"Feas: mean={np.mean(data['feas']):.2e}, std={np.std(data['feas']):.2e}  |  "
          f"Opt:  mean={np.mean(data['opt']):.2e}, std={np.std(data['opt']):.2e}")


def save_errors_to_csv(
    all_results,
    results_dict,
    save_dir=None,
    filename_detail="pinn_7_errors_detail",
    filename_summary="pinn_7_errors_summary",
):
    """
    将三种测试场景（conservative / interpolation / extrapolation）的测试误差
    保存为 CSV 文件，与生成的图片保存在同一文件夹下。

    - 明细 CSV：每个样本的可行性误差与最优性误差。
    - 汇总 CSV：每种测试场景的均值、标准差、最小/最大值及样本数。
    使用 utf-8-sig 编码，便于 Excel 正确显示中文。
    """
    if save_dir is None:
        print("[WARNING] save_dir 为空，跳过 CSV 保存。")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 逐样本明细
    detail_path = save_dir / f"{filename_detail}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # writer.writerow(
        #     ["mode", "sample_index", "model_id", "ev_count",
        #      "error_feas", "error_opt"]
        # )
        # for mode in TEST_MODES:
        #     for idx, r in enumerate(all_results[mode]):
        #         writer.writerow([
        #             mode, idx, r["model_id"], r["ev_count"],
        #             r["error_feas"], r["error_opt"],
        #         ])
        writer.writerow(
            [
                "mode",
                "sample_index",
                "model_id",
                "ev_count",
                "cal_direction",
                "error_feas",
                "error_opt",
            ]
        )

        for mode in TEST_MODES:
            for idx, r in enumerate(all_results[mode]):

                feas_errors = r["error_feas"]
                opt_errors = r["error_opt"]

                for cal_idx in range(len(feas_errors)):
                    writer.writerow([
                        mode,
                        idx,
                        r["model_id"],
                        r["ev_count"],
                        cal_idx + 1,
                        feas_errors[cal_idx],
                        opt_errors[cal_idx],
                    ])
    # 2) 汇总统计
    summary_path = save_dir / f"{filename_summary}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["mode", "feas_mean", "feas_std", "feas_min", "feas_max",
             "opt_mean", "opt_std", "opt_min", "opt_max", "n_samples"]
        )
        for mode in TEST_MODES:
            feas = results_dict[mode.capitalize()]["feas"]
            opt = results_dict[mode.capitalize()]["opt"]
            writer.writerow([
                mode,
                np.mean(feas), np.std(feas), np.min(feas), np.max(feas),
                np.mean(opt), np.std(opt), np.min(opt), np.max(opt),
                len(feas),
            ])

    print(f"[INFO] Saved CSV (detail):  {detail_path}")
    print(f"[INFO] Saved CSV (summary): {summary_path}")


# ===================== 生成带抖动的箱线图 =====================
save_dir = r"D:\important\share\results\aggregation\figures_pictures"
# D:\important\share\Simulator\pictures\data
save_errors_to_csv(
    all_results=all_mode_results,
    results_dict=results_dict,
    save_dir=save_dir,
)
#
# plot_error_boxplot(
#     results=results_dict,
#     save_dir=save_dir,
#     filename="test_boxplot_from_errors"
# )

