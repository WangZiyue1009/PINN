
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from Simulator.Approximator import PreTrainNet, FullNet, ErrorCalculator
from Simulator.cases.aggregation_case2 import sampled_models_test

from Simulator import PROJECT_ROOT
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import time
from pathlib import Path
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import csv

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
TCL_COUNT = 8
PV_COUNT = 2
TRAIN_EV_RANGE = (6, 10)
SEED_BASE = 1000
test_ev_range=(3,14)
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
        test_ev_range=test_ev_range,
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
                f'{PROJECT_ROOT}\\results\\{case["casename"]}\\pretrainnet_weights_20260813_025617.pth',
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
                f'{PROJECT_ROOT}\\results\\{case["casename"]}\\fullnet_weights_20260902_215206.pth',
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

        # error_calculator.calculate(n_cal=10, cal_feas=True, cal_opt=True)
        #
        #
        # feas_error = error_calculator.training_history['feas'][-1]
        # opt_error = error_calculator.training_history['opt'][-1]
        #
        # len_his = len(error_calculator.training_history['feas'])
        # print(f"    FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
        #       f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
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

import csv
# ===================== 数据保存 =====================
SAVE_DIR = Path(r"D:\important\share\Simulator\pictures\data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

csv_path = SAVE_DIR / "ess_error_results4.csv"

# 每个模型 × 10个cal方向 = 10行
fieldnames = [
    "mode",
    "model_id",
    "ev_count",
    "cal_direction",
    "error_feas",
    "error_opt",
]

with open(
    csv_path,
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for mode_name, mode_res in all_mode_results.items():

        for item in mode_res:

            # 取出10个cal方向的误差
            feas_errors = np.asarray(
                item["error_feas"],
                dtype=float
            )

            opt_errors = np.asarray(
                item["error_opt"],
                dtype=float
            )

            # 检查是否为10个方向
            if len(feas_errors) != 10 or len(opt_errors) != 10:
                raise ValueError(
                    f"模型 {item['model_id']} 的cal方向数量不是10："
                    f"feas={len(feas_errors)}, "
                    f"opt={len(opt_errors)}"
                )

            # ==================================================
            # 每个cal方向单独保存一行
            # ==================================================
            for cal_idx in range(10):

                writer.writerow({
                    "mode": mode_name,
                    "model_id": item["model_id"],
                    "ev_count": item["ev_count"],
                    "cal_direction": cal_idx + 1,
                    "error_feas": feas_errors[cal_idx],
                    "error_opt": opt_errors[cal_idx],
                })

print(f"\n[数据保存成功] 误差数据已保存至: {csv_path}")

# 统计实际保存行数
total_models = sum(
    len(mode_res)
    for mode_res in all_mode_results.values()
)

print(f"[数据统计] 模型数量: {total_models}")
print(f"[数据统计] 每个模型cal方向: 10")
print(f"[数据统计] CSV数据行数: {total_models * 10}")