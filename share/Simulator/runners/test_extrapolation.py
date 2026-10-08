# 外推测试
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Simulator.Approximator import PreTrainNet, FullNet, ErrorCalculator
from Simulator import PROJECT_ROOT
import torch
import numpy as np
import pandas as pd
import time
from Simulator.cases.aggregation_case3 import sampled_models_test_extrapolation_no_retrain

sampled_models_list = sampled_models_test_extrapolation_no_retrain(
    n_samples=40,
    train_ev_range= (60, 100),
    test_ev_range = (30, 140),
    tcl_count=80,
    pv_count=20,
    seed_base=666,
    discrete_rate=0.0,
    T=24
)
#n_samples: int = 20,
    # train_ev_range: tuple = (60, 100),
    # test_ev_range: tuple = (30, 140),
    # tcl_count: int = 150,
    # pv_count: int = 30,
    # seed_base: int = 1000,
    # discrete_rate: float = 0.0,
    # T: int = 24,
print(f"\n采样完成，共获取 {len(sampled_models_list)} 个外推模型\n")
for i, model in enumerate(sampled_models_list):
    cfg = model['config']
    nf = cfg['norm_features']
    def _tag(v):
        return "OOD" if abs(v) > 1.0 else "in-range"
    print(f"模型{i + 1}: "
          f"EV数量={cfg['ev_count']} (norm={nf['ev_norm']:+.3f}, {_tag(nf['ev_norm'])}), "
          f"TCL平均温度={cfg['tcl_temp_ambient_avg']:.2f}°C (norm={nf['temp_norm']:+.3f}, {_tag(nf['temp_norm'])}), "
          f"PV发电量={cfg['pv_energy']:.2f} (norm={nf['pv_norm']:+.3f}, {_tag(nf['pv_norm'])})")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 外推(no_retrain)只跑单一模式，保留 results_dict 结构以便复用绘图代码
TEST_MODES = ['extrapolation']
all_mode_results = {}

# ===================== 运行测试 =====================
for mode in TEST_MODES:
    print(f"\n{'='*60}")
    print(f"  Testing mode: {mode}")
    print(f"{'='*60}")

    t0 = time.time()
    mode_results = []

    for i, sample in enumerate(sampled_models_list):
        case = sample['case']
        config = sample['config']
        params = sample['params']
        error_calculator = case['errorcalculator']

        error_calculator.configure(cal=10,feas_tol=1e-8, opt_tol=1e-8)

        # 加载 PreTrainNet
        pretrainnet = PreTrainNet(case['A_hat'], case['b_hat'], device=device)
        pretrainnet.load_state_dict(
            torch.load(
                f'{PROJECT_ROOT}\\results\\{case["casename"]}\\pretrainnet_weights_20260504_161742.pth',
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
                f'{PROJECT_ROOT}\\results\\{case["casename"]}\\fullnet_weights_20260504_201739.pth',
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
# 每个模型有10个cal方向
results_dict = {}

for mode in TEST_MODES:
    res = all_mode_results[mode]

    # shape = (n_models, 10)
    feas = np.array([r['error_feas'] for r in res], dtype=float)
    opt = np.array([r['error_opt'] for r in res], dtype=float)

    results_dict[mode.capitalize()] = {
        'feas': feas,
        'opt': opt,
    }


# ===================== 打印统计摘要 =====================
print(f"\n{'='*60}")
print("  Summary")
print(f"{'='*60}")

for mode_name, data in results_dict.items():

    feas = data['feas']
    opt = data['opt']

    # 展平成所有模型×所有cal方向
    feas_all = feas.reshape(-1)
    opt_all = opt.reshape(-1)

    print(
        f"  {mode_name:15s}  "
        f"Feas: mean={np.mean(feas_all):.2e}, "
        f"std={np.std(feas_all):.2e}  |  "
        f"Opt: mean={np.mean(opt_all):.2e}, "
        f"std={np.std(opt_all):.2e}"
    )


# ===================== 保存结果到 CSV =====================

# 用 model_id 建立 config 查找表
config_lookup = {
    s['config']['model_id']: s['config']
    for s in sampled_models_list
}


def _ood_tag(v):
    return "OOD" if abs(v) > 1.0 else "in-range"


# ============================================================
# 1. 明细 CSV
# ============================================================
# 每个模型 × 10个cal方向 = 10行
detail_rows = []

for mode in TEST_MODES:

    for r in all_mode_results[mode]:

        cfg = config_lookup[r['model_id']]
        nf = cfg['norm_features']

        # 10个cal方向
        feas_errors = np.asarray(
            r['error_feas'],
            dtype=float
        )

        opt_errors = np.asarray(
            r['error_opt'],
            dtype=float
        )

        # 检查
        if len(feas_errors) != 10 or len(opt_errors) != 10:
            raise ValueError(
                f"模型 {r['model_id']} 的 cal 方向数量不是10："
                f"feas={len(feas_errors)}, "
                f"opt={len(opt_errors)}"
            )

        # ========================================================
        # 每一个cal方向单独保存一行
        # ========================================================
        for cal_idx in range(10):

            detail_rows.append({

                'mode': mode,

                'model_id': r['model_id'],

                'ev_count': r['ev_count'],

                'tcl_count': cfg['tcl_count'],

                'pv_count': cfg['pv_count'],

                'tcl_temp_ambient_avg':
                    cfg['tcl_temp_ambient_avg'],

                'pv_energy':
                    cfg['pv_energy'],

                'ev_norm':
                    nf['ev_norm'],

                'temp_norm':
                    nf['temp_norm'],

                'pv_norm':
                    nf['pv_norm'],

                'ev_tag':
                    _ood_tag(nf['ev_norm']),

                'temp_tag':
                    _ood_tag(nf['temp_norm']),

                'pv_tag':
                    _ood_tag(nf['pv_norm']),

                # ★ 当前是第几个cal方向
                'cal_direction':
                    cal_idx + 1,

                # ★ 当前方向的误差
                'error_feas':
                    feas_errors[cal_idx],

                'error_opt':
                    opt_errors[cal_idx],
            })


detail_df = pd.DataFrame(detail_rows)


# ============================================================
# 2. 汇总统计表
# ============================================================
summary_rows = []

for mode_name, data in results_dict.items():

    feas = data['feas']
    opt = data['opt']

    # 展平：
    # 40模型 × 10方向
    # ↓
    # 400个误差
    feas_all = feas.reshape(-1)
    opt_all = opt.reshape(-1)

    summary_rows.append({

        'mode': mode_name,

        # 模型数量
        'n_samples': feas.shape[0],

        # 每个模型的cal方向数
        'n_cal_directions': feas.shape[1],

        # 总方向数
        'n_total_cal': len(feas_all),

        # ----------------------------------------------------
        # 所有模型、所有cal方向的统计
        # ----------------------------------------------------
        'feas_mean':
            float(np.mean(feas_all)),

        'feas_std':
            float(np.std(feas_all)),

        'feas_min':
            float(np.min(feas_all)),

        'feas_max':
            float(np.max(feas_all)),

        'opt_mean':
            float(np.mean(opt_all)),

        'opt_std':
            float(np.std(opt_all)),

        'opt_min':
            float(np.min(opt_all)),

        'opt_max':
            float(np.max(opt_all)),
    })


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# 3. 保存
# ============================================================
save_dir = os.path.join(
    PROJECT_ROOT,
    'results',
    'aggregation',
    'figures_pictures'
)

os.makedirs(save_dir, exist_ok=True)


detail_path = os.path.join(
    save_dir,
    'extrapolation_test_detail_10cal.csv'
)

summary_path = os.path.join(
    save_dir,
    'extrapolation_test_summary_10cal.csv'
)


detail_df.to_csv(
    detail_path,
    index=False,
    encoding='utf-8-sig'
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# 4. 输出保存结果信息
# ============================================================
print(f"\n{'='*60}")
print("  结果已保存到 CSV")
print(f"{'='*60}")

print(f"  明细 -> {detail_path}")
print(f"  汇总 -> {summary_path}")

print(f"\n  模型数量: {len(sampled_models_list)}")
print(f"  每个模型 cal 方向数: 10")
print(f"  明细数据行数: {len(detail_df)}")