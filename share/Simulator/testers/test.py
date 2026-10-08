
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from Simulator.Approximator import PreTrainNet, FullNet, ErrorCalculator
from Simulator.cases.aggregation_case import sampled_models_test
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
SEED_BASE = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_MODES = ['conservative', 'interpolation', 'extrapolation']

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

        error_calculator.calculate(n_cal=5, cal_feas=True, cal_opt=True)
        feas_error = error_calculator.training_history['feas'][-1]
        opt_error = error_calculator.training_history['opt'][-1]

        len_his = len(error_calculator.training_history['feas'])
        print(f"    FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        mode_results.append({
            'model_id': config['model_id'],
            'ev_count': config['ev_count'],
            'error_feas': feas_error,
            'error_opt': opt_error,
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

# ===================== 绘图函数：论文风格带抖动箱线图 =====================

def set_chinese_font():
    """
    自动寻找系统中的中文字体，避免中文乱码。
    """
    candidates = [
        "Microsoft YaHei",
        "SimHei",
    ]

    installed = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in candidates:
        if font_name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["mathtext.fontset"] = "dejavusans"
            print(f"[INFO] 使用中文字体: {font_name}")
            return

    print("[WARNING] 未找到合适的中文字体，中文可能显示异常。")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


set_chinese_font()


MODE_LABELS = {
    "Conservative": "保守测试",
    "Interpolation": "分布内测试",
    "Extrapolation": "分布外测试",
}

COLORS = {
    "feas": "#5DA5DA",   # 蓝色
    "opt": "#F17C67",    # 红色
}


def sci_notation_formatter(y, pos):
    """
    将对数坐标刻度格式化为科学计数形式。
    例如：
    1e-4 -> 10^{-4}
    2e-1 -> 2×10^{-1}
    """
    if y <= 0:
        return ""

    exp = int(np.floor(np.log10(y)))
    coeff = y / (10 ** exp)
    coeff_round = round(coeff, 6)

    # 避免部分字体对 Unicode 数学负号不兼容
    exp_str = str(exp).replace("-", r"\mathrm{-}")

    if np.isclose(coeff_round, 1.0):
        return rf"$10^{{{exp_str}}}$"
    else:
        if abs(coeff_round - int(coeff_round)) < 1e-8:
            coeff_str = str(int(round(coeff_round)))
        else:
            coeff_str = f"{coeff_round:g}"
        return rf"${coeff_str}\times10^{{{exp_str}}}$"


def draw_one_axis(
    ax,
    data_list,
    labels,
    color,
    title,
    y_limits=None,
    y_ticks=None,
    y_ticklabels_formatter=None,
):
    positions = np.arange(1, len(data_list) + 1)
    rng = np.random.RandomState(42)

    # 箱线图
    box = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops=dict(linewidth=1.8, color="black"),
        boxprops=dict(linewidth=1.2, edgecolor=color),
        whiskerprops=dict(linewidth=1.1, color=color),
        capprops=dict(linewidth=1.1, color=color),
    )

    # 箱体填充
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.26)

    # 抖动散点
    for pos, data in zip(positions, data_list):
        data = np.asarray(data)
        data = data[data > 0]

        jitter = rng.uniform(-0.08, 0.08, size=len(data))

        ax.scatter(
            pos + jitter,
            data,
            s=22,
            color=color,
            alpha=0.65,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel(title, fontsize=16)

    ax.set_yscale("log")

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if y_ticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))

    if y_ticklabels_formatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(y_ticklabels_formatter))

    # 关闭次刻度，避免对数坐标下网格过密
    ax.yaxis.set_minor_locator(NullLocator())

    ax.set_ylabel("误差（对数坐标）", fontsize=13)

    # 只保留主网格线
    ax.grid(True, axis="y", which="major", alpha=0.22, linestyle="--", linewidth=0.8)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)


def plot_error_boxplot(results, save_dir=None, filename="test_boxplot_from_errors"):
    modes = ["Conservative", "Interpolation", "Extrapolation"]
    labels = [MODE_LABELS[m] for m in modes]

    feas_data = [np.clip(results[m]["feas"], 1e-15, None) for m in modes]
    opt_data = [np.clip(results[m]["opt"], 1e-15, None) for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=600)

    # 左图：可行性误差
    feas_ticks = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    draw_one_axis(
        ax=axes[0],
        data_list=feas_data,
        labels=labels,
        color=COLORS["feas"],
        title="(a) 可行性误差",
        y_limits=(2e-11, 3e-4),
        y_ticks=feas_ticks,
        y_ticklabels_formatter=sci_notation_formatter,
    )

    # 右图：最优性误差
    opt_ticks = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1]

    draw_one_axis(
        ax=axes[1],
        data_list=opt_data,
        labels=labels,
        color=COLORS["opt"],
        title="(b) 最优性误差",
        y_limits=(1e-1, 7e-1),
        y_ticks=opt_ticks,
        y_ticklabels_formatter=sci_notation_formatter,
    )

    legend_handles = [
        Line2D([0], [0], color=COLORS["feas"], lw=7, alpha=0.35, label="可行性误差"),
        Line2D([0], [0], color=COLORS["opt"], lw=7, alpha=0.35, label="最优性误差"),
        Line2D([0], [0], color="black", lw=2, label="中位数"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        png_path = save_dir / f"{filename}.png"
        svg_path = save_dir / f"{filename}.svg"

        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")

        print(f"[INFO] Saved PNG: {png_path}")
        print(f"[INFO] Saved SVG: {svg_path}")

    plt.show()


# ===================== 生成带抖动的箱线图 =====================
save_dir = f"{PROJECT_ROOT}/results/aggregation/figures_pictures"

plot_error_boxplot(
    results=results_dict,
    save_dir=save_dir,
    filename="test_boxplot_from_errors"
)

