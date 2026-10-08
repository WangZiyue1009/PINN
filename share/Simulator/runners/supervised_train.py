# 监督基线
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
from datetime import datetime
import numpy as np
import torch

import json
from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- 配置 --------
TRAIN_CONFIG = {
    "n_samples": 149,
    "ev_range": (60, 100),
    "tcl_count": 80,
    "pv_count": 20,
    "seed_base": 0,
    "discrete_rate": 0.0,
    "T": 24,
    "n_hidden": 64,            # 与 PINN 的 FullNet 一致
    "lr": 5e-3,
    "epochs": 36000,
    "lambda_b": 1.0,           # b 项相对 A 的权重
    "batch_size": 32,          # mini-batch；None=全批量
     "scheduler": {"type": "StepLR", "step_size": 2000, "gamma": 0.95}
}

PRETRAIN_WEIGHT_NAME = "pretrainnet_weights_20260830_191651.pth"

def build_train_samples_from_saved_jsons(cfg, json_folder):


    samples = []

    for i in range(cfg["n_samples"]):

        json_path = os.path.join(
            json_folder,
            f"polytope_model_{i}.json"
        )

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"未找到标签文件：{json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        A_star = torch.tensor(
            data["A_hat"],
            dtype=torch.float32,
            device=device
        )

        b_star = torch.tensor(
            data["b_hat"],
            dtype=torch.float32,
            device=device
        )
        params = data["parameters"]

        ev = float(params["ev_count"][0])

        tcl = float(
            params["tcl_temp_ambient_avg"][0]
        )

        pv = float(params["pv_energy"][0])

        theta = torch.tensor(
            [ev, tcl, pv],
            dtype=torch.float32,
            device=device
        )

        samples.append({
            "theta": theta,
            "A_star": A_star,
            "b_star": b_star,
        })

    print(
        f"成功从 JSON 读取 {len(samples)} 个监督学习样本"
    )

    return samples
def init_fullnet_like_pinn(json_folder):

    json_path = os.path.join(
        json_folder,
        "polytope_model_0.json"
    )

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"未找到初始化 JSON：{json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    A_init = np.asarray(
        data["A_hat"],
        dtype=np.float32
    )

    b_init = np.asarray(
        data["b_hat"],
        dtype=np.float32
    )

    pretrain_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "aggregation",
        PRETRAIN_WEIGHT_NAME
    )

    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"未找到预训练权重：{pretrain_path}"
        )

    pretrainnet = PreTrainNet(
        A_init=A_init,
        b_init=b_init,
        device=device
    ).to(device)

    state_dict = torch.load(
        pretrain_path,
        map_location=device
    )

    pretrainnet.load_state_dict(state_dict)

    pretrainnet.eval()

    with torch.no_grad():
        A_pretrained, b_pretrained = pretrainnet()

    A_pretrained = (
        A_pretrained[0]
        .detach()
        .cpu()
        .numpy()
    )

    b_pretrained = (
        b_pretrained[0]
        .detach()
        .cpu()
        .numpy()
    )

    # print("\n========== PreTrainNet 初始化 ==========")
    # print(f"A_pretrained shape = {A_pretrained.shape}")
    # print(f"b_pretrained shape = {b_pretrained.shape}")
    # print(f"预训练权重 = {pretrain_path}")
    # print("=========================================\n")

    model = FullNet(
        dim_theta=3,
        A_init=A_pretrained,
        b_init=b_pretrained,
        n_hidden=TRAIN_CONFIG["n_hidden"],
        device=device,
    ).to(device)

    return model


def build_scheduler(optimizer, cfg):

    sched_cfg = cfg.get("scheduler", None)
    if sched_cfg is None:
        return None
    stype = sched_cfg.get("type", "CosineAnnealingLR")
    if stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=float(sched_cfg.get("eta_min", 1e-5))
        )
    if stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg["step_size"]),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
    if stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 500)),
        )
    raise ValueError(f"不支持的 scheduler 类型: {stype}")


def train():
    cfg = TRAIN_CONFIG
    torch.manual_seed(cfg["seed_base"])
    np.random.seed(cfg["seed_base"])

    json_folder = r"D:\important\share\results\aggregation\history\Ab"

    samples = build_train_samples_from_saved_jsons(
        cfg,
        json_folder
    )

    # 堆成批量张量
    thetas = torch.stack(
        [s["theta"] for s in samples],
        dim=0
    )        # (N, dim_theta)
    A_stars = torch.stack([s["A_star"] for s in samples], dim=0)       # (N, nrows, T)
    b_stars = torch.stack([s["b_star"] for s in samples], dim=0)       # (N, nrows)
    N = thetas.shape[0]
    # print("\n========== JSON 数据检查 ==========")
    # print("thetas shape :", thetas.shape)
    # print("A_stars shape:", A_stars.shape)
    # print("b_stars shape:", b_stars.shape)
    #
    # print("\n第 0 个 theta:")
    # print(thetas[0])
    #
    # print("\n第 0 个 A:")
    # print(A_stars[0])
    #
    # print("\n第 0 个 b:")
    # print(b_stars[0])
    #
    # print("====================================\n")

    A_scale = A_stars.std().clamp_min(1e-6)
    b_scale = b_stars.std().clamp_min(1e-6)

    model = init_fullnet_like_pinn(json_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    scheduler = build_scheduler(optimizer, cfg)

    loss_history = []
    bs = cfg["batch_size"] or N
    rng = np.random.default_rng(cfg["seed_base"])

    start = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(N)
        epoch_loss = 0.0
        n_batch = 0
        # 增加 loss_A 和 loss_b 的 epoch 累加器
        epoch_loss_A = 0.0
        epoch_loss_b = 0.0

        for start_i in range(0, N, bs):
            idx = perm[start_i:start_i + bs]
            theta = thetas[idx]
            A_target = A_stars[idx]
            b_target = b_stars[idx]

            A_pred, b_pred = model(theta)

            loss_A = torch.mean(((A_pred - A_target) / A_scale) ** 2 )
            loss_b = torch.mean(((b_pred - b_target) / b_scale) ** 2 )
            loss = loss_A + cfg["lambda_b"] * loss_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_A += loss_A.item()
            epoch_loss_b += loss_b.item()


            epoch_loss += loss.item()
            n_batch += 1

        epoch_loss /= max(n_batch, 1)
        loss_history.append(epoch_loss)
        epoch_loss_A /= max(n_batch, 1)
        epoch_loss_b /= max(n_batch, 1)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        if epoch % 100 == 0 or epoch == cfg["epochs"] - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            # print(f"Epoch {epoch:4d}: loss={epoch_loss:.6e}, lr={cur_lr:.2e}")
            print(f"Epoch {epoch:4d}: Total loss={epoch_loss:.6e} "
                  f"| loss_A={epoch_loss_A:.6e} "
                  f"| loss_b={epoch_loss_b:.6e} "
                  f"| lr={cur_lr:.2e}")
    elapsed = time.time() - start

    # 保存权重
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(PROJECT_ROOT, "results", "aggregation")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"supervised_weights_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)

    # 保存 loss 历史
    history_path = os.path.join(save_dir, "figures_pictures", f"supervised_{timestamp}.csv")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    np.savetxt(history_path, np.array(loss_history), delimiter=',')

    print(f"\n训练完成，耗时 {elapsed:.1f}s")
    print(f"权重已保存: {save_path}")
    print(f"loss 历史: {history_path}")
    print(f"\n>> 评估时请把 Comparison Results.py 里的 SUPERVISED_WEIGHT_NAME "
          f"设为: supervised_weights_{timestamp}.pth")


if __name__ == "__main__":
    train()
