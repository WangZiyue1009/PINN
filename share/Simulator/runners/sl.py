import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import torch
import numpy as np
import time

from Simulator.Approximator import PreTrainNet, BiasNet, FullNet, compute_loss, Trainer
from Simulator.cases.aggregation_case6 import Aggregator, sampled_models_train, get_average_config
from Simulator import PROJECT_ROOT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cache_path = f'{PROJECT_ROOT}/results/aggregation/sampled_cache.json'
sampled_models_list = sampled_models_train(
    n_samples=150,
    ev_range=(60, 100),
    tcl_count=80,
    pv_count=20,
    seed_base=0,
    discrete_rate=0.0,
    T=24,
    cache_path=None,
)

figure_folder = r'D:\important\share\results\aggregation\history\Ab'
os.makedirs(figure_folder, exist_ok=True)  # 确保目录存在

training_start = time.time()



i=78
case = sampled_models_list[i]['case']
model_type = 'pretrainnet'
n_train =1500  # 这里可根据需要改成真实训练迭代数

# 初始化专属当前模型的网络和训练器[cite: 3]
model = PreTrainNet(case['A_hat'], case['b_hat'], device=device).to(device)
trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

# 配置训练参数[cite: 3]
trainer.configure(**case['trainer_configure'])
trainer.configure(
    rate_opt_feas=0.01,
    scheduler={"type": "StepLR", "step_size": 50, "gamma": 0.95},
    lr=3e-3,
)

# 启动训练[cite: 3]
trainer.initialize()
trainer.train(n_train=n_train, params_data=case['params'])

trainer.model.eval()

with torch.no_grad():
    A_final, b_final = trainer.model()
A_final = A_final[0].detach().cpu().numpy()
b_final = b_final[0].detach().cpu().numpy()

saved_params = {}
if 'params' in case and 'params_dict' in case['params']:
    for param_name, param_info in case['params']['params_dict'].items():
        if isinstance(param_info, dict):
            val = param_info.get('initial_value')
        else:
            val = param_info  # 如果已经是 ndarray，直接赋值

            # 2. 转换 numpy 数组为 python 列表以便保存为 json
        if isinstance(val, np.ndarray):
            saved_params[param_name] = val.tolist()
        else:
            saved_params[param_name] = val

# 打包数据
data_to_save = {
    'A_hat': A_final.tolist(),
    'b_hat': b_final.tolist(),
    'parameters': saved_params
}

# 保存为独立的 JSON 文件 (按索引编号命名)
save_path = os.path.join(figure_folder, f'polytope_model_{i}.json')
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=2)

print(f"模型 {i} 的最终多面体及参数已保存至: {save_path}\n")

training_end = time.time()
print(f'模型全部训练并保存完毕，总耗时: {training_end - training_start:.2f} 秒')

