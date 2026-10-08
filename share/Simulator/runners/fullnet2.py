#自适应λ
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator3 import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.aggregation_case import Aggregator, sampled_models_train, get_average_config
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# #pretrainnet — 用采样场景的平均参数构造基准 case
# model_type = 'pretrainnet'
# cache_path = f'{PROJECT_ROOT}/results/aggregation/sampled_cache.json'
sampled_models_list = sampled_models_train(
    n_samples=150,
    ev_range=(60, 100),
    tcl_count=80,
    pv_count=20,
    # bl_count=100,
    # wind_count=10,
    seed_base=0,
    discrete_rate=0.0,
    T=24,
    cache_path=None,
)
# avg_cfg = get_average_config(sampled_models_list)
# print(f"平均参数: EV={avg_cfg['ev_count']}, 温度={avg_cfg['tcl_temp']:.2f}, "
#       f"PV={avg_cfg['pv_energy']:.2f}")
#
# agg = Aggregator(seed=0, discrete_rate=0.0,
#                  temp_idx=avg_cfg['temp_idx'],
#                  pv_idx=avg_cfg['pv_idx'],)
# agg.gen_EV(avg_cfg['ev_count'])
# agg.gen_TCL(80)
# agg.gen_PV(20)
# agg.gen_BL(100)
# agg.gen_Wind(10)
# case = agg.case_aggregator(model_type=model_type)


#fullnet

case = sampled_models_list[0]['case'] #采样函数中已写,自带fullnet
model_type = 'fullnet'

if model_type=='pretrainnet':
    n_train = case.get('n_train',2)
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:
    n_train = case.get('n_train',2)
    model = PreTrainNet(case['A_hat'], case['b_hat'])
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights_20260504_161742.pth',map_location=device))
    #从gpu保存了预训练权重的时候，用下面这个
    # model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=torch.device('cpu')))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)
import time
training_start = time.time()
# #预训练，不分阶段
# trainer.configure(**case['trainer_configure'])
# trainer.configure(
#     #feas_tol = 1e-1,
#                   rate_opt_feas = 1.0,
#                   scheduler={ "type": "StepLR",
#                               "step_size":50,
#                               "gamma": 0.95
#                             },
#                   lr =1e-3,
# )
# trainer.initialize()
# trainer.train(n_train = 5000 , params_data = case['params'])


n_stage1 = 6000     # 阶段1步数
n_stage2 = 10000        # 阶段2步数
n_stage3 = 20000        # 阶段3步数
total_global_steps = n_stage1 + n_stage2 + n_stage3
# 通用的 lambda 渐变配置参数
lambda_config = {
    "use_exp_lambda": True,
    "lambda_start": 1.0,
    "lambda_end": 1e-4,
    "total_global_steps": total_global_steps,
}
#fullnet
trainer.configure(**case['trainer_configure'])
trainer.configure(
                  **lambda_config,
                  lr = 1e-3,#5e-5
                  scheduler={"type": "StepLR", "step_size": 200, "gamma": 0.95},
)
trainer.initialize()
trainer.train(n_train = n_train ,train_model_list=sampled_models_list)
# #2000轮
trainer.configure(**case['trainer_configure'])
trainer.configure(
                  **lambda_config,
                  lr =8e-4,
                  scheduler={ "type": "StepLR", "step_size": 200, "gamma": 0.95},
)
trainer.initialize()
trainer.train(n_train = 10000 ,train_model_list=sampled_models_list)

trainer.configure(**case['trainer_configure'])
trainer.configure(
                  **lambda_config,
                  lr=3e-4,
                  scheduler={ "type": "StepLR", "step_size": 300, "gamma": 0.95}
)
trainer.initialize()
trainer.train(n_train = 20000,train_model_list=sampled_models_list)

training_end = time.time()
print('总耗时',training_end-training_start)
torch.save(model.state_dict(), case['result_path'])

# 保存训练历史
history_dir = os.path.join(os.path.dirname(case['result_path']), 'history')
trainer.save_history(
    save_dir=history_dir,
    phase_name=model_type,
    config_info={
        'n_train': n_train,
        'model_type': model_type,
    }
)
