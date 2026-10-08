#加入ESS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.aggregation_case2 import Aggregator, sampled_models_train, get_average_config
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#pretrainnet — 用采样场景的平均参数构造基准 case
# model_type = 'pretrainnet'

sampled_models_list = sampled_models_train(
    n_samples=30,
    ev_range=(6, 10),
    tcl_count=6,
    pv_count=2,
    ess_count=2,
    seed_base=0,
    discrete_rate=0.0,
    T=24,
)
# for i, model in enumerate(sampled_models_list):
#     print(f"  模型{i + 1}: "
#           f"EV数量={model['config']['ev_count']}",
#           f"TCL平均温度={model['config']['tcl_temp_ambient_avg']}",
#           f"PV发电量={model['config']['pv_energy']}",
#           f"ESS={model['config']['ess_init_soc']}",
#           )


# avg_cfg = get_average_config(sampled_models_list)
# print(f"平均参数: EV={avg_cfg['ev_count']}, 温度={avg_cfg['tcl_temp']:.2f}, "
#       f"PV={avg_cfg['pv_energy']:.2f},初始能量={avg_cfg['ess_init_soc']:.2f}"
#       )
#
# agg = Aggregator(seed=0, discrete_rate=0.0,
#                  temp_idx=avg_cfg['temp_idx'],
#                  pv_idx=avg_cfg['pv_idx'],)
# agg.gen_EV(avg_cfg['ev_count'])
# agg.gen_TCL(6)
# agg.gen_PV(2)
# agg.gen_ESS(2,avg_cfg['ess_init_soc'])
# case = agg.case_aggregator(model_type=model_type)


# #fullnet

case = sampled_models_list[0]['case'] #采样函数中已写,自带fullnet
model_type = 'fullnet'

if model_type=='pretrainnet':
    n_train = case.get('n_train',2)
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:
    n_train = case.get('n_train',2)
    model = PreTrainNet(case['A_hat'], case['b_hat'])
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights_20260813_025617.pth',map_location=device))
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
# 预训练，不分阶段
# trainer.configure(**case['trainer_configure'])
# trainer.configure(
#     #feas_tol = 1e-1,
#                   # feas_tol=1e-3,
#                   # opt_tol=1e-3,
#                   rate_opt_feas = 1.0,
#                   scheduler={ "type": "StepLR",
#                               "step_size":50,
#                               "gamma": 0.95
#                             },
#                   lr =1e-3,
# )
# trainer.initialize()
# trainer.train(n_train = 3000 , params_data = case['params'])
#
#
# #fullnet
trainer.configure(**case['trainer_configure'])
trainer.configure(
                  rate_opt_feas = 1.0,
                  lr = 1e-3,#5e-5
                  scheduler={"type": "StepLR", "step_size": 200, "gamma": 0.95},
)
trainer.initialize()
trainer.train(n_train = 3600 ,train_model_list=sampled_models_list)
#2000轮
trainer.configure(**case['trainer_configure'])
trainer.configure(
                  rate_opt_feas = 0.1,
                  lr =8e-4,
                  scheduler={ "type": "StepLR", "step_size": 200, "gamma": 0.95},
)
trainer.initialize()
trainer.train(n_train = 3600 ,train_model_list=sampled_models_list)

trainer.configure(**case['trainer_configure'])
trainer.configure(
                  rate_opt_feas = 0.0001,
                  lr=3e-4,
                  scheduler={ "type": "StepLR", "step_size": 300, "gamma": 0.95}
)
trainer.initialize()
trainer.train(n_train = 7200,train_model_list=sampled_models_list)
#
training_end = time.time()
print('总耗时',training_end-training_start)
torch.save(model.state_dict(), case['result_path'])

# # 保存训练历史
history_dir = os.path.join(os.path.dirname(case['result_path']), 'history')
trainer.save_history(
    save_dir=history_dir,
    phase_name=model_type,
    config_info={
        'n_train': n_train,
        'model_type': model_type,
    }
)
