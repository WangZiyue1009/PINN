#没有预训练
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.aggregation_case import Aggregator, sampled_models_train, get_average_config
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
avg_cfg = get_average_config(sampled_models_list)
print(f"平均参数: EV={avg_cfg['ev_count']}, 温度={avg_cfg['tcl_temp']:.2f}, "
      f"PV={avg_cfg['pv_energy']:.2f}")

agg = Aggregator(seed=0, discrete_rate=0.0,
                 temp_idx=avg_cfg['temp_idx'],
                 pv_idx=avg_cfg['pv_idx'],)
agg.gen_EV(avg_cfg['ev_count'])
agg.gen_TCL(80)
agg.gen_PV(20)
case1 = agg.case_aggregator(model_type='pretrainnet')


# #fullnet

case = sampled_models_list[0]['case'] #采样函数中已写,自带fullnet
model_type = 'fullnet'
A_pretrained, b_pretrained = case1['A_hat'], case1['b_hat']

if torch.is_tensor(A_pretrained):
    A_pretrained = A_pretrained.detach().cpu().numpy()
if torch.is_tensor(b_pretrained):
    b_pretrained = b_pretrained.detach().cpu().numpy()
# b_pretrained = b_pretrained[0].detach().cpu().numpy()
# A_pretrained = A_pretrained[0].detach().cpu().numpy()

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
# trainer.train(n_train = 5000 , params_data = case['params'])


#fullnet
trainer.configure(**case['trainer_configure'])
trainer.configure(
                  # feas_tol = 1e-3,
                  rate_opt_feas = 1.0,
                  lr = 1e-3,#5e-5
                  scheduler={"type": "StepLR", "step_size": 200, "gamma": 0.95},
)
trainer.initialize()
trainer.train(n_train = 6000 ,train_model_list=sampled_models_list)
#2000轮
trainer.configure(**case['trainer_configure'])
trainer.configure(
                  # feas_tol = 1e-3,
                  rate_opt_feas = 0.1,
                  lr =8e-4,
                  scheduler={ "type": "StepLR", "step_size": 200, "gamma": 0.95},
)
trainer.initialize()
trainer.train(n_train = 10000 ,train_model_list=sampled_models_list)

trainer.configure(**case['trainer_configure'])
trainer.configure(
                  rate_opt_feas = 0.001,
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
        'n_train': 6000,
        'model_type': model_type,
    }
)
