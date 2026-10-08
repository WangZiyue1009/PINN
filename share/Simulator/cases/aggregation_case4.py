import numpy as np
import pandas as pd
import time
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator
import pyomo.environ as pyo
from Simulator.Plotter import ErrorVisualizer
import os
from typing import List,Dict, Any,Optional
import torch
from torch.utils.data import Dataset, DataLoader
import math
import scipy.io as sio
import pickle
from scipy.stats import qmc
import itertools
import json

class CaseData(Dataset):
    def __init__(self, ev_data: np.ndarray, tcl_data: np.ndarray, pv_data: np.ndarray,):

        if ev_data.ndim == 1:
            ev_data = ev_data.reshape(-1, 1)
        if tcl_data.ndim == 1:
            tcl_data = tcl_data.reshape(-1, 1)
        if pv_data.ndim == 1:
            pv_data = pv_data.reshape(-1, 1)


        assert ev_data.shape[0] == tcl_data.shape[0] == pv_data.shape[0], "所有数据样本数必须一致"
        assert ev_data.shape[1] == 1, "ev_data特征维度应为1"
        assert tcl_data.shape[1] == 1, "tcl_data特征维度应为1"
        assert pv_data.shape[1] == 1, "pv_data特征维度应为1"



        # 转换为torch张量
        self.ev_data = torch.tensor(ev_data, dtype=torch.float32)
        self.tcl_data = torch.tensor(tcl_data, dtype=torch.float32)
        self.pv_data = torch.tensor(pv_data, dtype=torch.float32)


        self.size = ev_data.shape[0]
    def __len__(self):
        return self.size
    def __getitem__(self, index: int):
        return {"ev_count": self.ev_data[index],
                "tcl_temp_ambient_avg": self.tcl_data[index],
                "pv_energy": self.pv_data[index],

                }


class Aggregator:
    def __init__(self,rng=None, seed=None,T = 24, data = None, discrete_rate = 0.0,temp_idx = None,pv_idx=None,

                 ):
        """
        初始化设备参数生成器

        参数:
            seed (int, optional): 随机种子，用于复现结果
        """
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        # self.rng = np.random.default_rng(seed)
        self.discrete_EV = 0
        self.discrete_TCL = 0
        self.T = T
        if data is None:
            self.EV_list = []
            self.TCL_list = []
            self.ESS_list = []
            self.PV_list = []

        else:
            self.EV_list = data['EV_list']
            self.discrete_EV = sum(self.EV_list[i]['is_discrete'] for i in range(len(self.EV_list)))

            self.TCL_list = data['TCL_list']
            self.discrete_TCL = sum(self.TCL_list[i]['is_discrete'] for i in range(len(self.TCL_list)))

            self.ESS_list = data['ESS_list']

            self.PV_list = data.get('PV_list')



        self.deltaT = 24/self.T
        self.discrete_rate = discrete_rate
        self.data_path = f'{PROJECT_ROOT}\\data\\'
        data = np.load(self.data_path + 'profiles_data\\profiles_data.npz')
        data_deltaT = 5/60

        # 环境温度
        temp_raw = data['temp_data']
        step = int(self.deltaT / data_deltaT)
        l = int(self.T * step)
        stride_raw = max(1, l // 4)  # 6h stride

        max_start_idx = len(temp_raw) - l
        if max_start_idx > 0:
            valid_start_indices = np.arange(0, max_start_idx + 1, stride_raw, dtype=int)
        else:
            valid_start_indices = np.array([0], dtype=int)

        if temp_idx is not None:
            candidate_idx = int(np.clip(temp_idx, 0, len(valid_start_indices) - 1))
            start_idx = valid_start_indices[candidate_idx]
        else:
            candidate_idx = self.rng.integers(0, len(valid_start_indices))
            start_idx = valid_start_indices[candidate_idx]

        self.theta_amb = temp_raw[start_idx: start_idx + l: step]

        #PV光伏
        pv_file_path = self.data_path + 'profiles_data\\PV_samples.mat'
        pv_mat = sio.loadmat(pv_file_path)
        pv_data_raw = pv_mat['PV_samples']
        PV_num_samples = pv_data_raw.shape[1]
        if pv_idx is not None:
            sample_idx = int(np.clip(pv_idx, 0, PV_num_samples - 1))
        else:
            sample_idx = self.rng.integers(0, PV_num_samples)

        raw_curve_96 = pv_data_raw[:, sample_idx]
        if len(raw_curve_96) == 96 and self.T == 24:
            self.pv_curve = raw_curve_96.reshape(self.T, 4).mean(axis=1)
        else:
            self.pv_curve = raw_curve_96[:self.T]


#分布式光伏集群
    def gen_PV(self, N):
        """生成N个光伏设备
        """
        for _ in range(N):
            # 组件额定功率 (DC端)
            # P_panel = self.rng.uniform(300, 800)
            P_panel = self.rng.uniform(15., 35.)
            # 容配比 (DC/AC比值)
            capacity_ratio = self.rng.uniform(1.0, 1.2)
            # 逆变器额定容量 (AC端)
            S_inv = P_panel / capacity_ratio
            eta_pv=self.rng.uniform(0.80, 0.95)
            base_power_curve = P_panel * eta_pv * self.pv_curve

            max_power_curve = np.minimum(base_power_curve, S_inv)

            self.PV_list.append({
                'P_panel': P_panel,
                'S_inv': S_inv,
                'capacity_ratio': capacity_ratio,
                'eta_pv': eta_pv,  # 综合效率系数
                'ramp_rate': self.rng.uniform(0.05, 0.15),# 爬坡率 5-15%
                'max_power_curve': max_power_curve
            })
        # print('PV generated')


    def gen_ESS(self, N):
        for i in range(N):
            self.ESS_list.append({
                'Pchg': self.rng.choice([25., 50.], p=[0.5, 0.5]),
                'init_SOC': self.rng.uniform(0.4, 0.6),
                'eta_chg': self.rng.uniform(0.95, 0.98),
                'eta_dis': self.rng.uniform(0.96, 0.98),
                'B': self.rng.choice([100, 200.], p=[0.5, 0.5]),
            })

        # print('ESS generated')



    def gen_TCL(self, N):
        #生成TCL的数量要固定
        hp_df = pd.read_csv(self.data_path + 'aggregator_data\\ZH_buildings.csv').head(N)
        COP = 3.85

        for index, row in hp_df.iterrows():
            # 提取必要的字段
            HBLD = row['HBLD']
            CBLD = row['CBLD']
            PRT = row['PRT']
            is_discrete= self.rng.choice([0, 1], p=[1-self.discrete_rate, self.discrete_rate])
            self.discrete_TCL += is_discrete
            # temp_aver = np.random.choice(available_temps)
            # 创建字典并添加到列表
            self.TCL_list.append({
                'H': HBLD,  # 热容
                'C': CBLD,  # 热阻
                'temp_min':17.0,
                'temp_max':23.0,
                'temp_set':20,
                'Qmax': PRT,  # 热功率
                'COP': COP,  # COP系数
                'is_discrete': is_discrete,
            })
        # print('TCL generated')

    def gen_EV(self,N,ta_mean=8.5):
        """
        生成随机个电动汽车(EV)的充电需求参数5-10
        完全按照MATLAB版本逻辑实现
        """
        P_chg = 7  # 充电功率 (kW)
        B = 50  # 电池容量 (kWh)
        eta_chg = 0.95  # 充电效率
        i = 0
        while i < N:
            is_discrete= self.rng.choice([0, 1], p=[1-self.discrete_rate, self.discrete_rate])
            self.discrete_EV += is_discrete
            # 白天充电 (不跨天)
            start_time = min(10, max(7, self.rng.normal(ta_mean, 1.5)))
            end_time = min(20, start_time + self.rng.uniform(8, 12))
            end_SOC = 1-P_chg/B
            start_SOC = min(1-2*P_chg/B, max(self.rng.normal(0.5, 0.2), 0.2))

            # 检查充电功率是否足够
            if (end_SOC - start_SOC) * B / (end_time - start_time) > P_chg:
                pass  # 原MATLAB代码中这里只是占位

            self.EV_list.append({
                'ta': start_time,
                'td': end_time,
                'SOCa': start_SOC,
                'SOCd': end_SOC,
                'SOCmax': 1,
                'P_chg': P_chg,
                'B': B,
                'eta_chg': eta_chg,
                'is_discrete': is_discrete,
            })
            i += 1
        print('EV generated')



    def case_aggregator(self,A = None, b = None, model_type='pretrainnet'):
        # 构建Pyomo模型

        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(self.T))
        model.var_proj = pyo.Var(model.T)  # 聚合功率变量
        p_max = np.zeros(self.T)
        p_min = np.zeros(self.T)


        model.ESSs= pyo.Set(initialize=range(len(self.ESS_list)))
        model.e_ESS= pyo.Var(model.ESSs, model.T, within=pyo.Reals)
        model.p_chg_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)
        model.p_dis_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)
        model.u_ESS = pyo.Var(model.ESSs, model.T, within=pyo.Binary)


        model.TCLs= pyo.Set(initialize=range(len(self.TCL_list)))
        model.discete_TCLs = pyo.Set(initialize=range(self.discrete_TCL))
        model.temp_TCL= pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
        model.u_TCL = pyo.Var(model.discete_TCLs, model.T, within=pyo.Binary)

        # 定义EV集合
        model.EVs = pyo.Set(initialize=range(len(self.EV_list)))
        model.discete_EVs = pyo.Set(initialize=range(self.discrete_EV))
        # 定义变量
        model.e_EV = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.u_EV = pyo.Var(model.discete_EVs, model.T, within=pyo.Binary)  # u_EV为0-1变量

        model.PVs = pyo.Set(initialize=range(len(self.PV_list)))
        model.p_PV = pyo.Var(model.PVs, model.T, within=pyo.NonPositiveReals)

        # PV 约束
        for i in model.PVs:
            pv_data = self.PV_list[i]
            p_min -= pv_data['max_power_curve']
            # 功率上下限约束
            def pv_power_limit_rule(m, t):
                return pyo.inequality(-pv_data['max_power_curve'][t], m.p_PV[i, t], 0)
            model.add_component(f"pv_power_limit_{i}", pyo.Constraint(model.T, rule=pv_power_limit_rule))

            #爬坡率约束
            def pv_ramp_up_constraint(m, t):
                if t > 0:
                    # 功率上升（数值变得更负），两者相减必须 <= 爬坡上限
                    return m.p_PV[i, t - 1] - m.p_PV[i, t] <= pv_data['ramp_rate'] * pv_data['S_inv']
                return pyo.Constraint.Skip
            model.add_component(f"pv_ramp_up_{i}", pyo.Constraint(model.T, rule=pv_ramp_up_constraint))

            def pv_ramp_down_constraint(m, t):
                if t > 0:
                    # 功率下降（数值向 0 靠近）
                    return m.p_PV[i, t] - m.p_PV[i, t - 1] <= pv_data['ramp_rate'] * pv_data['S_inv']
                return pyo.Constraint.Skip
            model.add_component(f"pv_ramp_down_{i}", pyo.Constraint(model.T, rule=pv_ramp_down_constraint))


        # ESS能量动态方程
        for i in model.ESSs:
            ess_data = self.ESS_list[i]
            p_max += ess_data['Pchg']
            p_min -= ess_data['Pchg']
            def energy_dynamics_rule(m, t):
                if t > 0:
                    return m.e_ESS[i, t] == m.e_ESS[i, t - 1] + (self.deltaT * (
                                m.p_chg_ESS[i, t] * ess_data['eta_chg'] - m.p_dis_ESS[i, t] / ess_data['eta_dis']))
                return pyo.Constraint.Skip

            model.add_component(f"ess_energy_dynamics_{i}", pyo.Constraint(model.T, rule=energy_dynamics_rule))

            def energy_limit_rule(m, t):
                return pyo.inequality(0, m.e_ESS[i, t], ess_data['B'])

            model.add_component(f"ess_energy_limit_{i}", pyo.Constraint(model.T, rule=energy_limit_rule))

            def charging_power_limit_rule(m, t):
                return m.p_chg_ESS[i, t]<=m.u_ESS[i,t]*ess_data['Pchg']

            model.add_component(f"ess_charging_power_limit_{i}", pyo.Constraint(model.T, rule=charging_power_limit_rule))

            def discharging_power_limit_rule(m, t):
                return m.p_dis_ESS[i, t]<=(1-m.u_ESS[i,t])*ess_data['Pchg']

            model.add_component(f"ess_discharging_power_limit_{i}",
                                pyo.Constraint(model.T, rule=discharging_power_limit_rule))

            def initial_energy_condition_rule(m):
                return m.e_ESS[i, 0] == ess_data['init_SOC'] * ess_data['B']

            model.add_component(f"ess_initial_energy_condition_{i}", pyo.Constraint(rule=initial_energy_condition_rule))

            def final_energy_condition_rule(m):
                return m.e_ESS[i, self.T - 1] >= m.e_ESS[i, 0]

            model.add_component(f"ess_final_energy_condition_{i}", pyo.Constraint(rule=final_energy_condition_rule))

        i_discrete = 0
        for i in model.TCLs:
            tcl_data = self.TCL_list[i]  # 默认使用第一个EV的数据
            p_max+=tcl_data['Qmax'] / tcl_data['COP']
            if tcl_data['is_discrete']:
                def power_limit_rule(m, t):
                    return m.p_TCL[i, t] == m.u_TCL[i_discrete, t] * tcl_data['Qmax']/tcl_data['COP']
                model.add_component(f"tcl_power_limit_discrete{i}", pyo.Constraint(model.T, rule=power_limit_rule))
                i_discrete+=1
            else:
                def power_limit_rule(m, t):
                    return pyo.inequality(0, m.p_TCL[i, t], tcl_data['Qmax']/tcl_data['COP'])
                model.add_component(f"tcl_power_limit{i}", pyo.Constraint(model.T, rule=power_limit_rule))

            alpha = np.exp(-self.deltaT*tcl_data['H']/tcl_data['C'])
            # 动态温度更新公式
            def temperature_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL[i, t] == (alpha * m.temp_TCL[i, t - 1] +(1 - alpha) * (self.theta_amb[t] +tcl_data['COP']/tcl_data['H']*model.p_TCL[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_dynamics{i}", pyo.Constraint(model.T, rule=temperature_dynamics_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_limit{i}", pyo.Constraint(model.T, rule=temperature_limit_rule))

            # # 温度终止条件：在最后时刻温度不小于设定的目标温度
            # def final_temperature_condition_rule(m):
            #     return m.temp_TCL[i, self.T - 1] >= tcl_data['temp_set']
            #
            # model.add_component(f"final_temperature_condition{i}",
            #                     pyo.Constraint(rule=final_temperature_condition_rule))
            def initial_temperature_condition_rule(m):
                return m.temp_TCL[i, 0] == tcl_data['temp_set']
            model.add_component(f"initial_temperature_condition{i}", pyo.Constraint(rule=initial_temperature_condition_rule))



        i_discrete = 0
        # 1. 确保每个EV的功率和能量满足边界条件
        for i in model.EVs:
            ev_data = self.EV_list[i]  # 默认使用第一个EV的数据
            # 计算离散化的到达和离开时间索引
            ta_idx = int(np.floor(ev_data['ta'] / self.deltaT))
            td_idx = int(np.ceil(ev_data['td'] / self.deltaT))
            p_max[ta_idx:td_idx]+=ev_data['P_chg']
            if ev_data['is_discrete']:
                def power_limit_rule(m, t):
                    if ta_idx <= t < td_idx:
                        return m.p_EV[i, t] == m.u_EV[i_discrete, t] * ev_data['P_chg']  # 功率为0或ev_data['P_chg']
                    else:
                        return m.p_EV[i, t] == 0  # 离开时功率为 0
                model.add_component(f"ev_power_limit_discrete{i}", pyo.Constraint(model.T, rule=power_limit_rule))
                i_discrete+=1
            else:
                # 约束1: 功率上下限约束
                def power_limit_rule(m, t):
                    if ta_idx <= t < td_idx:
                        return (0, m.p_EV[i,t], ev_data['P_chg'])
                    else:
                        return m.p_EV[i,t] == 0
                model.add_component(f"ev_power_limit{i}", pyo.Constraint(model.T, rule=power_limit_rule))

            # 约束2: 初始能量状态
            def initial_energy_rule(m):
                return m.e_EV[i,ta_idx] == ev_data['SOCa'] * ev_data['B']
            model.add_component(f"initial_energy{i}", pyo.Constraint(rule=initial_energy_rule))

            # 约束3: 能量动态更新
            def energy_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV[i,t + 1] == m.e_EV[i,t] + m.p_EV[i,t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_dynamics{i}", pyo.Constraint(model.T, rule=energy_dynamics_rule))

            # 约束4: 能量状态上下限
            def energy_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV[i,t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_limit{i}", pyo.Constraint(model.T, rule=energy_limit_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_rule(m):
                return m.e_EV[i, td_idx] >= ev_data['SOCd'] * ev_data['B']
            model.add_component(f"departure_energy{i}", pyo.Constraint(rule=departure_energy_rule))


        def agg_power_rule(m, t):
            return (m.var_proj[t]*(p_max[t]-p_min[t])+p_min[t] == sum(m.p_EV[i, t] for i in m.EVs)+sum(m.p_TCL[i, t] for i in m.TCLs)
                    +sum(m.p_PV[i, t] for i in m.PVs)

                    )
        model.agg_power_constraint = pyo.Constraint(model.T, rule=agg_power_rule)


        # 近似器矩阵（功率+能量约束）
        lower_tri = np.tril(np.ones((self.T, self.T)))
        lower_tri_normalized = lower_tri / lower_tri.sum(axis=1, keepdims=True)
        if A is None:
            A_hat = np.vstack([
                np.eye(self.T),  # 功率上限
                -np.eye(self.T),  # 功率下限
                lower_tri_normalized,  # 累积能量下限
                -lower_tri_normalized  # 累积能量上限
            ])
            # 误差计算器
            errorcalculator = ErrorCalculator(
                original_model={'model': model},
                A_hat=A_hat,
                solver='gurobi',
            )
        else:
            errorcalculator = ErrorCalculator(
                original_model={'model': model},
                A_hat=A,
                b_hat=b,
                solver='gurobi',
            )

        # 训练回调函数
        case_name = 'aggregation'
        cont = 'continuous' if not self.discrete_rate else 'discrete'
        figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\{cont}\\figures'
        os.makedirs(figure_folder, exist_ok=True)
        visualizer = ErrorVisualizer()
        num_sample = 20
        np.savetxt(f'{figure_folder}/b_{errorcalculator._iter}.csv', errorcalculator.b_hat, delimiter=',')
        np.savetxt(f'{figure_folder}/A_{errorcalculator._iter}.csv', errorcalculator.A_hat, delimiter=',')
        n_train =6000#预训练 6000
        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            if error_calculator._iter % 50 == 0:
                np.savetxt(f'{figure_folder}/b_{error_calculator._iter}.csv', error_calculator.b_hat, delimiter=',')
                np.savetxt(f'{figure_folder}/A_{error_calculator._iter}.csv', error_calculator.A_hat, delimiter=',')


            # visualizer.compute_errors(error_calculator, num_sample=num_sample)
            # print((np.mean(visualizer.error_history['error_feas'][-1]),np.mean(visualizer.error_history['error_opt'][-1])))
            # if error_calculator._iter % n_train == 0:
            #     visualizer.plot_dual_violin(save_path=f'{figure_folder}/{model_type}_errors_violin.png')
        # 训练参数配置
        if model_type.lower() == 'pretrainnet':
            trainer_configure = {
                "call_interval": 1,
                "training_callback": training_callback,
                "optimizer": 'adam',#pre训练的时候使用adam
                # "lr_A": 5e-9,
                # "lr_b": 1e-1,
                "lr": 1e-3,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 50, "gamma": 0.98},
                # "scheduler" :{"type": "MultiStepLR","milestones": [400], "gamma": 0.1},
                "n_cal": 3,
                "cal_feas": True,
                "cal_opt": True,
                "rate_opt_feas": 1.0
            }
        else:
            trainer_configure = {
                "call_interval": 1,
                "training_callback": training_callback,
                "optimizer": 'sgd',
                # "lr_A": 5e-9,
                # "lr_b": 1e-1,
                "lr": 1e-3,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
                "n_cal": 3,
                "cal_feas": True,
                "cal_opt": True,
                "rate_opt_feas": 1.0,

            }
        #关键参数设置
        if model_type.lower() == 'pretrainnet':
            params = {
                'params_dict': {},
                'dataloader': [None],
                'count': 0,
            }

        else:
            # -------- 强制检查训练归一化边界是否已经绑定 --------
            required_attrs = [
                'ev_min', 'ev_max',
                'temp_min', 'temp_max',
                'pv_min', 'pv_max',
            ]
            missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing_attrs:
                raise AttributeError(
                    f"Aggregator 缺少归一化边界属性: {missing_attrs}。"
                    f"请先调用 bind_training_norm_bounds_to_agg(agg, norm_bounds)"
                )

            # -------- 按训练边界归一化 --------
            ev_count = len(self.EV_list)
            ev_norm = 2.0 * (ev_count - self.ev_min) / (self.ev_max - self.ev_min) - 1.0

            tcl_temp_ambient_avg = np.mean(self.theta_amb)
            tcl_norm = 2.0 * (tcl_temp_ambient_avg - self.temp_min) / (self.temp_max - self.temp_min) - 1.0

            pv_total_energy = np.sum(self.pv_curve) * self.deltaT
            pv_norm = 2.0 * (pv_total_energy - self.pv_min) / (self.pv_max - self.pv_min) - 1.0



            ev_data_np = np.array([ev_norm])
            tcl_data_np = np.array([tcl_norm])
            pv_data_np = np.array([pv_norm])


            case_dataset = CaseData(
                ev_data=ev_data_np,
                tcl_data=tcl_data_np,
                pv_data=pv_data_np,

            )

            case_dataloader = DataLoader(
                dataset=case_dataset,
                batch_size=1,
            )

            params_dict = {
                "ev_count": ev_data_np,
                "tcl_temp_ambient_avg": tcl_data_np,
                "pv_energy": pv_data_np,

            }
            params = {
                'params_dict': params_dict,
                'dataloader': case_dataloader,
                'count': len(params_dict),
            }

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return {
            'model': model,
            'casename': case_name,
            'A_hat': errorcalculator.A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': f'{PROJECT_ROOT}\\results\\{case_name}\\{model_type}_weights_{timestamp}.pth',
            'metadata': {
                'T': self.T,
                'num_evs': len(self.EV_list),
                'ev_data_list': self.EV_list,
            },
            'n_train': n_train
        }

    def build_cube_approximation(self):
        start_t = time.perf_counter()
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(self.T))
        model.P_max = pyo.Var(model.T, within=pyo.Reals)
        model.P_min = pyo.Var(model.T, within=pyo.Reals)

        pmax = np.zeros(self.T)
        pmin = np.zeros(self.T)

        model.TCLs = pyo.Set(initialize=range(len(self.TCL_list)))
        model.temp_TCL_u = pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL_u = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
        model.temp_TCL_l = pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL_l = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
        model.PVs = pyo.Set(initialize=range(len(self.PV_list)))
        model.p_PV_u = pyo.Var(model.PVs, model.T, within=pyo.NonPositiveReals)
        model.p_PV_l = pyo.Var(model.PVs, model.T, within=pyo.NonPositiveReals)


        # 定义EV集合
        model.EVs = pyo.Set(initialize=range(len(self.EV_list)))
        model.e_EV_u = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV_u = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.e_EV_l = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV_l = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量

        for i in model.TCLs:
            tcl_data = self.TCL_list[i]
            pmax += tcl_data['Qmax'] / tcl_data['COP']

            def power_upper_limit_rule(m, t):
                return pyo.inequality(0, m.p_TCL_u[i, t], tcl_data['Qmax'] / tcl_data['COP'])

            model.add_component(f"tcl_power_upper_limit{i}", pyo.Constraint(model.T, rule=power_upper_limit_rule))

            def power_lower_limit_rule(m, t):
                return pyo.inequality(0, m.p_TCL_l[i, t], tcl_data['Qmax'] / tcl_data['COP'])

            model.add_component(f"tcl_power_lower_limit{i}", pyo.Constraint(model.T, rule=power_lower_limit_rule))

            def tcl_lower_upper_rule(m, t):
                return m.p_TCL_l[i, t] <= m.p_TCL_u[i, t]

            model.add_component(f"tcl_lower_upper{i}", pyo.Constraint(model.T, rule=tcl_lower_upper_rule))

            alpha = np.exp(-self.deltaT * tcl_data['H'] / tcl_data['C'])

            # 动态温度更新公式
            def temperature_u_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL_u[i, t] == (alpha * m.temp_TCL_u[i, t - 1] + (1 - alpha) * (
                            self.theta_amb[t] + tcl_data['COP'] / tcl_data['H'] * model.p_TCL_u[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_u_dynamics{i}", pyo.Constraint(model.T, rule=temperature_u_dynamics_rule))

            # 动态温度更新公式
            def temperature_l_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL_l[i, t] == (alpha * m.temp_TCL_l[i, t - 1] + (1 - alpha) * (
                            self.theta_amb[t] + tcl_data['COP'] / tcl_data['H'] * model.p_TCL_l[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_l_dynamics{i}", pyo.Constraint(model.T, rule=temperature_l_dynamics_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_u_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL_u[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_u_limit{i}", pyo.Constraint(model.T, rule=temperature_u_limit_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_l_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL_l[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_l_limit{i}", pyo.Constraint(model.T, rule=temperature_l_limit_rule))

            def initial_temperature_u_condition_rule(m):
                return m.temp_TCL_u[i, 0] == tcl_data['temp_set']

            model.add_component(f"initial_temperature_u_condition{i}",
                                pyo.Constraint(rule=initial_temperature_u_condition_rule))

            def initial_temperature_l_condition_rule(m):
                return m.temp_TCL_l[i, 0] == tcl_data['temp_set']

            model.add_component(f"initial_temperature_l_condition{i}",
                                pyo.Constraint(rule=initial_temperature_l_condition_rule))
        for i in model.PVs:
            pv_data = self.PV_list[i]
            pmin -= pv_data['max_power_curve']

            def pv_power_u_limit_rule(m, t):
                return pyo.inequality(-pv_data['max_power_curve'][t], m.p_PV_u[i, t], 0)

            model.add_component(f"pv_power_u_limit{i}", pyo.Constraint(model.T, rule=pv_power_u_limit_rule))

            def pv_power_l_limit_rule(m, t):
                return pyo.inequality(-pv_data['max_power_curve'][t], m.p_PV_l[i, t], 0)

            model.add_component(f"pv_power_l_limit{i}", pyo.Constraint(model.T, rule=pv_power_l_limit_rule))

            def pv_lower_upper_rule(m, t):
                return m.p_PV_l[i, t] <= m.p_PV_u[i, t]

            model.add_component(f"pv_lower_upper{i}", pyo.Constraint(model.T, rule=pv_lower_upper_rule))

            def pv_ramp_u_up_rule(m, t):
                if t > 0:
                    return m.p_PV_u[i, t - 1] - m.p_PV_u[i, t] <= pv_data['ramp_rate'] * pv_data['S_inv']
                return pyo.Constraint.Skip

            model.add_component(f"pv_ramp_u_up{i}", pyo.Constraint(model.T, rule=pv_ramp_u_up_rule))

            def pv_ramp_u_down_rule(m, t):
                if t > 0:
                    return m.p_PV_u[i, t] - m.p_PV_u[i, t - 1] <= pv_data['ramp_rate'] * pv_data['S_inv']
                return pyo.Constraint.Skip

            model.add_component(f"pv_ramp_u_down{i}", pyo.Constraint(model.T, rule=pv_ramp_u_down_rule))

            def pv_ramp_l_up_rule(m, t):
                if t > 0:
                    return m.p_PV_l[i, t - 1] - m.p_PV_l[i, t] <= pv_data['ramp_rate'] * pv_data['S_inv']
                return pyo.Constraint.Skip

            model.add_component(f"pv_ramp_l_up{i}", pyo.Constraint(model.T, rule=pv_ramp_l_up_rule))

            def pv_ramp_l_down_rule(m, t):
                if t > 0:
                    return m.p_PV_l[i, t] - m.p_PV_l[i, t - 1] <= pv_data['ramp_rate'] * pv_data['S_inv']
                return pyo.Constraint.Skip

            model.add_component(f"pv_ramp_l_down{i}", pyo.Constraint(model.T, rule=pv_ramp_l_down_rule))

        for i in model.EVs:
            ev_data = self.EV_list[i]  # 默认使用第一个EV的数据
            # 计算离散化的到达和离开时间索引
            ta_idx = int(np.floor(ev_data['ta'] / self.deltaT))
            td_idx = int(np.ceil(ev_data['td'] / self.deltaT))
            pmax[ta_idx:td_idx] += ev_data['P_chg']

            # 约束1: 功率上下限约束
            def power_u_limit_rule(m, t):
                if ta_idx <= t < td_idx:
                    return (0, m.p_EV_u[i, t], ev_data['P_chg'])
                else:
                    return m.p_EV_u[i, t] == 0

            model.add_component(f"ev_power_u_limit{i}", pyo.Constraint(model.T, rule=power_u_limit_rule))

            # 约束1: 功率上下限约束
            def power_l_limit_rule(m, t):
                if ta_idx <= t < td_idx:
                    return (0, m.p_EV_l[i, t], ev_data['P_chg'])
                else:
                    return m.p_EV_l[i, t] == 0

            model.add_component(f"ev_power_l_limit{i}", pyo.Constraint(model.T, rule=power_l_limit_rule))

            def ev_lower_upper_rule(m, t):
                return m.p_EV_l[i, t] <= m.p_EV_u[i, t]

            model.add_component(f"ev_lower_upper{i}", pyo.Constraint(model.T, rule=ev_lower_upper_rule))

            # 约束2: 初始能量状态
            def initial_energy_u_rule(m):
                return m.e_EV_u[i, ta_idx] == ev_data['SOCa'] * ev_data['B']

            model.add_component(f"initial_energy_u{i}", pyo.Constraint(rule=initial_energy_u_rule))

            # 约束2: 初始能量状态
            def initial_energy_l_rule(m):
                return m.e_EV_l[i, ta_idx] == ev_data['SOCa'] * ev_data['B']

            model.add_component(f"initial_energy_l{i}", pyo.Constraint(rule=initial_energy_l_rule))

            # 约束3: 能量动态更新
            def energy_u_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV_u[i, t + 1] == m.e_EV_u[i, t] + m.p_EV_u[i, t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_u_dynamics{i}", pyo.Constraint(model.T, rule=energy_u_dynamics_rule))

            # 约束3: 能量动态更新
            def energy_l_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV_l[i, t + 1] == m.e_EV_l[i, t] + m.p_EV_l[i, t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_l_dynamics{i}", pyo.Constraint(model.T, rule=energy_l_dynamics_rule))

            # 约束4: 能量状态上下限
            def energy_u_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV_u[i, t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_u_limit{i}", pyo.Constraint(model.T, rule=energy_u_limit_rule))

            # 约束4: 能量状态上下限
            def energy_l_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV_l[i, t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_l_limit{i}", pyo.Constraint(model.T, rule=energy_l_limit_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_u_rule(m):
                return m.e_EV_u[i, td_idx] >= ev_data['SOCd'] * ev_data['B']

            model.add_component(f"departure_energy_u{i}", pyo.Constraint(rule=departure_energy_u_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_l_rule(m):
                return m.e_EV_l[i, td_idx] >= ev_data['SOCd'] * ev_data['B']

            model.add_component(f"departure_energy_l{i}", pyo.Constraint(rule=departure_energy_l_rule))

        def agg_power_u_rule(m, t):
            return m.P_max[t] * (pmax[t] - pmin[t]) + pmin[t] == (
                sum(m.p_EV_u[i, t] for i in m.EVs)
                + sum(m.p_TCL_u[i, t] for i in m.TCLs)
                + sum(m.p_PV_u[i, t] for i in m.PVs)

            )

        model.agg_power_u_constraint = pyo.Constraint(model.T, rule=agg_power_u_rule)

        def agg_power_l_rule(m, t):
            return m.P_min[t] * (pmax[t] - pmin[t]) + pmin[t] == (
                sum(m.p_EV_l[i, t] for i in m.EVs)
                + sum(m.p_TCL_l[i, t] for i in m.TCLs)
                + sum(m.p_PV_l[i, t] for i in m.PVs)

            )

        model.agg_power_l_constraint = pyo.Constraint(model.T, rule=agg_power_l_rule)

        model.obj = pyo.Objective(expr=-sum(model.P_max[t] - model.P_min[t] for t in model.T))
        solver = pyo.SolverFactory('gurobi')
        solver.solve(model)
        comp_time = time.perf_counter() - start_t
        P_max = np.array([pyo.value(model.P_max[i]) for i in model.P_max])
        P_min = np.array([pyo.value(model.P_min[i]) for i in model.P_min])

        A_hat = np.vstack([np.eye(self.T), -np.eye(self.T)])
        b_hat = np.hstack([P_max, -P_min])
        return (A_hat, b_hat), comp_time

    def build_power_energy_approximation(
            self,
            include_base_load=False,
            solver_name="gurobi",
            eps=1e-8,
    ):


        start_t = time.perf_counter()

        T = self.T
        dt = self.deltaT

        def fast_tcl_energy_bounds(
                pmax,
                alpha,
                temp_min,
                temp_max,
                temp_set,
                cop,
                heat,
                theta_amb,
        ):

            theta_amb = np.asarray(theta_amb, dtype=float)

            E_lb_one = np.zeros(T, dtype=float)
            E_ub_one = np.zeros(T, dtype=float)

            E_ub_one[:] = np.cumsum(np.full(T, pmax * dt, dtype=float))

            beta = (1.0 - alpha) * cop / max(heat, eps)

            temp_no_power = np.zeros(T, dtype=float)
            temp_no_power[0] = temp_set

            for r in range(1, T):
                temp_no_power[r] = (
                        alpha * temp_no_power[r - 1]
                        + (1.0 - alpha) * theta_amb[r]
                )


            max_lower_power_sum = 0.0

            for r in range(1, T):
                required_weighted_sum = (temp_min - temp_no_power[r]) / max(beta, eps)
                required_power_sum = max(0.0, required_weighted_sum)

                max_lower_power_sum = max(max_lower_power_sum, required_power_sum)

                for k in range(r, T):
                    E_lb_one[k] = max(E_lb_one[k], max_lower_power_sum * dt)

            for r in range(1, T):
                allowed_weighted_sum = (temp_max - temp_no_power[r]) / max(beta, eps)

                if allowed_weighted_sum < 0:

                    allowed_power_sum_until_r = 0.0
                else:
                    allowed_power_sum_until_r = allowed_weighted_sum / max(alpha ** (r - 1), eps)

                for k in range(r, T):

                    remaining_steps = max(0, k - r)
                    upper_power_sum = (
                            pmax
                            + allowed_power_sum_until_r
                            + remaining_steps * pmax
                    )
                    E_ub_one[k] = min(E_ub_one[k], upper_power_sum * dt)

            E_lb_one = np.maximum(E_lb_one, 0.0)
            E_ub_one = np.minimum(E_ub_one, np.cumsum(np.full(T, pmax * dt, dtype=float)))


            E_ub_one = np.maximum(E_ub_one, E_lb_one + eps)

            return E_lb_one, E_ub_one


        p_lb = np.zeros(T, dtype=float)
        p_ub = np.zeros(T, dtype=float)
        E_lb = np.zeros(T, dtype=float)
        E_ub = np.zeros(T, dtype=float)


        if include_base_load and hasattr(self, "base_load_curve"):
            base = np.asarray(self.base_load_curve, dtype=float)

            p_lb += base
            p_ub += base

            base_energy = np.cumsum(base * dt)
            E_lb += base_energy
            E_ub += base_energy


        for pv_data in self.PV_list:
            pv_max = np.asarray(pv_data["max_power_curve"], dtype=float)


            p_lb += -pv_max

            E_lb += np.cumsum(-pv_max * dt)



        wind_list = getattr(self, "Wind_list", [])

        for wind_data in wind_list:
            wind_max = np.asarray(wind_data["max_power_curve"], dtype=float)

            p_lb += -wind_max
            E_lb += np.cumsum(-wind_max * dt)

        for ev_data in self.EV_list:
            ta_idx = int(np.floor(ev_data["ta"] / dt))
            td_idx = int(np.ceil(ev_data["td"] / dt))

            ta_idx = max(0, min(ta_idx, T))
            td_idx = max(0, min(td_idx, T))

            p_chg = float(ev_data["P_chg"])
            eta = float(ev_data.get("eta_chg", 1.0))
            battery = float(ev_data["B"])

            e0 = float(ev_data["SOCa"]) * battery
            e_req = float(ev_data["SOCd"]) * battery
            e_max = float(ev_data["SOCmax"]) * battery

            e_need_grid = max(0.0, (e_req - e0) / max(eta, eps))

            e_cap_grid = max(0.0, (e_max - e0) / max(eta, eps))

            if ta_idx < td_idx:
                p_ub[ta_idx:td_idx] += p_chg

            ev_E_lb = np.zeros(T, dtype=float)
            ev_E_ub = np.zeros(T, dtype=float)

            for k in range(T):
                k_end = k + 1

                elapsed_steps = max(0, min(k_end, td_idx) - ta_idx)
                remaining_steps = max(0, td_idx - max(k_end, ta_idx))

                max_charge_until_k = p_chg * elapsed_steps * dt
                max_charge_after_k = p_chg * remaining_steps * dt

                ev_E_ub[k] = min(max_charge_until_k, e_cap_grid)

                ev_E_lb[k] = max(0.0, e_need_grid - max_charge_after_k)

            E_lb += ev_E_lb
            E_ub += ev_E_ub

        theta_amb = np.asarray(self.theta_amb, dtype=float)

        for tcl_data in self.TCL_list:
            p_tcl_max = float(tcl_data["Qmax"] / tcl_data["COP"])

            p_ub += p_tcl_max

            alpha = np.exp(-self.deltaT * tcl_data["H"] / tcl_data["C"])
            temp_min = float(tcl_data["temp_min"])
            temp_max = float(tcl_data["temp_max"])
            temp_set = float(tcl_data["temp_set"])
            cop = float(tcl_data["COP"])
            heat = float(tcl_data["H"])

            tcl_E_lb, tcl_E_ub = fast_tcl_energy_bounds(
                pmax=p_tcl_max,
                alpha=alpha,
                temp_min=temp_min,
                temp_max=temp_max,
                temp_set=temp_set,
                cop=cop,
                heat=heat,
                theta_amb=theta_amb,
            )

            E_lb += tcl_E_lb
            E_ub += tcl_E_ub

        scale = p_ub - p_lb
        scale_safe = np.where(np.abs(scale) < eps, 1.0, scale)

        eye = np.eye(T)
        lower_tri = np.tril(np.ones((T, T))) * dt

        diag_scale = np.diag(scale_safe)


        A_energy = lower_tri @ diag_scale
        E_shift = lower_tri @ p_lb

        A_hat = np.vstack([
            eye,  # x <= 1
            -eye,  # -x <= 0, i.e. x >= 0
            A_energy,  # cumulative energy upper bound
            -A_energy,  # cumulative energy lower bound
        ])

        b_hat = np.hstack([
            np.ones(T),
            np.zeros(T),
            E_ub - E_shift,
            -(E_lb - E_shift),
        ])

        comp_time = time.perf_counter() - start_t

        return (A_hat, b_hat), comp_time

    def count_complexity(self):
        # 1. 查看连续变量个数
        continuous_vars = sum(1 for v in self.model.component_data_objects(pyo.Var, active=True)
                              if not v.is_binary() and not v.is_integer())
        print(f"连续变量个数: {continuous_vars}")

        # 2. 查看0-1变量（二进制变量）个数
        binary_vars = sum(1 for v in self.model.component_data_objects(pyo.Var, active=True)
                          if v.is_binary())
        print(f"0-1变量个数: {binary_vars}")

        # 3. 查看约束数量
        constraints = sum(1 for c in self.model.component_data_objects(pyo.Constraint, active=True))
        print(f"约束数量: {constraints}")

#固定训练归一化边界
def get_training_norm_bounds(
    ev_range: tuple,
    ta_mean_range: tuple = (7.5, 9.5),  # [新增] 训练集平均到达时间窗口 (小时)
    T: int = 24,
):

    deltaT = 24 / T

    temp_raw = np.load(f'{PROJECT_ROOT}\\data\\profiles_data\\profiles_data.npz')['temp_data']
    step = int(deltaT / (5 / 60))
    l = int(T * step)
    stride_raw = max(1, l // 4)  # 6h stride
    max_start_idx = len(temp_raw) - l

    if max_start_idx > 0:
        valid_start_indices = np.arange(0, max_start_idx + 1, stride_raw, dtype=int)
    else:
        valid_start_indices = np.array([0], dtype=int)

    avg_temps = np.array([
        np.mean(temp_raw[idx: idx + l: step][:T])
        for idx in valid_start_indices
    ], dtype=float)

    tcl_temp_min = float(np.floor(np.min(avg_temps)))
    tcl_temp_max = float(np.ceil(np.max(avg_temps)))

    # -------- PV 能量边界 --------
    pv_data_raw = sio.loadmat(f'{PROJECT_ROOT}\\data\\profiles_data\\PV_samples.mat')['PV_samples']
    num_pv_samples = pv_data_raw.shape[1]

    pv_total_energy = []
    for idx in range(num_pv_samples):
        raw_curve = pv_data_raw[:, idx]
        if len(raw_curve) == 96 and T == 24:
            energy = np.sum(raw_curve.reshape(T, 4).mean(axis=1)) * deltaT
        else:
            energy = np.sum(raw_curve[:T]) * deltaT
        pv_total_energy.append(energy)
    pv_total_energy = np.asarray(pv_total_energy, dtype=float)

    pv_energy_min = float(np.min(pv_total_energy))
    pv_energy_max = float(np.max(pv_total_energy))


    return {
        'ev_min': float(ev_range[0]),
        'ev_max': float(ev_range[1]),

        'tcl_temp_min': tcl_temp_min,
        'tcl_temp_max': tcl_temp_max,

        'pv_energy_min': pv_energy_min,
        'pv_energy_max': pv_energy_max,

        'ta_mean_min': float(ta_mean_range[0]),  # [新增]
        'ta_mean_max': float(ta_mean_range[1]),  # [新增]

    }

def get_train_split_norm_bounds(
    ev_range: tuple,
    reference_tables: dict,
    profile_pools: dict,
    ta_mean_range: tuple = (7.5, 9.5),  # [新增]
):

    temp_train_values = reference_tables['avg_temps'][profile_pools['temp_train_ids']]
    tcl_temp_min = float(np.floor(np.min(temp_train_values)))
    tcl_temp_max = float(np.ceil(np.max(temp_train_values)))

    pv_train_values = reference_tables['pv_total_energy'][profile_pools['pv_train_ids']]
    pv_energy_min = float(np.min(pv_train_values))
    pv_energy_max = float(np.max(pv_train_values))



    return {
        'ev_min': float(ev_range[0]),
        'ev_max': float(ev_range[1]),

        'tcl_temp_min': tcl_temp_min,
        'tcl_temp_max': tcl_temp_max,

        'pv_energy_min': pv_energy_min,
        'pv_energy_max': pv_energy_max,

        'ta_mean_min': float(ta_mean_range[0]),  # [新增]
        'ta_mean_max': float(ta_mean_range[1]),  # [新增]
    }

#统一归一化
def normalize_to_minus1_1(x, xmin, xmax):

    if xmax <= xmin:
        raise ValueError(f'归一化边界非法: xmin={xmin}, xmax={xmax}')
    return 2.0 * (float(x) - float(xmin)) / (float(xmax) - float(xmin)) - 1.0

#反归一化函数
def denormalize_from_minus1_1(x_norm, xmin, xmax):

    if xmax <= xmin:
        raise ValueError(f'反归一化边界非法: xmin={xmin}, xmax={xmax}')
    return 0.5 * (float(x_norm) + 1.0) * (float(xmax) - float(xmin)) + float(xmin)

#把归一化边界绑定到 Aggregator
def bind_training_norm_bounds_to_agg(agg: Aggregator, norm_bounds: dict):
    """
    将训练归一化边界绑定到 Aggregator。
    后续 case_aggregator() 会读取这些属性来构造 params_dict。
    """
    agg.ev_min = norm_bounds['ev_min']
    agg.ev_max = norm_bounds['ev_max']

    agg.temp_min = norm_bounds['tcl_temp_min']
    agg.temp_max = norm_bounds['tcl_temp_max']

    agg.pv_min = norm_bounds['pv_energy_min']
    agg.pv_max = norm_bounds['pv_energy_max']


#把物理配置构造成 Aggregator
def build_case_from_physical_config(
    config: dict,
    norm_bounds: dict,
    seed: int,
    discrete_rate: float = 0.0,
    T: int = 24,
    temp_idx: int = None,
    pv_idx: int = None,
    # wind_idx: int = None,
):
    """
    根据物理配置构建 case。

    参数:
        config: 物理配置字典，至少包含:
            - ev_count
            - tcl_count
            - pv_count
            - bl_count
            - wind_count
        norm_bounds: 训练归一化边界
        seed: 随机种子
        discrete_rate: 离散比例
        T: 时间长度
        temp_idx / pv_idx / wind_idx:
            若指定，则固定使用这些外部样本索引
    """
    rng = np.random.default_rng(seed)

    agg = Aggregator(
        rng=rng,
        T=T,
        data=None,
        discrete_rate=discrete_rate,
        temp_idx=temp_idx,
        pv_idx=pv_idx,
    )

    bind_training_norm_bounds_to_agg(agg, norm_bounds)

    ta_mean = config.get('ta_mean', 17.5)
    agg.gen_EV(int(config['ev_count']),ta_mean=ta_mean)
    agg.gen_TCL(int(config['tcl_count']))
    agg.gen_PV(int(config['pv_count']))

    case_dict = agg.case_aggregator(model_type='fullnet')

    return agg, case_dict

#从目标物理量映射到最近的外部样本索引
def build_profile_reference_tables(T: int = 24):

    deltaT = 24 / T


    temp_raw = np.load(f'{PROJECT_ROOT}\\data\\profiles_data\\profiles_data.npz')['temp_data']
    step = int(deltaT / (5 / 60))
    l = int(T * step)
    stride_raw = max(1, l // 4)  # 6h stride
    max_start_idx = len(temp_raw) - l

    if max_start_idx > 0:
        valid_start_indices = np.arange(0, max_start_idx + 1, stride_raw, dtype=int)
    else:
        valid_start_indices = np.array([0], dtype=int)

    avg_temps = np.array([
        np.mean(temp_raw[idx: idx + l: step][:T])
        for idx in valid_start_indices
    ], dtype=float)


    # -------- PV 能量参考 --------
    pv_data_raw = sio.loadmat(f'{PROJECT_ROOT}\\data\\profiles_data\\PV_samples.mat')['PV_samples']
    num_pv_samples = pv_data_raw.shape[1]

    pv_total_energy = []
    for idx in range(num_pv_samples):
        raw_curve = pv_data_raw[:, idx]
        if len(raw_curve) == 96 and T == 24:
            energy = np.sum(raw_curve.reshape(T, 4).mean(axis=1)) * deltaT
        else:
            energy = np.sum(raw_curve[:T]) * deltaT
        pv_total_energy.append(energy)
    pv_total_energy = np.asarray(pv_total_energy, dtype=float)

    return {
        'avg_temps': avg_temps,
        'pv_total_energy': pv_total_energy,
        'num_pv_samples': num_pv_samples,

    }

def build_train_test_profile_pools(reference_tables: dict):

    def split_extreme_ids_by_value(values: np.ndarray, ratio: float = 0.2):

        values = np.asarray(values, dtype=float)
        n_total = len(values)
        if n_total < 3:
            raise ValueError("profile 样本数量过少，无法构建 train/test 划分。")

        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]

        n_test_each = max(1, int(np.round(n_total * ratio)))
        n_test_each = min(n_test_each, (n_total - 1) // 2)

        low_boundary = float(sorted_values[n_test_each - 1])
        high_boundary = float(sorted_values[-n_test_each])

        test_mask = (values <= low_boundary) | (values >= high_boundary)
        train_mask = (values > low_boundary) & (values < high_boundary)

        if not np.any(train_mask):
            raise ValueError("严格值域切分后 train pool 为空，请检查 profile 分布。")
        if not np.any(test_mask):
            raise ValueError("严格值域切分后 test pool 为空，请检查 profile 分布。")

        train_ids = np.sort(np.flatnonzero(train_mask).astype(int))
        test_ids = np.sort(np.flatnonzero(test_mask).astype(int))
        return train_ids, test_ids

    # ---------- temperature：按平均温度排序 ----------
    avg_temps = reference_tables['avg_temps']
    temp_train_ids, temp_test_ids = split_extreme_ids_by_value(avg_temps, ratio=0.2)

    # ---------- PV：按总发电量排序 ----------
    pv_energy = reference_tables['pv_total_energy']
    pv_train_ids, pv_test_ids = split_extreme_ids_by_value(pv_energy, ratio=0.2)

    temp_train_values = avg_temps[temp_train_ids]
    temp_test_values = avg_temps[temp_test_ids]
    pv_train_values = pv_energy[pv_train_ids]
    pv_test_values = pv_energy[pv_test_ids]

    if len(temp_test_ids) == 0:
        raise ValueError("温度 test 索引池为空，请检查温度窗口数量。")
    if len(temp_train_ids) == 0:
        raise ValueError("温度 train 索引池为空，请检查温度窗口数量。")
    if len(pv_test_ids) == 0:
        raise ValueError("PV test 索引池为空，请检查 PV 样本数量。")
    if len(pv_train_ids) == 0:
        raise ValueError("PV train 索引池为空，请检查 PV 样本数量。")
    if np.any(np.isclose(temp_test_values[:, None], temp_train_values[None, :], atol=1e-12, rtol=0.0)):
        raise ValueError("温度 train/test 划分存在重复取值，无法保证分布外测试未见过。")
    if np.any(np.isclose(pv_test_values[:, None], pv_train_values[None, :], atol=1e-12, rtol=0.0)):
        raise ValueError("PV train/test 划分存在重复取值，无法保证分布外测试未见过。")

    return {
        "temp_train_ids": temp_train_ids,
        "temp_test_ids": temp_test_ids,
        "pv_train_ids": pv_train_ids,
        "pv_test_ids": pv_test_ids,
    }


def nearest_index_from_target_in_pool(
    target_value: float,
    reference_array: np.ndarray,
    allowed_ids: np.ndarray,
) -> int:

    allowed_ids = np.asarray(allowed_ids, dtype=int)
    if allowed_ids.size == 0:
        raise ValueError("allowed_ids 为空，无法选择 profile。")

    allowed_values = reference_array[allowed_ids]
    local_idx = int(np.argmin(np.abs(allowed_values - float(target_value))))
    return int(allowed_ids[local_idx])


def physical_config_to_profile_indices_split(
    config: dict,
    reference_tables: dict,
    pools: dict,
    split: str = 'train',
):

    if split not in ('train', 'test'):
        raise ValueError(f"split 必须是 'train' 或 'test'，当前为: {split}")

    temp_ids = pools[f'temp_{split}_ids']
    pv_ids = pools[f'pv_{split}_ids']
    # wind_ids = pools[f'wind_{split}_ids']

    tcl_idx = nearest_index_from_target_in_pool(
        config['tcl_temp_ambient_avg'],
        reference_tables['avg_temps'],
        temp_ids
    )

    pv_idx = nearest_index_from_target_in_pool(
        config['pv_energy'],
        reference_tables['pv_total_energy'],
        pv_ids
    )



    return tcl_idx, pv_idx


def print_profile_split_summary(reference_tables: dict, pools: dict):
    """
    打印 train / test profile 划分摘要，方便检查。
    """
    print("=== Profile train/test split summary ===")
    print(f"Temperature windows: total={len(reference_tables['avg_temps'])}, "
          f"train={len(pools['temp_train_ids'])}, test={len(pools['temp_test_ids'])}, "
          f"train_range=[{pools['temp_train_ids'][0]}, {pools['temp_train_ids'][-1]}], "
          f"test_range=[{pools['temp_test_ids'][0]}, {pools['temp_test_ids'][-1]}]")

    print(f"PV samples: total={reference_tables['num_pv_samples']}, "
          f"train={len(pools['pv_train_ids'])}, test={len(pools['pv_test_ids'])}, "
          f"train_range=[{pools['pv_train_ids'][0]}, {pools['pv_train_ids'][-1]}], "
          f"test_range=[{pools['pv_test_ids'][0]}, {pools['pv_test_ids'][-1]}]")


def nearest_index_from_target(target_value: float, reference_array: np.ndarray) -> int:

    return int(np.argmin(np.abs(reference_array - float(target_value))))

#把归一化参数点映射成物理配置
def norm_point_to_physical_config(
    norm_point: np.ndarray,
    norm_bounds: dict,
    base_counts: dict = None,
):

    if base_counts is None:
        base_counts = {
            'tcl_count': 150,
            'pv_count': 30,

        }

    ev_count = int(round(
        denormalize_from_minus1_1(
            norm_point[0],
            norm_bounds['ev_min'],
            norm_bounds['ev_max']
        )
    ))

    tcl_temp = denormalize_from_minus1_1(
        norm_point[1],
        norm_bounds['tcl_temp_min'],
        norm_bounds['tcl_temp_max']
    )

    pv_energy = denormalize_from_minus1_1(
        norm_point[2],
        norm_bounds['pv_energy_min'],
        norm_bounds['pv_energy_max']
    )

    # [新增] 反归一化得到平均到达时间 ta_mean
    ta_mean = denormalize_from_minus1_1(
        norm_point[3],
        norm_bounds['ta_mean_min'],
        norm_bounds['ta_mean_max']
    )

    return {
        'ev_count': ev_count,
        'tcl_temp_ambient_avg': float(tcl_temp),
        'pv_energy': float(pv_energy),

        'ta_mean': float(ta_mean),  # [新增] 返回到达时间

        'tcl_count': int(base_counts['tcl_count']),
        'pv_count': int(base_counts['pv_count']),

    }


# 把物理配置落到最近的可生成 profile 上
def physical_config_to_profile_indices(config: dict, reference_tables: dict):

    tcl_idx = nearest_index_from_target(
        config['tcl_temp_ambient_avg'],
        reference_tables['avg_temps']
    )

    pv_idx = nearest_index_from_target(
        config['pv_energy'],
        reference_tables['pv_total_energy']
    )



    return tcl_idx, pv_idx

#训练集
def sampled_models_train(
    n_samples: int = 150,
    ev_range: tuple = (60, 100),
    tcl_count: int = 150,
    pv_count: int = 30,
    seed_base: int = 0,
    discrete_rate: float = 0.0,
    T: int = 24,
    cache_path: str = None,
    corner_ratio: float = 0.15,
):

    if cache_path and os.path.exists(cache_path):
        print(f"从缓存加载训练采样配置: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        norm_bounds = cache_data['norm_bounds']
        configs = cache_data['configs']
        ref_tables = build_profile_reference_tables(T=T)
        profile_pools = build_train_test_profile_pools(ref_tables)
        samples = []
        for i, cfg in enumerate(configs):
            tcl_idx, pv_idx = physical_config_to_profile_indices_split(
                config=cfg,
                reference_tables=ref_tables,
                pools=profile_pools,
                split='train',
            )

            agg, case_dict = build_case_from_physical_config(
                config={
                    'ev_count': cfg['ev_count'],
                    'tcl_count': cfg['tcl_count'],
                    'pv_count': cfg['pv_count'],
                },
                norm_bounds=norm_bounds,
                seed=seed_base + i,
                discrete_rate=discrete_rate,
                T=T,
                temp_idx=tcl_idx,
                pv_idx=pv_idx,
            )

            samples.append({
                'model': case_dict['model'],
                'case': case_dict,
                'params': case_dict['params'],
                'config': cfg,
                'norm_bounds': norm_bounds,
                'sample_type': 'train',
            })

        print(f"训练集成功从缓存恢复 {len(samples)} 个场景")
        return samples

    # ---------- 训练参考表与 split ----------
    ref_tables = build_profile_reference_tables(T=T)
    profile_pools = build_train_test_profile_pools(ref_tables)

    # ---------- 训练归一化边界：只基于 train split ----------
    norm_bounds = get_train_split_norm_bounds(
        ev_range=ev_range,
        reference_tables=ref_tables,
        profile_pools=profile_pools,
    )

    base_counts = {
        'tcl_count': tcl_count,
        'pv_count': pv_count,
    }

    n_corner = int(round(n_samples * corner_ratio))
    n_corner = max(4, min(8, n_corner))
    n_fill = max(0, n_samples - n_corner)

    if n_fill > 0:
        sampler = qmc.Sobol(d=4, scramble=True, seed=seed_base)
        m = int(np.ceil(np.log2(n_fill)))
        fill_samples = sampler.random_base2(m=m)[:n_fill]
        fill_samples = 2.0 * fill_samples - 1.0  # [0,1] -> [-1,1]
    else:
        fill_samples = np.empty((0, 4), dtype=float)

    all_corners = np.array(list(itertools.product([-1.0, 1.0], repeat=4)), dtype=float)
    rng = np.random.default_rng(seed_base)
    rng.shuffle(all_corners)

    if n_corner <= len(all_corners):
        corner_samples = all_corners[:n_corner]
    else:
        extra_idx = rng.choice(len(all_corners), size=n_corner, replace=True)
        corner_samples = all_corners[extra_idx]

    norm_points = np.vstack([corner_samples, fill_samples]) if n_corner > 0 else fill_samples

    samples = []
    used_keys = set()

    for i, norm_point in enumerate(norm_points):
        physical_cfg = norm_point_to_physical_config(
            norm_point=norm_point,
            norm_bounds=norm_bounds,
            base_counts=base_counts
        )

        tcl_idx, pv_idx = physical_config_to_profile_indices_split(
            config=physical_cfg,
            reference_tables=ref_tables,
            pools=profile_pools,
            split='train',
        )

        unique_key = (
            int(physical_cfg['ev_count']),
            int(tcl_idx),
            int(pv_idx),
            int(tcl_count),
            int(pv_count),
            round(physical_cfg['ta_mean'], 2)
        )
        if unique_key in used_keys:
            continue
        used_keys.add(unique_key)

        agg, case_dict = build_case_from_physical_config(
            config={
                'ev_count': physical_cfg['ev_count'],
                'tcl_count': tcl_count,
                'pv_count': pv_count,
            },
            norm_bounds=norm_bounds,
            seed=seed_base + i,
            discrete_rate=discrete_rate,
            T=T,
            temp_idx=tcl_idx,
            pv_idx=pv_idx,
        )

        sample = {
            'model': case_dict['model'],
            'case': case_dict,
            'params': case_dict['params'],
            'config': {
                'model_id': len(samples),
                'ev_count': int(physical_cfg['ev_count']),
                'tcl_temp_ambient_avg': float(ref_tables['avg_temps'][tcl_idx]),
                'pv_energy': float(ref_tables['pv_total_energy'][pv_idx]),
                'tcl_count': int(tcl_count),
                'pv_count': int(pv_count),

                'ta_mean': physical_cfg['ta_mean'],  # [新增] 记录
            },
            'norm_bounds': norm_bounds,
            'sample_type': 'train',
        }
        samples.append(sample)

        if len(samples) >= n_samples:
            break

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'mode': 'train',
            'norm_bounds': norm_bounds,
            'configs': [s['config'] for s in samples],
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"训练采样缓存已保存: {cache_path}")

    print(
        f"训练集成功采样 {len(samples)} 个场景 "
        f"(Sobol={len(fill_samples)}, corner={len(corner_samples)})"
    )
    return samples

def sampled_models_test(
    n_samples: int = 20,
    train_ev_range: tuple = (60, 100),
    test_ev_range: tuple = None,
    tcl_count: int = 80,
    pv_count: int = 30,
    seed_base: int = 1000,
    discrete_rate: float = 0.0,
    T: int = 24,
    test_mode: str = 'interpolation',
):

    valid_modes = ('conservative', 'interpolation', 'extrapolation')
    if test_mode not in valid_modes:
        raise ValueError(f"test_mode 必须是 {valid_modes}，当前: {test_mode}")

    ref_tables = build_profile_reference_tables(T=T)
    profile_pools = build_train_test_profile_pools(ref_tables)
    print_profile_split_summary(ref_tables, profile_pools)

    # 归一化边界始终使用训练边界
    norm_bounds = get_train_split_norm_bounds(
        ev_range=train_ev_range,
        reference_tables=ref_tables,
        profile_pools=profile_pools,
    )

    rng = np.random.default_rng(seed_base)

    # ---------- 根据 test_mode 确定 EV 范围 ----------
    if test_mode == 'conservative':
        # EV 取训练范围的中间 1/3，如 [60,100] → [73, 87]
        span = train_ev_range[1] - train_ev_range[0]
        ev_low = int(train_ev_range[0] + span / 3)
        ev_high = int(train_ev_range[0] + 2 * span / 3)
    elif test_mode == 'interpolation':
        ev_low, ev_high = train_ev_range
    else:  # extrapolation
        if test_ev_range is None:
            test_ev_range = (30, 140)
        ev_low, ev_high = test_ev_range


    if test_mode == 'conservative':

        temp_all = profile_pools['temp_train_ids']
        avg_temps = ref_tables['avg_temps']
        temp_sorted = temp_all[np.argsort(avg_temps[temp_all])]
        n_cut = max(1, len(temp_sorted) // 3)
        temp_pool = temp_sorted[n_cut:-n_cut] if n_cut < len(temp_sorted) else temp_sorted

        pv_all = profile_pools['pv_train_ids']
        pv_energy = ref_tables['pv_total_energy']
        pv_sorted = pv_all[np.argsort(pv_energy[pv_all])]
        n_cut_pv = max(1, len(pv_sorted) // 3)
        pv_pool = pv_sorted[n_cut_pv:-n_cut_pv] if n_cut_pv < len(pv_sorted) else pv_sorted

    elif test_mode == 'interpolation':
        temp_pool = profile_pools['temp_train_ids']
        pv_pool = profile_pools['pv_train_ids']

    else:  # extrapolation
        temp_pool = profile_pools['temp_test_ids']
        pv_pool = profile_pools['pv_test_ids']
        seen_temp_values = ref_tables['avg_temps'][profile_pools['temp_train_ids']]
        seen_pv_values = ref_tables['pv_total_energy'][profile_pools['pv_train_ids']]
        temp_test_id_set = set(np.asarray(profile_pools['temp_test_ids'], dtype=int).tolist())
        pv_test_id_set = set(np.asarray(profile_pools['pv_test_ids'], dtype=int).tolist())

    samples = []
    used_keys = set()

    while len(samples) < n_samples:

        ev_count = int(rng.integers(ev_low, ev_high + 1))
        if test_mode == 'extrapolation' and train_ev_range[0] <= ev_count <= train_ev_range[1]:
            continue  # 外推模式跳过训练范围内的 EV

        tcl_idx = int(rng.choice(temp_pool))
        pv_idx = int(rng.choice(pv_pool))

        if test_mode == 'extrapolation':
            temp_value = float(ref_tables['avg_temps'][tcl_idx])
            pv_value = float(ref_tables['pv_total_energy'][pv_idx])
            if tcl_idx not in temp_test_id_set:
                raise ValueError("分布外测试抽到了非 temp_test_ids 的温度样本。")
            if pv_idx not in pv_test_id_set:
                raise ValueError("分布外测试抽到了非 pv_test_ids 的光伏样本。")
            if np.any(np.isclose(seen_temp_values, temp_value, atol=1e-12, rtol=0.0)):
                raise ValueError(f"分布外测试温度 {temp_value} 在训练集中出现过。")
            if np.any(np.isclose(seen_pv_values, pv_value, atol=1e-12, rtol=0.0)):
                raise ValueError(f"分布外测试 PV 能量 {pv_value} 在训练集中出现过。")

        unique_key = (int(ev_count), int(tcl_idx), int(pv_idx), int(tcl_count), int(pv_count))
        if unique_key in used_keys:
            continue
        used_keys.add(unique_key)

        agg, case_dict = build_case_from_physical_config(
            config={
                'ev_count': ev_count,
                'tcl_count': tcl_count,
                'pv_count': pv_count,
            },
            norm_bounds=norm_bounds,
            seed=seed_base + len(samples),
            discrete_rate=discrete_rate,
            T=T,
            temp_idx=tcl_idx,
            pv_idx=pv_idx,
        )

        samples.append({
            'model': case_dict['model'],
            'case': case_dict,
            'params': case_dict['params'],
            'config': {
                'model_id': len(samples),
                'ev_count': int(ev_count),
                'tcl_temp_ambient_avg': float(ref_tables['avg_temps'][tcl_idx]),
                'pv_energy': float(ref_tables['pv_total_energy'][pv_idx]),
                'tcl_count': int(tcl_count),
                'pv_count': int(pv_count),
                'test_mode': test_mode,
            },
            'norm_bounds': norm_bounds,
            'sample_type': 'test',
        })

    print(f"测试集({test_mode})成功采样 {len(samples)} 个场景")
    return samples


def sampled_models_same_theta_test(
    n_samples: int = 40,
    ev_count: int = 80,
    train_ev_range=(60, 100),
    tcl_idx: int = None,
    pv_idx: int = None,
    tcl_count: int = 80,
    pv_count: int = 30,
    seed_base: int = 5000,
    discrete_rate: float = 0.0,
    T: int = 24,
):
    ref_tables = build_profile_reference_tables(
        T=T
    )
    profile_pools = build_train_test_profile_pools(
        ref_tables
    )
    norm_bounds = get_train_split_norm_bounds(
        ev_range=train_ev_range,
        reference_tables=ref_tables,
        profile_pools=profile_pools,
    )
    rng=np.random.default_rng(seed_base)

    if tcl_idx is None:

        tcl_idx=int(
            rng.choice(
                profile_pools['temp_train_ids']
            )
        )
    if pv_idx is None:

        pv_idx=int(
            rng.choice(
                profile_pools['pv_train_ids']
            )
        )

    samples=[]
    for i in range(n_samples):

        agg, case_dict = build_case_from_physical_config(
            config={
                'ev_count': ev_count,
                'tcl_count': tcl_count,
                'pv_count': pv_count,
            },
            norm_bounds=norm_bounds,
            seed=seed_base+i,
            discrete_rate=discrete_rate,
            T=T,
            temp_idx=tcl_idx,
            pv_idx=pv_idx,
        )
        samples.append({
            'model':case_dict['model'],
            'case':case_dict,
            'params':case_dict['params'],
            'agg':agg,
            'xi_id':i,
            'config':{
                'model_id':i,
                'ev_count':int(ev_count),
                'tcl_temp_ambient_avg':float(ref_tables['avg_temps'][tcl_idx]),
                'pv_energy':float(ref_tables['pv_total_energy'][pv_idx]),
                'tcl_count':int(tcl_count),
                'pv_count':int(pv_count),
                },
            'norm_bounds':norm_bounds,
            'sample_type':'same_theta_test',
        })

    print(
        f"Same theta测试生成 {len(samples)} 个不同xi模型"
    )
    print(
        "固定theta:",
        {
            "ev_count":ev_count,
            "temp_idx":tcl_idx,
            "pv_idx":pv_idx,
            "tcl_count":tcl_count,
            "pv_count":pv_count,
        }
    )
    return samples



def inspect_xi_lists(samples, save_dir=None):

    import csv
    from pathlib import Path

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    if not samples:
        emit("[inspect_xi_lists] samples 为空，无可打印项。")
        return

    numeric_keys = [
        'n_ev', 'EV_ta', 'EV_td', 'EV_SOCa',
        'n_tcl', 'TCL_H', 'TCL_C', 'TCL_Qmax',
        'n_pv', 'PV_Ppanel', 'PV_eta', 'PV_energy',
        'theta_amb_mean',
    ]

    rows = []
    for s in samples:
        agg = s.get('agg')
        if agg is None:
            emit(f"[inspect_xi_lists] xi_id={s.get('xi_id')} 无 'agg'，"
                 f"跳过（旧样本未保留 Aggregator）。")
            continue

        ev = agg.EV_list
        tcl = agg.TCL_list
        pv = agg.PV_list

        rows.append({
            'xi_id': s.get('xi_id'),
            'n_ev': len(ev),
            'EV_ta':   np.mean([d['ta']   for d in ev]) if ev  else np.nan,
            'EV_td':   np.mean([d['td']   for d in ev]) if ev  else np.nan,
            'EV_SOCa': np.mean([d['SOCa'] for d in ev]) if ev  else np.nan,
            'n_tcl': len(tcl),
            'TCL_H':    np.mean([d['H']    for d in tcl]) if tcl else np.nan,
            'TCL_C':    np.mean([d['C']    for d in tcl]) if tcl else np.nan,
            'TCL_Qmax': np.mean([d['Qmax'] for d in tcl]) if tcl else np.nan,
            'n_pv': len(pv),
            'PV_Ppanel': np.mean([d['P_panel'] for d in pv]) if pv else np.nan,
            'PV_eta':    np.mean([d['eta_pv']  for d in pv]) if pv else np.nan,
            'PV_energy': float(np.sum(agg.pv_curve) * agg.deltaT),
            'theta_amb_mean': float(np.mean(agg.theta_amb)),
        })

    if not rows:
        emit("[inspect_xi_lists] 没有可用的 'agg'，未打印。")
        return

    # ---- 逐样本表格 ----
    emit("\n========== ξ 内部设备列表摘要 ==========")
    emit(
        f"{'xi':>3} | "
        f"{'EV: n  ta   td   SOCa':>22} | "
        f"{'TCL: n   H     C     Qmax':>26} | "
        f"{'PV: n  P     eta   E':>22} | {'amb':>5}"
    )
    emit("-" * 90)
    for r in rows:
        emit(
            f"{r['xi_id']:>3} | "
            f"{r['n_ev']:>3} {r['EV_ta']:>5.2f} {r['EV_td']:>5.2f} {r['EV_SOCa']:>5.2f} | "
            f"{r['n_tcl']:>3} {r['TCL_H']:>6.1f} {r['TCL_C']:>6.1f} {r['TCL_Qmax']:>7.1f} | "
            f"{r['n_pv']:>3} {r['PV_Ppanel']:>5.2f} {r['PV_eta']:>4.2f} {r['PV_energy']:>7.1f} | "
            f"{r['theta_amb_mean']:>5.2f}"
        )


    variation = []
    emit("\n跨样本变化情况（std≈0 固定 / std>0 在变）:")
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        std = float(np.nanstd(vals))
        flag = "变化" if std > 1e-9 else "固定"
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        emit(
            f"  {k:<16} std={std:.4e}  [{flag}]  "
            f"range=[{vmin:.4g}, {vmax:.4g}]"
        )
        variation.append({
            'field': k, 'std': std, 'status': flag,
            'min': vmin, 'max': vmax, 'n': len(rows),
        })
    emit("=" * 42)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        txt_path = save_dir / "xi_inspect_report.txt"
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        detail_cols = ['xi_id'] + numeric_keys
        detail_path = save_dir / "xi_device_summary.csv"
        with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(detail_cols)
            for r in rows:
                w.writerow([r[c] for c in detail_cols])

        # 3) 跨样本变化情况 CSV
        var_path = save_dir / "xi_variation_summary.csv"
        with open(var_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(['field', 'std', 'status', 'min', 'max', 'n'])
            for v in variation:
                w.writerow([v['field'], v['std'], v['status'],
                            v['min'], v['max'], v['n']])

        print(f"[INFO] Saved TXT:    {txt_path}")
        print(f"[INFO] Saved CSV:    {detail_path}")
        print(f"[INFO] Saved CSV:    {var_path}")


def get_average_config(sampled_models_list, T=24):

    # 计算平均值，config中的都是实际值
    ev_count_avg = int(round(np.mean([m['config']['ev_count'] for m in sampled_models_list])))
    tcl_temp_avg = np.mean([m['config']['tcl_temp_ambient_avg'] for m in sampled_models_list])
    pv_energy_avg = np.mean([m['config']['pv_energy'] for m in sampled_models_list])
    ta_mean_avg = float(np.mean([m['config'].get('ta_mean', 8.5) for m in sampled_models_list]))
    # wd_energy_avg = np.mean([m['config']['wd_energy'] for m in sampled_models_list])

    # 找最接近平均温度的索引
    temp_raw = np.load(f'{PROJECT_ROOT}\\data\\profiles_data\\profiles_data.npz')['temp_data']
    deltaT = 24 / T
    step = int(deltaT / (5 / 60))
    l = int(T * step)
    max_start_idx = len(temp_raw) - l

    stride_raw = max(1, l // 4)
    if max_start_idx > 0:
        valid_start_indices = np.arange(0, max_start_idx + 1, stride_raw, dtype=int)
    else:
        valid_start_indices = np.array([0], dtype=int)

    avg_temps = np.array([
        np.mean(temp_raw[idx: idx + l: step][:T])
        for idx in valid_start_indices
    ], dtype=float)
    temp_idx_avg = int(np.argmin(np.abs(avg_temps - tcl_temp_avg)))

    pv_data_raw = sio.loadmat(f'{PROJECT_ROOT}\\data\\profiles_data\\PV_samples.mat')['PV_samples']
    pv_step = int(pv_data_raw.shape[0] / T) or 1
    pv_energies = np.array([
        np.sum(pv_data_raw[::pv_step, idx][:T] / max(1e-5, np.max(pv_data_raw[::pv_step, idx][:T]))) * deltaT
        for idx in range(pv_data_raw.shape[1])
    ])
    pv_idx_avg = np.argmin(np.abs(pv_energies - pv_energy_avg))

    return {
        'ev_count': ev_count_avg,
        'tcl_temp': tcl_temp_avg,
        'pv_energy': pv_energy_avg,
        'ta_mean': ta_mean_avg,

        'temp_idx': int(temp_idx_avg),
        'pv_idx': int(pv_idx_avg),

    }


