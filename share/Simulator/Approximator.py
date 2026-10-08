# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:53:21 2025

@author: 11421
"""
import copy

from pyomo.environ import *
import numpy as np
# import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import time
import random
import json
from datetime import datetime
import os

class ErrorCalculator():
    def __init__(self,
                 original_model: Dict,
                 A_hat,
                 b_hat=None,
                 is_epigraph=False,
                 solver: str = 'gurobi', ):
        """
        Args:
            original_model: 包含以下键的字典
                - 'variables': 变量字典 {变量名: cvxpy变量}
                - 'constraints': 约束列表
                - 'agg_var_name': 'P_AGG'
            solver: 优化求解器
        """
        #初始化实力基础属性
        self.is_epigraph = is_epigraph
        self.solver_str = solver

        #保留原始模型字典和克隆后的核心模型
        self.original_model_dict = original_model
        self.baseline = original_model.get('baseline')
        self.original_model = original_model['model'].clone()
        #获取模型维度（基于模型内的var_proj属性）
        self.dim = len(self.original_model.var_proj)  # 维度
        # 初始化模型映射表（新增）
        self.model_original_mapping = {}  # 模型ID到模型基准的映射
        self.active_model_id = None  # 当前活跃的模型ID
        # 将初始模型添加到映射表中
        if 'config' in original_model and 'model_id' in original_model['config']:
            model_id = original_model['config']['model_id']
            self.model_original_mapping[model_id] = {
                'model': self.original_model,
                'baseline': self.baseline,
                'config': original_model['config']
            }
            self.active_model_id = model_id
        else:
            # 如果没有config，创建一个默认的
            model_id = 0
            if 'config' not in original_model:
                original_model['config'] = {}
            original_model['config']['model_id'] = model_id

            self.model_original_mapping[model_id] = {
                'model': self.original_model,
                'baseline': self.baseline,
                'config': original_model['config']
            }
            self.active_model_id = model_id
        #初始化模型专属参数和目标函数
        self.init_model_attributes()

        self.solver = SolverFactory(solver, tee=True)
        # self.solver.options['OutputFlag'] = 0

        # self.solver.options['constr_viol_tol'] = 1e-6
        # self.solver.options['TimeLimit'] = 60
        self.approx_model = ConcreteModel()
        self.approx_model.var_proj = Var(range(self.dim), initialize=0.0)
        self.approx_model.x_org = Param(range(self.dim), initialize=0, mutable=True)
        self.approx_model.direction = Param(range(self.dim), initialize=0, mutable=True)
        self.approx_model.min_direction = (
            Objective(expr=sum(self.approx_model.direction[j] * self.approx_model.var_proj[j]
                               for j in range(self.dim)),
                      sense=minimize))
        self.approx_model.min_error = (
            Objective(expr=sum((self.approx_model.var_proj[j] - self.approx_model.x_org[j]) ** 2
                               for j in range(self.dim)),
                      sense=minimize))
        # 初始化近似多面体
        n_cons = A_hat.shape[0]
        self.approx_model.A = Param(range(n_cons), range(self.dim), mutable=True)
        self.approx_model.b = Param(range(n_cons), mutable=True)
        self._initialization(A_hat, b_hat)


        def matrix_constraint_rule(model, i):
            return sum(model.A[i, j] * model.var_proj[j]
                       for j in range(self.dim)) <= model.b[i]

        self.approx_model.constraints = Constraint(range(n_cons), rule=matrix_constraint_rule)

        FR_params = original_model.get('FR_params', None)
        if is_epigraph and (FR_params is not None) and (FR_params['A_fr'] is not None) and (
                FR_params['b_fr'] is not None):
            A_fr = FR_params['A_fr']
            b_fr = FR_params['b_fr']

            def epigraph_fr_rule(model, i):
                return sum(A_fr[i, j] * model.var_proj[j]
                           for j in range(self.dim - 1)) <= b_fr[i]

            self.approx_model.epigraph_fr = Constraint(range(A_fr.shape[0]), rule=epigraph_fr_rule)
        self.cvx_solver = SolverFactory('gurobi', tee=True)

        self.params = {'active_tol': 1e-5}
        self.training_history = {
            'feas': [],
            'opt': []
        }
        self._iter = 0

    def init_model_attributes(self):
        self.original_model.x_apx = Param(
            range(self.dim),
            initialize=0,
            mutable=True
        )

        self.original_model.direction = Param(
            range(self.dim),
            initialize=0,
            mutable=True
        )
        self.original_model.min_direction = (
            Objective(expr=sum(self.original_model.direction[j] * self.original_model.var_proj[j] for j in range(self.dim)),
            sense=minimize
        ))

        self.original_model.min_error = (
            Objective(expr=sum((self.original_model.var_proj[j] - self.original_model.x_apx[j]) ** 2 for j in range(self.dim)),
            sense=minimize
        ))

    def _initialization(self, A_hat, b_hat=None):
        """初始化近似多面体结构"""
        if self.is_epigraph:
            self._cal_max_obj()
            self.original_model.fmax = Param(initialize=self.fmax, mutable=True)
            self.original_model.f_ub = Constraint(
                expr=self.original_model.var_proj[self.dim - 1] <= self.original_model.fmax)
        self.A_hat = A_hat
        n_constraints = self.A_hat.shape[0]
        for i in range(n_constraints):
            for j in range(self.dim):
                self.approx_model.A[i, j].value = self.A_hat[i, j]
        if b_hat is None:
            self.b_hat = np.zeros(n_constraints)
            for i in range(n_constraints):
                self.b_hat[i] = self.A_hat[i] @ self.optimize_direction(-self.A_hat[i])
                self.approx_model.b[i].value = self.b_hat[i]
        else:
            self.b_hat = b_hat
            for i in range(n_constraints):
                self.approx_model.b[i].value = self.b_hat[i]



    def _cal_max_obj(self):
        self.original_model.min_direction.deactivate()
        self.original_model.min_error.deactivate()
        self.original_model.max_obj.activate()
        if hasattr(self.original_model, 'fmax'):
            self.original_model.fmax.set_value(np.inf)
        solver = SolverFactory(self.original_model_dict.get('fmax_solver', self.solver_str))
        results = solver.solve(self.original_model,)
        if results.solver.termination_condition == TerminationCondition.optimal:
            self.fmax = value(self.original_model.obj)
        self.original_model.max_obj.deactivate()

    def configure(self, **kwargs):
        """更新配置参数"""
        self.params.update(kwargs)

    def project(self, target: np.ndarray, to_approx: bool = False) -> Optional[np.ndarray]:
        if to_approx:
            self.approx_model.min_direction.deactivate()
            self.approx_model.min_error.activate()
            for j in range(self.dim):
                self.approx_model.x_org[j].value = target[j]
            try:
                results = self.cvx_solver.solve(self.approx_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.approx_model.var_proj[j].value for j in range(self.dim)])
                print(('project', to_approx, results.solver.termination_condition, target))
            except:
                print(self._iter)
                return None
        else:
            self.original_model.min_direction.deactivate()
            self.original_model.min_error.activate()
            for j in range(self.dim):
                self.original_model.x_apx[j].value = target[j]
            try:
                results = self.solver.solve(self.original_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.original_model.var_proj[j].value for j in range(self.dim)])
                # else:
                #     print(1)
                print(('project', to_approx, results.solver.termination_condition, target))
            except:
                print(self._iter)
                return None
        return None

    def optimize_direction(self,
                           direction: np.ndarray,
                           in_approx: bool = False) -> Optional[np.ndarray]:
        if in_approx:
            self.approx_model.min_error.deactivate()
            self.approx_model.min_direction.activate()
            for j in range(self.dim):
                self.approx_model.direction[j].value = direction[j]
            # print((self.A_hat,self.b_hat))
            try:
                results = self.cvx_solver.solve(self.approx_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.approx_model.var_proj[j].value for j in range(self.dim)])
                print(('direction', in_approx, results.solver.termination_condition, direction))
            except:
                print(self._iter)
                return None
        else:
            self.original_model.min_error.deactivate()
            self.original_model.min_direction.activate()
            for j in range(self.dim):
                self.original_model.direction[j].value = direction[j]
            try:
                results = self.solver.solve(self.original_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.original_model.var_proj[j].value for j in range(self.dim)])

                print(('direction', in_approx, results.solver.termination_condition, direction))
            except:
                print(self._iter)
                return None
        return None

    def _find_active(self, x_apx):
        # 1. 寻找激活约束
        residuals = self.A_hat @ x_apx - self.b_hat
        active_indices = np.where(np.abs(residuals) < self.params['active_tol'])[0]
        # if 28 in active_indices:
        #     print(1)
        return active_indices
#测试模型，之后ec，输入fullnet关键参数得到ab，调用ec中的更新多边形，调calculate，提取误差，
    def calculate(self, n_cal=1, cal_feas=True, cal_opt=True):
        feas_results = [
            {'is_valid': False, 'error':0, 'active_indices': np.array([], dtype=int), 'x_org': np.zeros(self.dim)} for
            _ in range(n_cal)]
        opt_results = [
            {'is_valid': False, 'error': 0, 'active_indices': np.array([], dtype=int), 'x_org': np.zeros(self.dim)} for
            _ in range(n_cal)]

        for i in range(n_cal):
            # 随机优化方向
            c = np.random.randn(self.dim)
            if self.is_epigraph:
                c[-1] = 1.

            if cal_feas:
                # 可行性误差路径
                x_apx = self.optimize_direction(c, in_approx=True)
                x_org = self.project(x_apx) if x_apx is not None else None
                if x_apx is not None and x_org is not None:
                    e_feas = np.sum((x_apx - x_org) ** 2)
                    feas_results[i]['error'] = e_feas
                    if e_feas > self.params['feas_tol']:
                        feas_results[i]['is_valid'] = True
                        feas_results[i]['active_indices'] = self._find_active(x_apx)
                        feas_results[i]['x_org'] = x_org
                else:
                    print(self._iter)
            if cal_opt:
                # 最优性误差路径
                x_org = self.optimize_direction(c)
                x_apx = self.project(x_org, to_approx=True) if x_org is not None else None
                if x_org is not None and x_apx is not None:
                    e_opt = np.sum((x_apx - x_org) ** 2)
                    opt_results[i]['error'] = e_opt
                    if e_opt > self.params['opt_tol']:
                        opt_results[i]['is_valid'] = True
                        opt_results[i]['active_indices'] = self._find_active(x_apx)
                        opt_results[i]['x_org'] = x_org
                        # print(self.A_hat[28,3])
                else:
                    print(self._iter)
        # 记录误差
        self.training_history['feas'].append(np.mean([feas_results[i]['error'] for i in range(n_cal)]))
        self.training_history['opt'].append(np.mean([opt_results[i]['error'] for i in range(n_cal)]))
        # print(f"\n==== Iter {self._iter} debug feas raw errors ====")
        # for idx, res in enumerate(feas_results):
        #     print(f"sample{idx}: error={res['error']}, is_valid={res['is_valid']}")
        # print(f"history append mean: {np.mean([r['error'] for r in feas_results])}")
        self._iter += 1
        return feas_results, opt_results

    def update_polytope(self, A_hat=None, b_hat=None):
        n_constraints = self.A_hat.shape[0]
        if A_hat is not None:
            self.A_hat = A_hat
            for i in range(n_constraints):
                for j in range(self.dim):
                    self.approx_model.A[i, j].value = self.A_hat[i, j]
        if b_hat is not None:
            self.b_hat = b_hat
            for i in range(n_constraints):
                self.approx_model.b[i].value = self.b_hat[i]

    def update_model(self, new_original_model: Dict,):
        """
        更新模型，同步重新绑定参数和目标函数
        参数:
            new_original_model: 包含模型信息的字典
            skip_clone: 如果为 True，直接使用传入的模型（假设已经 clone 过）
        """
        if 'model' not in new_original_model:
            raise ValueError("新模型字典必须包含 'model' 键，对应核心模型对象")
        if 'config' not in new_original_model or 'model_id' not in new_original_model['config']:
            raise ValueError("新模型字典必须包含 'config.model_id'，用于绑定专属基准")
        current_model_id= new_original_model['config']['model_id']
        # 【优化】支持跳过克隆，直接使用已克隆的模型
        current_model= new_original_model['model'].clone()
        current_original_dict = {
        'model': current_model,
        'baseline': new_original_model.get('baseline'),
        'config': new_original_model['config']
        }
        self.model_original_mapping[current_model_id] =current_original_dict
        self._switch_to_active_model(current_model_id)
        print(f"模型 {current_model_id} 专属基准已注册，当前活跃模型切换为 {self.active_model_id}")



    def _switch_to_active_model(self, target_model_id: int):

        if target_model_id not in self.model_original_mapping:
            raise ValueError(f"模型 {target_model_id} 未注册专属基准，请先调用 update_model() 初始化")
        target_original_dict = self.model_original_mapping[target_model_id]
        self.active_model_id = target_model_id
        self.original_model_dict = target_original_dict
        self.baseline = target_original_dict['baseline']
        self.original_model = target_original_dict['model']
        self.dim = len(self.original_model.var_proj) if hasattr(self.original_model, 'var_proj') else 10

        self.init_model_attributes()


    def update_parameters(self, param_dict):
        """通用参数更新函数
        Args:
            model: Pyomo模型对象
            param_dict: 参数字典，支持格式：
                - 标量：直接赋值
                - 一维：np数组或字典{索引: 值}
                - 二维：np数组或字典{(i,j): 值}
        """
        for param_name, value in param_dict.items():
            param = getattr(self.original_model, param_name)

            # 标量参数处理
            if not param.is_indexed():
                param.set_value(float(value))
                continue

            # 多维参数处理
            indices = param.index_set()

            # 一维参数 (索引格式为单个元素)
            if all(not isinstance(idx, tuple) for idx in indices):
                if isinstance(value, np.ndarray):
                    if value.ndim != 1:
                        raise ValueError(f"参数 {param_name} 需要一维数组")
                    for i, idx in enumerate(indices):
                        param[idx].set_value(value[i])
                elif isinstance(value, dict):
                    for idx, val in value.items():
                        param[idx].set_value(val)
                else:
                    raise ValueError(f"不支持的类型: {type(value)}")

            # 二维参数 (索引格式为元组)
            else:
                if isinstance(value, np.ndarray):
                    if value.ndim != 2:
                        raise ValueError(f"参数 {param_name} 需要二维数组")
                    for (i, j), idx in zip(np.ndindex(value.shape), indices):
                        param[idx].set_value(value[i, j])
                elif isinstance(value, dict):
                    for (i, j), val in value.items():
                        param[i, j].set_value(val)
                else:
                    raise ValueError(f"不支持的类型: {type(value)}")
        if self.is_epigraph:
            self._cal_max_obj()
            self.original_model.fmax.set_value(self.fmax)

    def copy(self):
        ec = ErrorCalculator(original_model=self.original_model_dict,
                             A_hat=self.A_hat,
                             b_hat=self.b_hat,
                             is_epigraph=self.is_epigraph,
                             solver=self.solver_str)
        ec.configure(**self.params)
        return ec


class PreTrainNet(nn.Module):
    def __init__(self, A_init, b_init, is_epigraph=False, device='cpu'):
        super().__init__()
        self.nrows, self.dim_x = A_init.shape
        self.is_epigraph = is_epigraph
        if is_epigraph:
            self.A_net = nn.Sequential(
                ZeroInitLinear(0, (self.nrows - 1) * self.dim_x, init_bias=np.array(A_init[:-1]).flatten(),
                               device=device)
            )
        else:
            self.A_net = nn.Sequential(
                ZeroInitLinear(0, self.nrows * self.dim_x, init_bias=np.array(A_init).flatten(), device=device)
            )
        self.b_net = nn.Sequential(
            ZeroInitLinear(0, self.nrows, init_bias=np.array(b_init), device=device)
        )

    def forward(self):
        # 生成A矩阵
        A_flat = self.A_net(torch.empty(0))
        if self.is_epigraph:
            A = A_flat.view(-1, self.nrows - 1, self.dim_x)  # shape: (batch_size, 2, 2)
            final_row = torch.tensor(np.hstack([np.zeros(self.dim_x - 1), [1.0]]), dtype=torch.float32,
                                     device=A.device).reshape(-1, 1, self.dim_x)
            A = torch.cat([A, final_row], dim=1)
        else:
            A = A_flat.view(-1, self.nrows, self.dim_x)  # shape: (batch_size, 2, 2)
        # 归一化A的行向量
        row_norms = torch.norm(A, dim=2, keepdim=True)  # shape: (batch_size, 2, 1)
        A_normalized = A / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero

        # 生成b向量
        b = self.b_net(torch.empty(0))  # shape: (batch_size, 2)

        # 归一化b向量（除以对应的A行范数）
        b_normalized = b / (row_norms.squeeze(2) + 1e-8)  # shape: (batch_size, 2)

        return A_normalized, b_normalized


class BiasNet(nn.Module):
    def __init__(self, dim_theta, b_init, n_hidden=64, device='cpu'):
        super().__init__()
        self.nrows = b_init.shape[0]
        if dim_theta:
            self.b_net = nn.Sequential(
                nn.Linear(dim_theta, n_hidden),
                nn.ReLU(),
                ZeroInitLinear(n_hidden, self.nrows, init_bias=np.array(b_init), device=device)
            )
        else:
            self.b_net = nn.Sequential(
                ZeroInitLinear(0, self.nrows, init_bias=np.array(b_init), device=device)
            )

    def forward(self, delta_theta):
        b = self.b_net(delta_theta)
        return b

class FullNet(nn.Module):
    def __init__(self, dim_theta, A_init, b_init, is_epigraph=False, n_hidden=64, device='cpu'):
        super().__init__()
        self.nrows, self.dim_x = A_init.shape
        self.is_epigraph = is_epigraph
        if is_epigraph:
            self.A_net = nn.Sequential(
                nn.Linear(dim_theta, n_hidden),
                nn.ReLU(),
                ZeroInitLinear(n_hidden, (self.nrows - 1) * self.dim_x, init_bias=np.array(A_init[:-1]).flatten(),
                               device=device)
            )
        else:
            self.A_net = nn.Sequential(
                nn.Linear(dim_theta, n_hidden),
                nn.ReLU(),
                ZeroInitLinear(n_hidden, self.nrows * self.dim_x, init_bias=np.array(A_init).flatten(), device=device)
            )
        self.b_net = nn.Sequential(
            nn.Linear(dim_theta, n_hidden),
            nn.ReLU(),
            ZeroInitLinear(n_hidden, self.nrows, init_bias=np.array(b_init), device=device)
        )

    def forward(self, delta_theta):
        # 生成A矩阵
        A_flat = self.A_net(delta_theta)
        if self.is_epigraph:
            A = A_flat.view(-1, self.nrows - 1, self.dim_x)  # shape: (batch_size, 2, 2)
            batch_size = A.shape[0]
            final_row = torch.tensor(np.hstack([np.zeros(self.dim_x - 1), [1.0]]), dtype=torch.float32,
                                     device=A.device).reshape(-1, 1, self.dim_x)
            final_row_batch = final_row.repeat(batch_size, 1, 1)  # 扩展为[batch_size,1,2]
            A = torch.cat([A, final_row_batch], dim=1)
        else:
            A = A_flat.view(-1, self.nrows, self.dim_x)  # shape: (batch_size, 2, 2)
        # 归一化A的行向量
        row_norms = torch.norm(A, dim=2, keepdim=True)  # shape: (batch_size, 2, 1)
        A_normalized = A / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero

        # 生成b向量
        b = self.b_net(delta_theta)  # shape: (batch_size, 2)

        # 归一化b向量（除以对应的A行范数）
        b_normalized = b / (row_norms.squeeze(2) + 1e-8)  # shape: (batch_size, 2)

        return A_normalized, b_normalized


class ZeroInitLinear(nn.Module):
    """自定义全连接层：权重初始为0，bias初始化为给定值"""

    def __init__(self, in_features, out_features, init_bias, device='cpu'):
        super().__init__()
        self.device = device
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32, device=device))

    def forward(self, x):
        return torch.matmul(x.to(self.device), self.weight.t()) + self.bias


# 向量方法计算损失函数
def compute_loss(A, b, batch_data):
    # 提取数据并转换为张量
    batch_size = len(batch_data)
    if batch_size == 0:
        return torch.tensor(0.0, device=A.device)

    n_cal = len(batch_data[0])
    m, n = A.shape[-2], A.shape[-1]
    device = A.device

    # 初始化存储张量
    x_org_list = []
    is_valid_list = []
    active_indices_coords = []

    # 遍历batch_data提取信息
    for batch_idx, batch in enumerate(batch_data):
        x_batch = []
        is_valid_batch = []
        for ncal_idx, data in enumerate(batch):
            # x_org处理
            x_org = torch.from_numpy(data['x_org']).float().to(device)
            x_batch.append(x_org)

            # is_valid处理
            is_valid_batch.append(data['is_valid'])

            # active_indices坐标收集
            active_indices = torch.from_numpy(data['active_indices']).long().to(device)
            for idx in active_indices:
                active_indices_coords.append((batch_idx, ncal_idx, idx.item()))

        x_org_list.append(torch.stack(x_batch))
        is_valid_list.append(torch.tensor(is_valid_batch, device=device))

    # 构造核心张量
    x_org = torch.stack(x_org_list)  # (batch_size, n_cal, n)
    is_valid = torch.stack(is_valid_list).bool()  # (batch_size, n_cal)

    # 生成active_indices的0-1掩码矩阵
    mask = torch.zeros(batch_size, n_cal, m, device=device)
    if active_indices_coords:
        batch_idx, ncal_idx, active_idx = zip(*active_indices_coords)
        batch_idx = torch.tensor(batch_idx, device=device)
        ncal_idx = torch.tensor(ncal_idx, device=device)
        active_idx = torch.tensor(active_idx, device=device)
        mask[batch_idx, ncal_idx, active_idx] = 1.0

    # 计算预测值 (向量化实现)
    A_expanded = A.unsqueeze(1)  # (batch, 1, m, n)
    x_org_expanded = x_org.unsqueeze(3)  # (batch, n_cal, n, 1)
    mask_expanded = mask.unsqueeze(3)  # (batch, n_cal, m, 1)

    # 矩阵乘法 (batch, n_cal, m, n) × (batch, n_cal, n, 1) → (batch, n_cal, m, 1)
    pred = torch.matmul(A_expanded * mask_expanded, x_org_expanded).squeeze(-1)

    # 计算残差
    b_expanded = b.unsqueeze(1).expand(-1, n_cal, -1)  # (batch, n_cal, m)
    b_masked = b_expanded * mask
    residual = pred - b_masked

    # 计算平方损失并应用有效性掩码
    squared_loss = residual.pow(2) * mask  # (batch, n_cal, m)
    loss_per_sample = squared_loss.sum(dim=-1)  # (batch, n_cal)
    valid_loss = loss_per_sample * is_valid.float()

    # 计算平均损失
    total_loss = valid_loss.sum()
    num_valid = is_valid.sum().float()
    # print((num_valid,total_loss))
    # return total_loss / num_valid.clamp(min=1e-6)
    return total_loss / num_valid.clamp(min=1)


# 循环计算损失函数
# def compute_loss(A, b, batch_data):
#     """
#     计算损失函数
#
#     参数:
#         A: tensor, shape (batch_size, m, n)
#         b: tensor, shape (batch_size, m)
#         batch_data: list of lists of dictionaries
#
#     返回:
#         loss: scalar tensor
#     """
#     batch_size = len(batch_data)
#     total_loss = 0.0
#     valid_count = 0
#     device = A.device
#     for i in range(batch_size):  # 遍历每个batch
#         for j in range(len(batch_data[i])):  # 遍历每个n_cal元素
#             data = batch_data[i][j]
#             # if not data['is_valid']:
#             #     continue  # 跳过无效数据
#
#             active_indices = data['active_indices']
#             x_org = torch.tensor(data['x_org'], dtype=torch.float32, device=device)
#
#             # 计算A_sub * x_org
#             A_subx = torch.matmul(A[i, active_indices, :], x_org)  # shape (len(active_indices),)
#
#             # 获取对应的b值
#             b_sub = b[i, active_indices]  # shape (len(active_indices),)
#
#             # 计算平方误差
#             error = torch.sum((A_subx - b_sub)  ** 2)
#             total_loss += error
#             valid_count += 1
#
#     if valid_count == 0:
#         return torch.tensor(0.0, device=device,requires_grad=True)  # 避免除以零
#
#     # 计算平均损失
#     loss = total_loss / valid_count
#     return loss

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            error_calculator: ErrorCalculator,
            compute_loss: Callable,
            params: Optional[Dict] = None,
            device='cpu',
    ):
        """
        极简版训练器
        - configure() 更新配置
        - initialize() 初始化组件
        - train() 执行训练
        """
        self.model = model
        self.error_calculator = error_calculator
        self.compute_loss = compute_loss
        self.loss_history = []
        self.grad_history = []
        # 默认参数配置
        self.default_params = {
            "optimizer": "SGD",
            "lr": 5e-1,
            "scheduler": None,
            "batch_size": 1,
            "n_cal": 1,
            "cal_feas": True,
            "cal_opt": True,
            "feas_tol": 1e-8,
            "opt_tol": 1e-8,
            "call_interval": 20,
            "training_callback": None,
            "iterations_per_model" :10
        }
        self.params = {**self.default_params, **(params or {})}



    def configure(self, **kwargs):
        """动态更新配置参数"""
        self.params.update(kwargs)

    def save_history(self, save_dir, phase_name='train', config_info=None):
        """保存训练历史，用时间戳+阶段名自动命名，避免覆盖

        Args:
            save_dir: 保存目录路径
            phase_name: 阶段名称（如 'pretrain', 'phase1', 'phase2'）
            config_info: 可选的配置信息字典，会一起保存
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{phase_name}_{timestamp}"

        data = {
            'phase': phase_name,
            'timestamp': timestamp,
            'loss_history': [(float(f), float(o)) for f, o in self.loss_history],
            'grad_history': [float(g) for g in self.grad_history],
            'feas_error': [float(e) for e in self.error_calculator.training_history['feas']],
            'opt_error': [float(e) for e in self.error_calculator.training_history['opt']],
            'total_iterations': self.error_calculator._iter,
        }
        if config_info:
            data['config'] = config_info

        save_path = os.path.join(save_dir, f"{filename}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"训练历史已保存: {save_path}")
        return save_path

    def initialize(self):
        """一键初始化所有组件"""
        # 配置误差计算器
        self.error_calculator.configure(
            feas_tol=self.params["feas_tol"],
            opt_tol=self.params["opt_tol"],
            cal_feas=self.params["cal_feas"],
            cal_opt=self.params["cal_opt"]
        )
        # 优化器设置
        opt_type = self.params["optimizer"].lower()
        base_lr = self.params["lr"]  # 默认学习率
        lr_A = self.params.get("lr_A", base_lr)  # 优先使用专用学习率
        lr_b = self.params.get("lr_b", base_lr)
        # 自动构建参数组
        param_groups = []
        #判断模型是否包含A_net属性
        if hasattr(self.model, 'A_net'):
            param_groups.append({'params': self.model.A_net.parameters(), 'lr': lr_A})
        if hasattr(self.model, 'b_net'):
            param_groups.append({'params': self.model.b_net.parameters(), 'lr': lr_b})
        # 如果没有找到任何子网络则报错
        if not param_groups:
            raise RuntimeError("模型中没有找到A_net或b_net")
        # 创建优化器
        if opt_type == "sgd":
            self.optimizer = optim.SGD(param_groups)
            # self.optimizer = optim.SGD(param_groups, momentum=0.95, nesterov=True)
        elif opt_type == "adam":
            self.optimizer = optim.Adam(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        # 初始化调度器
        self.scheduler = None
        if self.params["scheduler"]:
            config = self.params["scheduler"]
            if config["type"] == "StepLR":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.get("step_size", 100),
                    gamma=config.get("gamma", 0.95)
                )
            elif config["type"] == "MultiStepLR":
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=config.get("milestones", [400]),  # 提取里程碑列表，默认 fallback 设为你想要的 [400, 1500]
                    gamma=config.get("gamma", 0.1)  # 提取衰减率，默认设为 0.1
                )

    def train(self, n_train: int = 1, params_data=None, parallel: bool = False, train_model_list: List[Dict] = None):
    # def train(self, n_train: int = 1, params_data=None, parallel: bool = False,):
        """训练入口
                参数:
                    n_train: 训练迭代次数
                    params_data: 训练数据
                    parallel: 是否使用并行训练，默认是串行
                """
        if not parallel:
            self._train_serial(n_train=n_train,params_data=params_data,train_model_list=train_model_list)
            # self._train_serial(n_train=n_train, params_data=params_data)
        else:
            self._train_parallel(n_train=n_train, params_data=params_data,train_model_list=train_model_list)


    def _train_serial(self, n_train: int = None, params_data=None, train_model_list=None):
        """通用训练入口，支持每个模型连续训练多次"""
        if not isinstance(self.model, (PreTrainNet, BiasNet, FullNet)):
            warnings.warn(f"暂不支持 {type(self.model).__name__} 模型的训练，已跳过")
            return

        i_iter = 0
        # if isinstance(self.model, PreTrainNet):
        #     dataloader = [None]
        #     for epoch in range(n_train + 1):
        #         for batch_data in dataloader:
        if isinstance(self.model, PreTrainNet):
            params_data['dataloader'] = [None]
            for epoch in range(n_train+1):
                for batch_data in params_data['dataloader']:
                    # start_time = time.time()  # 开始计时
                    # 梯度清零
                    self.optimizer.zero_grad()
                    # 前向传播
                    if isinstance(self.model, PreTrainNet):
                        A_pred, b_pred = self.model()
                        A_pred = A_pred.repeat(self.params["batch_size"], 1, 1)
                        b_pred = b_pred.repeat(self.params["batch_size"], 1)
                    elif isinstance(self.model, BiasNet):
                        b_pred = self.model(self._combine_batch(batch_data))
                        A_pred = self.params["A_pretrained"].repeat(self.params["batch_size"], 1, 1)
                    else:
                        A_pred, b_pred = self.model(self._combine_batch(batch_data))

                    # 批量处理
                    feas_results, opt_results = [], []
                    for i in range(self.params["batch_size"]):
                        # 更新多面体参数
                        self.error_calculator.update_polytope(
                            A_hat=A_pred[i].detach().cpu().numpy(),
                            b_hat=b_pred[i].detach().cpu().numpy()
                        )
                        # 更新模型参数
                        if batch_data is not None:
                            params_upd = {}
                            for name, data in batch_data.items():
                                params_upd[name] = params_data['params_dict'][name]['initial_value'] + data[
                                    i].detach().cpu().numpy()

                            self.error_calculator.update_parameters(params_upd)

                        # 计算结果
                        f, o = self.error_calculator.calculate(
                            n_cal=self.params["n_cal"],
                            cal_feas=self.params["cal_feas"],
                            cal_opt=self.params["cal_opt"]
                        )
                        feas_results.append(f)
                        opt_results.append(o)

                    # # 计算双重损失
                    loss_feas = self.compute_loss(A_pred, b_pred, feas_results)
                    loss_opt = self.params['rate_opt_feas'] * self.compute_loss(A_pred, b_pred, opt_results)
                    #
                    # # 反向传播
                    # loss_feas.backward(retain_graph=True)
                    # loss_opt.backward()

                    # 计算双重损失
                    loss_total = loss_feas + loss_opt

                    # 反向传播
                    loss_total.backward()
                    # 将梯度强行裁剪到一个很小的安全范围内，防止地雷爆炸炸飞模型
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.grad_history.append(np.max(np.abs(self.model.A_net[0].bias.grad.detach().cpu().numpy())))
                    # 参数更新
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()

                    # 记录损失
                    self.loss_history.append((loss_feas.item(), loss_opt.item()))

                    # 触发回调
                    if i_iter % self.params["call_interval"] == 0:
                        # print(np.max(self.grad_history[-min(5,len(self.grad_history)):]))
                        # print(self.loss_history[-1])
                        callback = self.params["training_callback"]
                        if callback:
                            callback(self.error_calculator, i_iter)
                    i_iter += 1
        else:
            MASTER_SEED = 1000
            current_epoch = 0  # 新增：记录当前抽取的轮次
            # 1. 初始化索引列表并进行第一次随机打乱 (Epoch 0)
            shuffle_rng = np.random.default_rng(MASTER_SEED + current_epoch)
            model_indices = list(range(len(train_model_list)))
            shuffle_rng.shuffle(model_indices)
            idx_in_list = 0  # 列表指针
            while i_iter < n_train:
                if idx_in_list >= len(model_indices):
                    idx_in_list = 0
                    current_epoch += 1  # 轮次加 1
                    epoch_rng = np.random.default_rng(MASTER_SEED + current_epoch)
                    epoch_rng.shuffle(model_indices)
                    print(f"所有模型已轮抽一遍。开启新的一轮(Epoch {current_epoch})固定随机训练。当前进度: {i_iter}/{n_train}")
                # 2. 从打乱后的索引列表中取出一个模型索引
                model_idx = model_indices[idx_in_list]
                model_item = train_model_list[model_idx]

                #切换模型 - 使用预先 clone 的模型
                switch_model_dict = {
                    'model': model_item['model'],
                    'baseline': model_item.get('baseline'),
                    'config': model_item['config']
                }
                try:
                    self.error_calculator.update_model(switch_model_dict,)
                except Exception as e:
                    # 失败了也移动到下一个随机索引
                    idx_in_list = (idx_in_list + 1) % len(model_indices)
                    if idx_in_list == 0: random.shuffle(model_indices)
                    continue
                # A_pred, b_pred = None, None  # 用于记录最后一次迭代的输出
                # batch_data = None
                # 当前模型连续训练
                n=self.params.get("iterations_per_model", 10)
                for _ in range(n):
                    if i_iter >= n_train:
                        break
                    dataloader = model_item['params']['dataloader']
                    batch_data = next(iter(dataloader))
                    device = next(self.model.parameters()).device
                    if batch_data is not None:
                        for key in batch_data:
                            if isinstance(batch_data[key], torch.Tensor):
                                batch_data[key] = batch_data[key].to(device)

                    current_batch_size = 1

                    # 梯度清零
                    self.optimizer.zero_grad()

                    #前向传播
                    if isinstance(self.model, PreTrainNet):
                        A_pred, b_pred = self.model()
                        A_pred = A_pred.repeat(current_batch_size, 1, 1)
                        b_pred = b_pred.repeat(current_batch_size, 1)

                    elif isinstance(self.model, BiasNet):
                        if batch_data is None:
                            continue
                        b_pred = self.model(self._combine_batch(batch_data))
                        A_pred = self.params["A_pretrained"].repeat(current_batch_size, 1, 1)

                    else:  # FullNet
                        if batch_data is None:
                            continue
                        A_pred, b_pred = self.model(self._combine_batch(batch_data))


                    # 误差计算
                    feas_results, opt_results = [], []

                    for i in range(current_batch_size):
                        self.error_calculator.update_polytope(
                            A_hat=A_pred[i].detach().cpu().numpy(),
                            b_hat=b_pred[i].detach().cpu().numpy()
                        )
                        f, o = self.error_calculator.calculate(
                            n_cal=self.params["n_cal"],
                            cal_feas=self.params["cal_feas"],
                            cal_opt=self.params["cal_opt"]
                        )
                        feas_results.append(f)
                        opt_results.append(o)

                    # loss
                    loss_feas = self.compute_loss(A_pred, b_pred, feas_results)
                    # loss_feas = torch.tensor(0.0, device=A_pred.device)
                    loss_opt = self.params['rate_opt_feas'] * self.compute_loss(A_pred, b_pred, opt_results)
                    loss_total = loss_feas + loss_opt

                    #  backward
                    loss_total.backward()

                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.grad_history.append(np.max(np.abs(self.model.A_net[0].bias.grad.detach().cpu().numpy())))

                    self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    self.loss_history.append((loss_feas.item(), loss_opt.item()))

                    if i_iter % self.params["call_interval"] == 0:
                        # print(np.max(self.grad_history[-min(5,len(self.grad_history)):]))
                        # print(self.loss_history[-1])
                        callback = self.params.get("training_callback")
                        if callback:
                            callback(self.error_calculator, i_iter)
                    i_iter += 1

                idx_in_list += 1  # 移动到下一个随机模型

    def _train_parallel(self, n_train: int = None, params_data=None):
        """通用训练入口"""
        if not isinstance(self.model, (PreTrainNet, BiasNet, FullNet)):
            warnings.warn(f"暂不支持 {type(self.model).__name__} 模型的训练，已跳过")
            return
        # ================ 初始化并行化设施 ================
        batch_size = self.params["batch_size"]#1
        error_calculators = []
        for _ in range(batch_size):
            ec = self.error_calculator.copy()
            error_calculators.append(ec)
            # 1. 预创建多个独立的error_calculator实例

        def process_task(args):
            i, A_hat, b_hat, params_upd = args
            # 直接使用预先生成的第i个实例
            ec = error_calculators[i]
            ec.update_polytope(A_hat, b_hat)
            if params_upd is not None:
                ec.update_parameters(params_upd)
            f, o = ec.calculate(
                n_cal=self.params["n_cal"],
                cal_feas=self.params["cal_feas"],
                cal_opt=self.params["cal_opt"]
            )
            return (f, o)

        i_iter = 0
        if isinstance(self.model, PreTrainNet):
            params_data['dataloader'] = [None]
        for epoch in range(n_train):
            for batch_data in params_data['dataloader']:
                # start_time = time.time()  # 开始计时
                device = next(self.model.parameters()).device
                if batch_data is not None:
                    for key in batch_data:
                        if isinstance(batch_data[key], torch.Tensor):
                            batch_data[key] = batch_data[key].to(device)

                self.optimizer.zero_grad()

                if isinstance(self.model, PreTrainNet):
                    A_pred, b_pred = self.model()
                    A_pred = A_pred.repeat(self.params["batch_size"], 1, 1)
                    b_pred = b_pred.repeat(self.params["batch_size"], 1)
                elif isinstance(self.model, BiasNet):
                    b_pred = self.model(self._combine_batch(batch_data))
                    A_pred = self.params["A_pretrained"].repeat(self.params["batch_size"], 1, 1)
                else:
                    A_pred, b_pred = self.model(self._combine_batch(batch_data))

                with ThreadPoolExecutor(max_workers=10) as executor:
                    # 生成任务参数时绑定索引i
                    params_dict_list = [None for _ in range(self.params["batch_size"])]
                    for i in range(self.params["batch_size"]):
                        params_upd = {}
                        if batch_data is not None:
                            for name, data in batch_data.items():
                                params_upd[name] = params_data['params_dict'][name]['initial_value'] + data[
                                    i].detach().cpu().numpy()
                        params_dict_list[i] = params_upd
                    task_args = [
                        (i, A_pred[i].detach().cpu().numpy(),
                         b_pred[i].detach().cpu().numpy(),
                         params_dict_list[i])
                        for i in range(self.params["batch_size"])
                    ]

                    # 提交任务并获取结果
                    results = executor.map(process_task, task_args)
                    feas_results, opt_results = zip(*results)

                # 后续的损失计算和反向传播保持不变
                loss_feas = self.compute_loss(A_pred, b_pred, feas_results)
                loss_opt = self.params['rate_opt_feas'] * self.compute_loss(A_pred, b_pred, opt_results)
                #
                # loss_feas.backward(retain_graph=True)
                # loss_opt.backward()

                # 计算双重损失
                loss_total = loss_feas + loss_opt

                # 反向传播
                loss_total.backward()

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.loss_history.append((loss_feas.item(), loss_opt.item()))

                if i_iter % self.params["call_interval"] == 0:
                    if self.params["training_callback"]:
                        self.params["training_callback"](error_calculators[0], i_iter)
                i_iter += 1
                # end_time = time.time()
                # elapsed = end_time - start_time  # 计算耗时（秒）
                # print(f"迭代{i_iter}，耗时: {elapsed:.4f} 秒")

    def _combine_batch(self, batch_dict):
        '''
        将多个不同形状的批处理张量合并成一个统一的二维张量
        保持批次结构不变，同时将各种形状的特征统一为适合神经网络处理的二维格式
        '''
        if not batch_dict:
            return torch.empty((1, 0))
        features = []
        for key in batch_dict.keys():
            tensor = batch_dict[key]
            # 展平除批次维度外的所有维度（例如将 (5,2,3) 展平为 (5,6)）
            flattened = tensor.view(tensor.size(0), -1)
            features.append(flattened)
        return torch.cat(features, dim=1)

def pyomo_params_to_numpy(model):
    params_dict = {}
    param_count = 0

    def _get_matrix_dimensions(matrix_param):
        rows = set()
        cols = set()
        for (i, j) in matrix_param.index_set():
            rows.add(i)
            cols.add(j)
        return len(rows), len(cols)

    for param in model.component_objects(Param, descend_into=True):
        if not param.mutable:
            continue
        param_name = param.name
        if not param.is_indexed():  # 0维参数（标量）
            value = param.value
            size = 1
            param_count += size
        else:  # 多维参数
            # 获取参数维度
            dim = param.dim()
            if dim == 1:  # 1维参数（向量）
                values = [param[idx].value for idx in param.index_set()]
                value = np.array(values)
                size = len(value)
                param_count += size
            elif dim == 2:  # 2维参数（矩阵）
                nrow, ncol = _get_matrix_dimensions(param)

                # 创建矩阵并填充值
                value = np.zeros((nrow, ncol))
                for count, idx in enumerate(param.index_set()):
                    i, j = count // ncol, count % ncol
                    value[i, j] = param[idx[0], idx[1]].value
                size = (nrow, ncol)
                param_count += size[0] * size[1]
            else:
                raise ValueError(f"参数 {param.name} 的维度 {dim} 超过2维，本代码不支持")
        # 将参数信息存入字典
        params_dict[param_name] = {
            'initial_value': value,
            'size': size
        }
    return params_dict, param_count
