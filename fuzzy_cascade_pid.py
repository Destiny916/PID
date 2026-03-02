"""
双容水箱模糊自适应串级PID控制系统
====================================
本程序实现了双容水箱液位控制系统的仿真，包含：
1. 双容水箱数学模型
2. 传统PID控制器
3. 模糊逻辑控制器
4. 模糊自适应串级PID控制器
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 双容水箱系统模型 ====================

@dataclass
class TankParameters:
    """水箱参数"""
    A1: float = 0.5      # 水箱1截面积 (m^2)
    A2: float = 0.5      # 水箱2截面积 (m^2)
    a1: float = 0.02     # 水箱1出水口截面积 (m^2)
    a2: float = 0.02     # 水箱2出水口截面积 (m^2)
    g: float = 9.81      # 重力加速度 (m/s^2)
    k_pump: float = 0.5  # 水泵增益
    
    # 线性化工作点
    h1_0: float = 0.5    # 水箱1工作点液位 (m)
    h2_0: float = 0.3    # 水箱2工作点液位 (m)


class DoubleTankSystem:
    """
    双容水箱系统
    水箱1 -> 水箱2 -> 出水
    """
    def __init__(self, params: TankParameters = None):
        self.params = params or TankParameters()
        self.h1 = 0.0  # 水箱1液位
        self.h2 = 0.0  # 水箱2液位
        self.reset()
        
    def reset(self):
        """重置系统状态"""
        self.h1 = self.params.h1_0
        self.h2 = self.params.h2_0
        
    def dynamics(self, h1: float, h2: float, u: float) -> Tuple[float, float]:
        """
        计算液位变化率
        :param h1: 水箱1液位
        :param h2: 水箱2液位
        :param u: 控制输入 (0-100%)
        :return: (dh1/dt, dh2/dt)
        """
        p = self.params
        
        # 水泵流量
        Q_in = p.k_pump * u / 100.0
        
        # 水箱1到水箱2的流量 (Torricelli定律)
        Q_12 = p.a1 * np.sqrt(2 * p.g * max(h1, 0))
        
        # 水箱2出水流量
        Q_out = p.a2 * np.sqrt(2 * p.g * max(h2, 0))
        
        # 液位变化率
        dh1_dt = (Q_in - Q_12) / p.A1
        dh2_dt = (Q_12 - Q_out) / p.A2
        
        return dh1_dt, dh2_dt
    
    def step(self, u: float, dt: float) -> Tuple[float, float]:
        """
        系统单步仿真 (使用RK4方法)
        :param u: 控制输入
        :param dt: 时间步长
        :return: (h1, h2)
        """
        # RK4积分
        k1_h1, k1_h2 = self.dynamics(self.h1, self.h2, u)
        k2_h1, k2_h2 = self.dynamics(
            self.h1 + 0.5*dt*k1_h1, 
            self.h2 + 0.5*dt*k1_h2, u
        )
        k3_h1, k3_h2 = self.dynamics(
            self.h1 + 0.5*dt*k2_h1, 
            self.h2 + 0.5*dt*k2_h2, u
        )
        k4_h1, k4_h2 = self.dynamics(
            self.h1 + dt*k3_h1, 
            self.h2 + dt*k3_h2, u
        )
        
        self.h1 += dt * (k1_h1 + 2*k2_h1 + 2*k3_h1 + k4_h1) / 6
        self.h2 += dt * (k1_h2 + 2*k2_h2 + 2*k3_h2 + k4_h2) / 6
        
        # 防止负液位
        self.h1 = max(self.h1, 0)
        self.h2 = max(self.h2, 0)
        
        return self.h1, self.h2
    
    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取线性化传递函数 (用于分析)
        G(s) = K / ((T1*s+1)(T2*s+1))
        :return: (num, den) 分子分母系数
        """
        p = self.params
        
        # 在工作点线性化
        R1 = np.sqrt(2 * p.h1_0 / p.g) / p.a1
        R2 = np.sqrt(2 * p.h2_0 / p.g) / p.a2
        
        T1 = p.A1 * R1
        T2 = p.A2 * R2
        K = p.k_pump * R2
        
        # 传递函数系数
        num = np.array([K])
        den = np.array([T1*T2, T1+T2, 1])
        
        return num, den


# ==================== 2. 传统PID控制器 ====================

class PIDController:
    """传统PID控制器"""
    def __init__(self, Kp: float = 10, Ki: float = 0.5, Kd: float = 0.5):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limit = (0, 100)  # 输出限幅 0-100%
        
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        计算PID输出
        :param setpoint: 设定值
        :param measurement: 测量值
        :param dt: 时间步长
        :return: 控制输出
        """
        error = setpoint - measurement
        
        # 比例项
        P = self.Kp * error
        
        # 积分项 (带抗积分饱和)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)
        I = self.Ki * self.integral
        
        # 微分项
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = self.Kd * derivative
        
        # 保存当前误差
        self.prev_error = error
        
        # 计算输出
        output = P + I + D
        
        # 输出限幅
        output = np.clip(output, *self.output_limit)
        
        return output


# ==================== 3. 模糊逻辑控制器 ====================

class FuzzyMembership:
    """模糊隶属度函数"""
    
    @staticmethod
    def trimf(x: float, params: List[float]) -> float:
        """三角隶属度函数"""
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0
    
    @staticmethod
    def gaussmf(x: float, mean: float, sigma: float) -> float:
        """高斯隶属度函数"""
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


class FuzzyController:
    """
    两输入一输出模糊控制器
    输入: 误差e, 误差变化率ec
    输出: 控制量调整
    """
    def __init__(self):
        self.e_range = (-1, 1)      # 误差论域
        self.ec_range = (-1, 1)     # 误差变化率论域
        self.u_range = (-1, 1)      # 输出论域
        
        # 模糊集合定义 (NB, NM, NS, ZO, PS, PM, PB)
        self.labels = ['NB', 'NM', 'NS', 'ZO', 'PS', 'PM', 'PB']
        
        # 输入e的隶属度函数参数 (三角函数中心点)
        self.e_mf_params = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
        self.ec_mf_params = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
        self.u_mf_params = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
        
        # 模糊规则表 (49条规则)
        self.rule_table = np.array([
            [0, 0, 0, 0, 1, 2, 3],   # NB
            [0, 0, 0, 1, 2, 3, 4],   # NM
            [0, 0, 1, 2, 3, 4, 5],   # NS
            [0, 1, 2, 3, 4, 5, 6],   # ZO
            [1, 2, 3, 4, 5, 6, 6],   # PS
            [2, 3, 4, 5, 6, 6, 6],   # PM
            [3, 4, 5, 6, 6, 6, 6],   # PB
        ])
        
    def fuzzify(self, x: float, mf_params: List[float]) -> List[float]:
        """模糊化: 计算x对各模糊集合的隶属度"""
        memberships = []
        for i, center in enumerate(mf_params):
            # 使用三角隶属度函数
            if i == 0:
                width = mf_params[1] - mf_params[0]
                params = [center - width, center, center + width]
            elif i == len(mf_params) - 1:
                width = mf_params[-1] - mf_params[-2]
                params = [center - width, center, center + width]
            else:
                width_left = center - mf_params[i-1]
                width_right = mf_params[i+1] - center
                params = [center - width_left, center, center + width_right]
            
            memberships.append(FuzzyMembership.trimf(x, params))
        return memberships
    
    def defuzzify(self, output_mf: List[float]) -> float:
        """重心法解模糊"""
        numerator = sum(m * c for m, c in zip(output_mf, self.u_mf_params))
        denominator = sum(output_mf)
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def compute(self, e: float, ec: float) -> float:
        """
        模糊推理计算
        :param e: 归一化误差
        :param ec: 归一化误差变化率
        :return: 归一化输出
        """
        # 限幅
        e = np.clip(e, *self.e_range)
        ec = np.clip(ec, *self.ec_range)
        
        # 模糊化
        e_mf = self.fuzzify(e, self.e_mf_params)
        ec_mf = self.fuzzify(ec, self.ec_mf_params)
        
        # 模糊推理 (Mamdani方法)
        output_activation = [0.0] * len(self.labels)
        
        for i, e_degree in enumerate(e_mf):
            for j, ec_degree in enumerate(ec_mf):
                if e_degree > 0 and ec_degree > 0:
                    # 取小运算
                    rule_strength = min(e_degree, ec_degree)
                    output_idx = self.rule_table[i, j]
                    # 取大运算 (聚合)
                    output_activation[output_idx] = max(
                        output_activation[output_idx], rule_strength
                    )
        
        # 解模糊
        output = self.defuzzify(output_activation)
        
        return output


# ==================== 4. 模糊自适应串级PID控制器 ====================

class FuzzyAdaptiveCascadePID:
    """
    模糊自适应串级PID控制器
    
    结构:
    - 主回路: 模糊自适应PID (控制水箱2液位)
    - 副回路: 传统PID (控制水箱1液位)
    
    模糊自适应: 根据误差和误差变化率在线调整PID参数
    """
    def __init__(self):
        # 主回路PID参数 (初始值)
        self.Kp0, self.Ki0, self.Kd0 = 10, 0.66, 1
        
        # 副回路PID参数 (固定)
        self.inner_pid = PIDController(Kp=11.5, Ki=1.55, Kd=0.45)
        
        # 三个模糊控制器分别调整Kp, Ki, Kd
        self.fuzzy_Kp = FuzzyController()
        self.fuzzy_Ki = FuzzyController()
        self.fuzzy_Kd = FuzzyController()
        
        # 调整因子
        self.Kp_scale = 5.0
        self.Ki_scale = 1.0
        self.Kd_scale = 2.0
        
        # 主回路PID状态
        self.integral = 0.0
        self.prev_error = 0.0
        
        # 参数历史记录
        self.Kp_history = []
        self.Ki_history = []
        self.Kd_history = []
        
    def reset(self):
        """重置控制器"""
        self.inner_pid.reset()
        self.integral = 0.0
        self.prev_error = 0.0
        self.Kp_history = []
        self.Ki_history = []
        self.Kd_history = []
        
    def compute(self, setpoint: float, h2: float, h1: float, dt: float) -> float:
        """
        计算串级控制输出
        :param setpoint: 水箱2目标液位
        :param h2: 水箱2实际液位
        :param h1: 水箱1实际液位
        :param dt: 时间步长
        :return: 水泵控制量 (0-100%)
        """
        # ========== 主回路: 模糊自适应PID ==========
        error = setpoint - h2
        
        # 归一化误差和误差变化率
        e_max = 0.5  # 假设最大误差0.5m
        ec_max = 0.5  # 假设最大误差变化率0.5m/s
        
        e_norm = np.clip(error / e_max, -1, 1)
        ec_norm = np.clip((error - self.prev_error) / dt / ec_max, -1, 1)
        
        # 模糊推理调整PID参数
        delta_Kp = self.fuzzy_Kp.compute(e_norm, ec_norm)
        delta_Ki = self.fuzzy_Ki.compute(e_norm, ec_norm)
        delta_Kd = self.fuzzy_Kd.compute(e_norm, ec_norm)
        
        # 计算实际PID参数
        Kp = self.Kp0 + self.Kp_scale * delta_Kp
        Ki = self.Ki0 + self.Ki_scale * delta_Ki
        Kd = self.Kd0 + self.Kd_scale * delta_Kd
        
        # 确保参数为正
        Kp = max(Kp, 0.1)
        Ki = max(Ki, 0.01)
        Kd = max(Kd, 0.01)
        
        # 记录参数
        self.Kp_history.append(Kp)
        self.Ki_history.append(Ki)
        self.Kd_history.append(Kd)
        
        # 主回路PID计算 (输出为水箱1的设定值)
        P = Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # 抗饱和
        I = Ki * self.integral
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = Kd * derivative
        
        # 水箱1设定值 (限幅在合理范围)
        h1_setpoint = np.clip(P + I + D, 0, 2.0)
        
        self.prev_error = error
        
        # ========== 副回路: 传统PID ==========
        u = self.inner_pid.compute(h1_setpoint, h1, dt)
        
        return u


# ==================== 5. 仿真和可视化 ====================

def simulate_system(controller_type: str = 'fuzzy_cascade', 
                   duration: float = 100,
                   dt: float = 0.1) -> dict:
    """
    运行仿真实验
    :param controller_type: 'pid', 'fuzzy', 'fuzzy_cascade'
    :param duration: 仿真时长 (秒)
    :param dt: 时间步长 (秒)
    :return: 仿真结果字典
    """
    # 创建系统
    tank = DoubleTankSystem()
    tank.reset()
    
    # 创建控制器 (使用相同参数便于公平对比)
    if controller_type == 'pid':
        controller = PIDController(Kp=10.0, Ki=0.66, Kd=1.0)
    elif controller_type == 'fuzzy':
        controller = FuzzyController()
        # 包装成PID形式使用
        pid_wrapper = PIDController(Kp=10.0, Ki=0.66, Kd=1.0)
    elif controller_type == 'fuzzy_cascade':
        controller = FuzzyAdaptiveCascadePID()
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # 仿真参数
    t = np.arange(0, duration, dt)
    n_steps = len(t)
    
    # 设定值 (阶跃变化) 0-100秒
    setpoint = np.ones(n_steps) * 0.3
    setpoint[int(0/dt):int(50/dt)] = 0.5  # 50-100秒: 0.5m
    setpoint[int(50/dt):int(100/dt)] = 0.5  # 50-100秒: 0.5m
    setpoint[int(100/dt):int(150/dt)] = 0.5  # 100-150秒: 0.4m
    setpoint[int(150/dt):] = 0.5  # 150秒后: 0.6m
    
    # 添加扰动
    disturbance = np.zeros(n_steps)
    disturbance[int(50/dt):int(65/dt)] = 0.02  # 50-65秒: 出水阀扰动
    
    # 记录数据
    h1_record = np.zeros(n_steps)
    h2_record = np.zeros(n_steps)
    u_record = np.zeros(n_steps)
    error_record = np.zeros(n_steps)
    
    # 运行仿真
    for i in range(n_steps):
        # 测量
        h1, h2 = tank.h1, tank.h2
        h1_record[i] = h1
        h2_record[i] = h2
        
        # 添加扰动影响
        if disturbance[i] > 0:
            tank.h2 = max(tank.h2 - disturbance[i] * dt, 0)
        
        # 计算控制量
        if controller_type == 'pid':
            u = controller.compute(setpoint[i], h2, dt)
        elif controller_type == 'fuzzy':
            # 简单模糊控制
            e = (setpoint[i] - h2) / 0.5
            ec = (e - (setpoint[max(0, i-1)] - h2_record[max(0, i-1)]) / 0.5) / dt if i > 0 else 0
            delta_u = controller.compute(e, ec) * 20
            u = pid_wrapper.compute(setpoint[i], h2, dt) + delta_u
            u = np.clip(u, 0, 100)
        else:  # fuzzy_cascade
            u = controller.compute(setpoint[i], h2, h1, dt)
        
        u_record[i] = u
        error_record[i] = setpoint[i] - h2
        
        # 系统单步
        tank.step(u, dt)
    
    results = {
        't': t,
        'setpoint': setpoint,
        'h1': h1_record,
        'h2': h2_record,
        'u': u_record,
        'error': error_record,
        'controller_type': controller_type
    }
    
    # 添加模糊自适应参数历史
    if controller_type == 'fuzzy_cascade':
        results['Kp_history'] = controller.Kp_history
        results['Ki_history'] = controller.Ki_history
        results['Kd_history'] = controller.Kd_history
    
    return results


def calculate_metrics(results: dict) -> dict:
    """计算控制性能指标"""
    error = results['error']
    u = results['u']
    dt = results['t'][1] - results['t'][0]
    
    # 忽略前10秒的瞬态
    steady_idx = int(10 / dt)
    
    metrics = {
        'IAE': np.sum(np.abs(error[steady_idx:])) * dt,  # 积分绝对误差
        'ISE': np.sum(error[steady_idx:]**2) * dt,       # 积分平方误差
        'ITAE': np.sum(results['t'][steady_idx:] * np.abs(error[steady_idx:])) * dt,
        'Max_Overshoot': np.max(results['h2'] - results['setpoint']) if np.max(results['h2'] - results['setpoint']) > 0 else 0,
        'Control_Effort': np.sum(np.abs(np.diff(u)))     # 控制量变化总量
    }
    
    return metrics


def plot_results(results_list: List[dict], save_path: str = None):
    """
    绘制仿真结果对比
    :param results_list: 多个控制器的仿真结果列表
    :param save_path: 保存路径
    """
    n_controllers = len(results_list)
    
    # 创建图形
    if n_controllers == 1 and results_list[0]['controller_type'] == 'fuzzy_cascade':
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['b', 'r', 'g', 'm']
    labels_map = {
        'pid': '传统PID',
        'fuzzy': '模糊PID',
        'fuzzy_cascade': '模糊自适应串级PID'
    }
    
    # 图1: 液位响应
    ax1 = axes[0, 0]
    for i, res in enumerate(results_list):
        ax1.plot(res['t'], res['h2'], colors[i], label=labels_map[res['controller_type']], linewidth=1.5)
    ax1.plot(results_list[0]['t'], results_list[0]['setpoint'], 'k--', label='设定值', linewidth=1)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('液位 (m)')
    ax1.set_title('水箱2液位响应')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 控制量
    ax2 = axes[0, 1]
    for i, res in enumerate(results_list):
        ax2.plot(res['t'], res['u'], colors[i], label=labels_map[res['controller_type']], linewidth=1.5)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('控制量 (%)')
    ax2.set_title('水泵控制量')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 误差
    ax3 = axes[1, 0]
    for i, res in enumerate(results_list):
        ax3.plot(res['t'], res['error'], colors[i], label=labels_map[res['controller_type']], linewidth=1.5)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('误差 (m)')
    ax3.set_title('跟踪误差')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 水箱1液位
    ax4 = axes[1, 1]
    for i, res in enumerate(results_list):
        ax4.plot(res['t'], res['h1'], colors[i], label=labels_map[res['controller_type']], linewidth=1.5)
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('液位 (m)')
    ax4.set_title('水箱1液位')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 如果是模糊自适应串级PID，显示参数自适应过程
    if n_controllers == 1 and results_list[0]['controller_type'] == 'fuzzy_cascade':
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        res = results_list[0]
        t = res['t']
        
        # Kp变化
        axes2[0, 0].plot(t[:len(res['Kp_history'])], res['Kp_history'], 'b', linewidth=1.5)
        axes2[0, 0].set_xlabel('时间 (s)')
        axes2[0, 0].set_ylabel('Kp')
        axes2[0, 0].set_title('比例系数自适应调整')
        axes2[0, 0].grid(True, alpha=0.3)
        
        # Ki变化
        axes2[0, 1].plot(t[:len(res['Ki_history'])], res['Ki_history'], 'r', linewidth=1.5)
        axes2[0, 1].set_xlabel('时间 (s)')
        axes2[0, 1].set_ylabel('Ki')
        axes2[0, 1].set_title('积分系数自适应调整')
        axes2[0, 1].grid(True, alpha=0.3)
        
        # Kd变化
        axes2[1, 0].plot(t[:len(res['Kd_history'])], res['Kd_history'], 'g', linewidth=1.5)
        axes2[1, 0].set_xlabel('时间 (s)')
        axes2[1, 0].set_ylabel('Kd')
        axes2[1, 0].set_title('微分系数自适应调整')
        axes2[1, 0].grid(True, alpha=0.3)
        
        # 相平面 (e vs ec)
        e = res['error']
        ec = np.gradient(e, t[1]-t[0])
        axes2[1, 1].plot(e, ec, 'm', linewidth=1)
        axes2[1, 1].set_xlabel('误差 e')
        axes2[1, 1].set_ylabel('误差变化率 ec')
        axes2[1, 1].set_title('误差相轨迹')
        axes2[1, 1].grid(True, alpha=0.3)
        axes2[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes2[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_adaptive.png'), dpi=150)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()
    
    return fig


def print_comparison_table(results_list: List[dict]):
    """打印性能对比表"""
    print("\n" + "="*80)
    print("控制性能对比")
    print("="*80)
    print(f"{'控制器类型':<20} {'IAE':<12} {'ISE':<12} {'ITAE':<12} {'超调量':<12} {'控制量变化':<12}")
    print("-"*80)
    
    labels_map = {
        'pid': '传统PID',
        'fuzzy': '模糊PID',
        'fuzzy_cascade': '模糊自适应串级PID'
    }
    
    for res in results_list:
        metrics = calculate_metrics(res)
        print(f"{labels_map[res['controller_type']]:<20} "
              f"{metrics['IAE']:<12.4f} "
              f"{metrics['ISE']:<12.6f} "
              f"{metrics['ITAE']:<12.4f} "
              f"{metrics['Max_Overshoot']:<12.4f} "
              f"{metrics['Control_Effort']:<12.4f}")
    print("="*80)


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("双容水箱模糊自适应串级PID控制系统仿真")
    print("="*50)
    
    # 运行三种控制器的仿真 (0-100秒)
    print("\n[1/3] 运行传统PID控制器仿真...")
    results_pid = simulate_system('pid', duration=100, dt=0.1)
    
    print("[2/3] 运行模糊PID控制器仿真...")
    results_fuzzy = simulate_system('fuzzy', duration=100, dt=0.1)
    
    print("[3/3] 运行模糊自适应串级PID控制器仿真...")
    results_fuzzy_cascade = simulate_system('fuzzy_cascade', duration=100, dt=0.1)
    
    # 绘制对比结果
    print("\n绘制结果对比图...")
    plot_results([results_pid, results_fuzzy, results_fuzzy_cascade], 
                 save_path='comparison_results.png')
    
    # 打印性能对比
    print_comparison_table([results_pid, results_fuzzy, results_fuzzy_cascade])
    
    print("\n仿真完成!")
    print("结果已保存为: comparison_results.png")
