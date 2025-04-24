import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from scipy.optimize import root
from tqdm import tqdm

# k = random.randint(3,7)
k = 5


def Zzhou(a, b):
    def integrand(t):
        result = 1/3 * a * t ** 3 + 1/2 * b * t ** 2
        return result

    # 反转函数
    def integrand_1(t):
        result = -integrand(t)
        return result

    def find_t0():
        target = 2 * np.pi/k
        # 使用数值优化来找到
        solution = root(lambda t: integrand(t) - target, 1.5)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) - target, 1.5)
        return solution.x[0]

    def find_t1():
        t_1 = 2 * np.pi/k
        return t_1

    def find_t2():
        # 使用数值优化来找到
        solution = root(lambda t: 2 * np.pi - abs(t - integrand(t)) - (k-1) * t, 0)
        return solution.x[0]

    # 定义判断是否相遇的函数
    def check_encounter(x1, y1, x2, y2):
        return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 < 0.2

    t0 = find_t0()
    t1 = find_t1()
    t2 = find_t2()
    theta = [0] * k
    if k % 2 == 0:
        if t0 < t1:
            for i in range(k):
                theta[i] = i * 2 * np.pi / k
        else:
            for i in range(k):
                if i % 2 == 0:
                    theta[i] = i * 2 * np.pi / k
                else:
                    theta[i] = (i - 1) * 2 * np.pi / k
    else:
        if t0 < t2:
            for i in range(k):
                theta[i] = i * 2 * np.pi / k
        else:
            for i in range(k):
                if i % 2 == 0:
                    theta[i] = t2 * i
                else:
                    theta[i] = t2 * (i-1)

    dt = 0.001
    bus_theta = np.random.rand() * 2 * np.pi
    robot_thetas = theta
    bus_speed = integrand(0)
    robot_speed = 1
    total_time = 0
    con = 0

    if k % 2 == 0:
        if t0 < t1:
            max = find_t0()
            while True:
                bus_theta += bus_speed * dt
                bus_x = np.cos(bus_theta)
                bus_y = np.sin(bus_theta)
                for i in range(k):
                    robot_x = np.cos(robot_thetas[i])
                    robot_y = np.sin(robot_thetas[i])
                    if check_encounter(bus_x, bus_y, robot_x, robot_y):
                        con = 1
                # 更新所需的时间
                total_time += dt
                # 更新bus速度
                bus_speed = integrand(total_time)
                if con == 1:
                    break
        else:
            max = find_t1()
            while True:
                # 更新bus和robot的位置
                bus_theta += bus_speed * dt
                for i in range(k):
                    if i % 2 == 0:
                        robot_thetas[i] += dt
                    else:
                        robot_thetas[i] -= dt
                # 计算位置
                bus_x = np.cos(bus_theta)
                bus_y = np.sin(bus_theta)
                for i in range(k):
                    robot_x = np.cos(robot_thetas[i])
                    robot_y = np.sin(robot_thetas[i])
                    if check_encounter(bus_x, bus_y, robot_x, robot_y):
                        con = 1
                # 更新所需的时间
                total_time += dt
                # 更新bus速度
                bus_speed = integrand(total_time)
                if con == 1:
                    break
    else:
        if t0 < t2:
            max = find_t0()
            while True:
                bus_theta += bus_speed * dt
                bus_x = np.cos(bus_theta)
                bus_y = np.sin(bus_theta)
                for i in range(k):
                    robot_x = np.cos(robot_thetas[i])
                    robot_y = np.sin(robot_thetas[i])
                    if check_encounter(bus_x, bus_y, robot_x, robot_y):
                        con = 1
                # 更新所需的时间
                total_time += dt
                # 更新bus速度
                bus_speed = integrand(total_time)
                if con == 1:
                    break
        else:
            max = find_t2()
            while True:
                # 更新bus和robot的位置
                bus_theta += bus_speed * dt
                for i in range(k):
                    if i <k-1:
                        if i % 2 == 0:
                            robot_thetas[i] += robot_speed * dt
                        else:
                            robot_thetas[i] -= robot_speed * dt
                    else:
                        robot_thetas[i] -= robot_speed * dt
                # 计算位置
                bus_x = np.cos(bus_theta)
                bus_y = np.sin(bus_theta)
                # 检查是否相遇
                for i in range(k):
                    robot_x = np.cos(robot_thetas[i])
                    robot_y = np.sin(robot_thetas[i])
                    if check_encounter(bus_x, bus_y, robot_x, robot_y):
                        con = 1
                # 更新所需的时间
                total_time += dt
                # 更新bus速度
                bus_speed = integrand(total_time)
                if con == 1:
                    break
    return total_time, max


# 定义 a 和 b 的取值范围
a_values = np.arange(0.05, 1.01, 0.01)
b_values = np.arange(0.05, 1.01, 0.01)

# 创建网格
A, B = np.meshgrid(a_values, b_values)

# 初始化 Z1 和 Z2
Z1 = np.zeros_like(A)
Z2 = np.zeros_like(B)

# 使用 tqdm 显示进度条
for i in tqdm(range(A.shape[0]), desc="Calculating Zzhou results", ncols=100):
    for j in range(A.shape[1]):
        # 获取对应的 a 和 b
        a, b = A[i, j], B[i, j]

        # 计算 Z1 和 Z2
        Z1[i, j], Z2[i, j] = Zzhou(a, b)

# 创建三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个结果的表面图
surf1 = ax.plot_surface(A, B, Z1, cmap='viridis', alpha=0.6, label='total_time')

# 绘制第二个结果的表面图
surf2 = ax.plot_surface(A, B, Z2, color='purple', alpha=0.6, label='max')


from matplotlib.lines import Line2D
# 添加图例
legend_elements = [
    Line2D([0], [0], color='green', lw=4, label='The experimental results'),
    Line2D([0], [0], color='purple', lw=4, label='The Upper Bound')
]
ax.legend(handles=legend_elements)

# 设置轴标签
ax.set_xlabel('The coefficient of the quadratic term (a)')
ax.set_ylabel('The coefficient of the first degree term (b)')
ax.set_zlabel('time (z)')

# 显示图形
plt.show()