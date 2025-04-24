import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from tqdm import tqdm
# 定义计算结果的函数
def Zzhou(a, b):
    def integrand(t):
        result = 1/3 * a * t ** 3 + 1/2 * b * t ** 2
        return result
    # 反转函数
    def integrand_1(t):
        result = -integrand(t)
        return result

    def find_t0():
        target = 2 * np.pi
        # 使用数值优化来找到
        solution = root(lambda t: integrand(t) + t - target, 3)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) + t - target, 3)
        return solution.x[0]

    def find_t1():
        fla = 1
        target = 2 * np.pi
        # 使用数值优化来找到t0
        solution = root(lambda t: integrand(t) + t - target, 3)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) + t - target, 3)
            fla = 2
        t_0 = solution.x[0]

        # 设置新的目标值
        target1 = 4 * np.pi

        # 使用数值优化来找到解
        if fla == 1:
            solution1 = root(lambda t: integrand(t) + t - target1, t_0 + 1)
            if not solution1.success or solution1.x[0] < 0:
                solution = root(lambda t: integrand_1(t) + t - target, 3)
                solution1 = root(lambda t: integrand_1(t) + t - 2 * target, 3)
        else:
            solution1 = root(lambda t: integrand_1(t) + t - target1, t_0 + 1)
        t_1 = solution1.x[0]
        return t_1

    # 计算bus走一周的时间
    def find_cycle():
        fla = 1
        target = 2 * np.pi
        # 使用数值优化来找到t0
        solution = root(lambda t: integrand(t) + t - target, 3)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) + t - target, 3)
            fla = 2
        t_0 = solution.x[0]

        # 设置新的目标值
        if fla == 1:
            target1 = 2 * np.pi + 2 * t_0
        else:
            target1 = 2 * np.pi + 2 * t_0
        # 使用数值优化来找到解
        if fla == 1:
            solution1 = root(lambda t: integrand(t) + t - target1, t_0 + 1)
            solution2 = root(lambda t: integrand(t) - target, t_0 + 1)
            if not solution1.success or solution1.x[0] < 0:
                solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
        else:
            solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
        return solution2.x[0]

    # 计算追击时间
    def find_chase():
        target = 2 * np.pi
        # 使用数值优化来找到
        solution = root(lambda t: integrand(t) - t - target, 10)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: -integrand(t) + t - target, 10)
            if not solution.success or solution.x[0] < 0:
                solution = root(lambda t: integrand_1(t) - t - target, 10)
                if not solution.success or solution.x[0] < 0:
                    solution = root(lambda t: -integrand_1(t) + t - target, 10)
        return solution.x[0]

    def cal_s0():
        fla = 1
        target = 2 * np.pi
        # 使用数值优化来找到t0
        solution = root(lambda t: integrand(t) + t - target, 3)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) + t - target, 3)
            fla = 2
        t_0 = solution.x[0]

        # 设置新的目标值
        target1 = 4 * np.pi
        # 使用数值优化来找到解
        if fla == 1:
            solution1 = root(lambda t: integrand(t) + t - target1, t_0 + 1)
            if not solution1.success or solution1.x[0] < 0:
                solution = root(lambda t: integrand_1(t) + t - target, 3)
                solution1 = root(lambda t: integrand_1(t) + t - target - 2 * solution.x[0], t_0 + 1)
        else:
            solution1 = root(lambda t: integrand_1(t) + t - target1, t_0 + 1)
        e = solution.x[0]
        x0 = 2 * np.pi - e
        return x0 / e

    def cal_s1():
        fla = 1
        target = 2 * np.pi
        # 使用数值优化来找到t0
        solution = root(lambda t: integrand(t) + t - target, 3)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) + t - target, 3)
            fla = 2
        t_0 = solution.x[0]

        # 设置新的目标值
        target1 = 4 * np.pi

        # 使用数值优化来找到解
        if fla == 1:
            solution1 = root(lambda t: integrand(t) + t - target1, t_0 + 1)
            if not solution1.success or solution1.x[0] < 0:
                solution = root(lambda t: integrand_1(t) + t - target, 3)
                solution1 = root(lambda t: integrand_1(t) + t - 2 * target, 3)
        else:
            solution1 = root(lambda t: integrand_1(t) + t - target1, t_0 + 1)
        t_0 = solution.x[0]
        t_1 = solution1.x[0]
        x_1 = 2 * np.pi - (t_1 - t_0)
        return x_1 / (t_1 - t_0)

    def cal_cycle():
        fla = 1
        target = 2 * np.pi
        # 使用数值优化来找到t0
        solution = root(lambda t: integrand(t) + t - target, 3)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) + t - target, 3)
            fla = 2
        t_0 = solution.x[0]

        # 设置新的目标值
        target1 = 4 * np.pi
        # 使用数值优化来找到解
        if fla == 1:
            solution1 = root(lambda t: integrand(t) + t - target1, t_0 + 1)
            solution2 = root(lambda t: integrand(t) - target, t_0 + 1)
            if not solution1.success or solution1.x[0] < 0:
                solution = root(lambda t: integrand_1(t) + t - target, 3)
                solution1 = root(lambda t: integrand_1(t) + t - target - target1, t_0 + 1)
                solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
        else:
            solution1 = root(lambda t: integrand_1(t) + t - target1, t_0 + 1)
            solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
        cycle_t = solution2.x[0] - solution.x[0]
        cycle_x = solution.x[0]
        return cycle_x / cycle_t

    # 定义判断是否相遇的函数
    def check_encounter(x1, y1, x2, y2):
        return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 < 0.2

    # 初始化参数
    dt = 0.001
    bus_theta = np.random.rand() * 2 * np.pi
    robot_theta = np.random.rand() * 2 * np.pi
    flag = np.random.choice([1, -1])
    bus_speed = integrand(0)
    robot_speed = 1 * flag
    bus_positions = []
    robot_positions = []
    total_time = 0
    s0 = cal_s0()
    s1 = cal_s1()
    s2 = cal_cycle()
    t0 = find_t0()
    q = 0

    # 模拟运动过程
    if s2 * ((1 + s0) / (1 + s1)) >= 1:
        max = find_cycle()
        while True:
            # 更新bus和robot的位置
            bus_theta += bus_speed * dt

            # 计算位置
            bus_x = np.cos(bus_theta)
            bus_y = np.sin(bus_theta)
            robot_x = np.cos(robot_theta)
            robot_y = np.sin(robot_theta)

            # 存储位置数据
            bus_positions.append((bus_x, bus_y))
            robot_positions.append((robot_x, robot_y))

            # 检查是否相遇
            if check_encounter(bus_x, bus_y, robot_x, robot_y):
                break

            # 更新所需的时间
            total_time += dt

            # 更新bus速度
            bus_speed = integrand(total_time)
    else:
        if 3 * s0 * s2 + s0 + s2 > 1:
            max = find_t1()
            while True:
                # 更新bus和robot的位置
                bus_theta += bus_speed * dt
                robot_theta += robot_speed * dt

                # 计算位置
                bus_x = np.cos(bus_theta)
                bus_y = np.sin(bus_theta)
                robot_x = np.cos(robot_theta)
                robot_y = np.sin(robot_theta)

                # 存储位置数据
                bus_positions.append((bus_x, bus_y))
                robot_positions.append((robot_x, robot_y))

                # 检查是否相遇
                if check_encounter(bus_x, bus_y, robot_x, robot_y):
                    break

                # 更新所需的时间
                total_time += dt
                if total_time == t0 and q == 0:
                    robot_speed = robot_speed * (-1)
                    q = 1
                # 更新bus速度
                bus_speed = integrand(total_time)

        else:
            max = find_chase()
            while True:
                # 更新bus和robot的位置
                bus_theta += bus_speed * dt
                robot_theta += robot_speed * dt

                # 计算位置
                bus_x = np.cos(bus_theta)
                bus_y = np.sin(bus_theta)
                robot_x = np.cos(robot_theta)
                robot_y = np.sin(robot_theta)

                # 存储位置数据
                bus_positions.append((bus_x, bus_y))
                robot_positions.append((robot_x, robot_y))

                # 检查是否相遇
                if check_encounter(bus_x, bus_y, robot_x, robot_y):
                    break

                # 更新所需的时间
                total_time += dt

                # 更新bus速度
                bus_speed = integrand(total_time)
    return total_time,max


# 定义 a 和 b 的取值范围

a_values = np.arange(0.01, 1.01, 0.01)
b_values = np.arange(0.01, 1.01, 0.01)
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
