import numpy as np
from scipy.optimize import root
import random
import pandas as pd

# 创建一个空的DataFrame
df = pd.DataFrame(columns=['速度函数','算法', 'time_1','time_2','time_3','time_4'])  # 根据数据结构修改列名




for i in range(100):
    print(i)
    # 生成随机数组作为多项式的系数
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    random_number = random.randint(2, 3)
    for k in range(random_number):
        a[k] = round(random.uniform(0, 0.1), 2)

    def generate_polynomial():
        global a, random_number
        polynomial = ""
        for m in range(random_number):
            if m == 0:
                polynomial += f"{a[m]}"
            else:
                polynomial += f"+{a[m]}t^{m}"
        return polynomial

    # 定义bus速度随时间变化的函数
    def integrand(t):
        global a, random_number
        result = 0
        for j in range(random_number):  # 计算多项式函数在 t 处的值
            result += 1 / (j + 1) * a[j] * t ** (j + 1)
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
                solution1 = root(lambda t: integrand_1(t) + t - target1, 3)
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
                print("无解或负解，调转函数（次轮）")
                solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
        else:
            solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
        return solution2.x[0]


    # 计算追击时间
    def find_chase():
        target = 2 * np.pi
        solution = root(lambda t: abs(integrand(t)- t) - target, 10)
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


    # 计算后半程平均速度
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

    def cal_4():
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
        target2 = np.pi
        # 使用数值优化来找到解
        if fla == 1:
            solution1 = root(lambda t: integrand(t) + t - target1, t_0 + 1)
            solution2 = root(lambda t: integrand(t) - target, t_0 + 1)
            solution_4 = root(lambda t: integrand(t + t_0) + t - target2, 3)
            if not solution1.success or solution1.x[0] < 0:
                solution = root(lambda t: integrand_1(t) + t - target, 3)
                solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
                solution_4 = root(lambda t: integrand_1(t) + t - target2, 3)
        else:
            solution2 = root(lambda t: integrand_1(t) - target, t_0 + 1)
            solution_4 = root(lambda t: integrand_1(t) + t - target2, 3)

        max =solution2.x[0] + solution_4.x[0]
        return max

    def generate_ui(u):
        ui = np.pi * (1 - np.e ** (-0.3 * u))
        return ui

    # 定义判断是否相遇的函数
    def check_encounter(x1, y1, x2, y2):
        return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 < 0.2


    bus_speed = integrand(0)
    robot_speed = 0
    bus_positions = []
    robot_positions = []
    total_time_1 = 0
    total_time_2 = 0
    total_time_3 = 0
    total_time_4 = 0
    s0 = cal_s0()
    s1 = cal_s1()
    s2 = cal_cycle()
    t0 = find_t0()
    t1 = find_t1()
    q = 0
    # 初始化参数
    dt = 0.001
    bus_theta = np.random.rand() * 2 * np.pi
    robot_theta = np.random.rand() * 2 * np.pi
    if abs(bus_theta - robot_theta) < np.pi/2:
        robot_theta += np.pi
    temp = bus_theta
    temp_1 = robot_theta

    if s2 * ((1 + s0) / (1 + s1)) >= 1:
        search_method = "method_1"
    else:
        if 3 * s0 * s2 + s0 + s2 > 1:
            search_method = "method_2"
        else:
            search_method = "method_3"
    # # 模拟运动过程
    while True:
        robot_speed = 0
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
        total_time_1 += dt

        # 更新bus速度
        bus_speed = integrand(total_time_1)

    bus_theta = temp
    robot_theta = temp_1
    bus_speed = integrand(0)
    robot_speed = 1
    bus_positions = []
    robot_positions = []
    q = 0
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
        total_time_2 += dt
        if total_time_2 == t0 and q == 0:
            robot_speed = robot_speed * (-1)
            q = 1
        # 更新bus速度
        bus_speed = integrand(total_time_2)

    bus_theta = temp
    robot_theta = temp_1
    bus_speed = integrand(0)
    robot_speed = 1
    bus_positions = []
    robot_positions = []
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
        total_time_3 += dt

        # 更新bus速度
        bus_speed = integrand(total_time_3)

    bus_theta = temp
    robot_theta = temp_1
    bus_speed = integrand(0)
    robot_speed = 1
    bus_positions = []
    robot_positions = []
    i = 1
    single = 0
    double = 0
    while True:
        ui = generate_ui(i)

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
        total_time_4 += dt
        single += dt
        double += dt
        if single >= ui % (2 * np.pi):
            single = 0
            robot_speed *= -1
            i += 1
        # 更新bus速度
        bus_speed = integrand(total_time_4)
    data_1 = [generate_polynomial(), search_method, round(total_time_1, 2), round(total_time_2, 2), round(total_time_3, 2), round(total_time_4, 2)]
    df.loc[len(df)] = data_1

df.to_csv('data_1.csv', index=False)