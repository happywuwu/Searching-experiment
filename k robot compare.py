import numpy as np
from scipy.optimize import root
import random
import pandas as pd


df = pd.DataFrame(columns=['速度函数','算法', 'time_1','time_2','time_3'])

for i in range(100):
    print(i)
    k = random.randint(3, 7)
    print(k)
    # 生成随机数组作为多项式的系数
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    random_number = random.randint(2, 3)
    for j in range(random_number):
        a[j] = round(random.uniform(0, 1), 2)

    def generate_polynomial():
        global a, random_number
        polynomial = ""
        for m in range(random_number):
            if m == 0:
                polynomial += f"{a[m]}"
            else:
                polynomial += f" + {a[m]}*t^{m}"
        return polynomial

    # 定义bus速度随时间变化的函数
    def integrand(t):
        global a, random_number
        result = 0
        for j in range(random_number):  # 计算多项式函数在 t 处的值
            result += 1 / (j + 1) * a[j] * t ** (j + 1)
        return result

    def integrand_1(t):
        result = -integrand(t)
        return result

    def find_t0():
        target = 2 * np.pi / k
        # 使用数值优化来找到
        solution = root(lambda t: integrand(t) - target, 1.5)
        if not solution.success or solution.x[0] < 0:
            solution = root(lambda t: integrand_1(t) - target, 1.5)
        return solution.x[0]

    def find_t1():
        t_1 = 2 * np.pi / k
        return t_1

    def find_t2():
        # 使用数值优化来找到
        solution = root(lambda t: 2 * np.pi - abs(t - integrand(t)) - (k - 1) * t, 0)
        return solution.x[0]

    # 定义判断是否相遇的函数
    def check_encounter(x1, y1, x2, y2):
        return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 < 0.01

    t0 = find_t0()
    t1 = find_t1()
    t2 = find_t2()
    dt = 0.01
    total_time_1 = 0
    total_time_2 = 0
    total_time_3 = 0
    robot_speed = 1
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
                    theta[i] = t2 * (i - 1)

    bus_theta = np.random.rand() * 2 * np.pi
    robot_thetas = theta
    bus_speed = integrand(0)
    temp = bus_theta
    temp_1 = robot_thetas
    con = 0

    if k % 2 == 0:
        if t0 < t1:
            search_method = "method_1"
        else:
            search_method = "method_2"
    else:
        if t0 < t2:
            search_method = "method_1"
        else:
            search_method = "method_3"


    if k % 2 == 0:
        bus_theta = temp
        robot_thetas = temp_1
        con = 0
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
            total_time_1 += dt
            # 更新bus速度
            bus_speed = integrand(total_time_1)
            if con == 1:
                break
        bus_theta = temp
        robot_thetas = temp_1
        con = 0
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
            total_time_2 += dt
            # 更新bus速度
            bus_speed = integrand(total_time_2)
            if con == 1:
                break
    else:
        bus_theta = temp
        robot_thetas = temp_1
        con = 0
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
            total_time_1 += dt
            # 更新bus速度
            bus_speed = integrand(total_time_1)
            if con == 1:
                break
        bus_theta = temp
        robot_thetas = temp_1
        con = 0

        while True:
            # 更新bus和robot的位置
            bus_theta += bus_speed * dt
            for i in range(k):
                if i < k - 1:
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
            total_time_3 += dt
            # 更新bus速度
            bus_speed = integrand(total_time_3)
            if con == 1:
                break


    data_1 = [generate_polynomial(), search_method, round(total_time_1, 2), round(total_time_2, 2), round(total_time_3, 2)]
    df.loc[len(df)] = data_1

df.to_csv('data_2.csv', index=False)
