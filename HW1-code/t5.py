from math import sin, cos, pi, sqrt, atan2
import random
import matplotlib.pyplot as plt
import numpy as np


class robot_arm:
    base = [0, 0]
    joint = [0, 0]
    end = [0, 0]

    def __init__(self, basex, basey, l1, l2):
        self.base[0] = basex
        self.base[1] = basey
        self.l1 = l1
        self.l2 = l2


def randPoint(radius, basex, basey):
    theta = random.random() * 2 * pi
    r = random.random() ** 0.5 * radius
    x = basex + r * cos(theta)
    y = basey + r * sin(theta)
    x = abs(round(x, 2))
    y = abs(round(y, 2))
    return x, y


def generate_task(robot_arm):
    arm_range = robot_arm.l1 + robot_arm.l2
    goal_list = []
    for i in range(4):
        goal_list.append(randPoint(arm_range, robot_arm.base[0], robot_arm.base[1]))
    return goal_list


def get_joint(p1, r1, p2, r2):
    x = p1[0]
    y = p1[1]
    R = r1
    a = p2[0]
    b = p2[1]
    S = r2
    d = sqrt((abs(a - x)) ** 2 + (abs(b - y)) ** 2)
    if d > (R + S) or d < (abs(R - S)):
        print("Two circles have no intersection")
        return
    elif d == 0 and R == S:
        print("Two circles have same center!")
        return
    else:
        A = (R ** 2 - S ** 2 + d ** 2) / (2 * d)
        h = sqrt(R ** 2 - A ** 2)
        x2 = x + A * (a - x) / d
        y2 = y + A * (b - y) / d
        x3 = round(x2 - h * (b - y) / d, 2)
        y3 = round(y2 + h * (a - x) / d, 2)
        x4 = round(x2 + h * (b - y) / d, 2)
        y4 = round(y2 - h * (a - x) / d, 2)
        c1 = [x3, y3]
        c2 = [x4, y4]
        return c1, c2


def get_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / pi)
    angle2 = atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def get_better_joint(robot_arm, j1, j2, goal):
    old_joint = robot_arm.joint
    old_end = robot_arm.end
    base = robot_arm.base
    old_arm1 = [base[0], base[1], old_joint[0], old_joint[1]]
    old_arm2 = [old_joint[0], old_joint[1], old_end[0], old_end[1]]
    # j1
    new_arm1 = [base[0], base[1], j1[0], j1[1]]
    new_arm2 = [j1[0], j1[1], goal[0], goal[1]]
    cost1 = get_angle(old_arm1, new_arm1) + get_angle(old_arm2, new_arm2)
    # j2
    new_arm1 = [base[0], base[1], j2[0], j2[1]]
    new_arm2 = [j2[0], j2[1], goal[0], goal[1]]
    cost2 = get_angle(old_arm1, new_arm1) + get_angle(old_arm2, new_arm2)
    if cost1 > cost2: return j2
    return j1


while True:
    try:
        arm = robot_arm(0, 0, random.randint(1, 5), random.randint(1, 5))
        goal_list = generate_task(arm)
        joint_list = []
        print('task sequence:',goal_list)
        j1, j2 = get_joint(arm.base, arm.l1, goal_list[0], arm.l2)
        robot_arm.joint = j1
        robot_arm.end = goal_list[0]
        joint_list.append(robot_arm.joint)

        for i in range(1, 4):
            j1, j2 = get_joint(arm.base, arm.l1, goal_list[i], arm.l2)
            joint_list.append(get_better_joint(robot_arm, j1, j2, goal_list[i]))
        #print(joint_list)

        plt.scatter(arm.base[0], arm.base[1], color='red')
        for joint in joint_list:
            plt.scatter(joint[0], joint[1], color='yellow')

        for i in range(3):
            plt.scatter(goal_list[i][0], goal_list[i][1], color='blue')

        plt.scatter(goal_list[3][0], goal_list[3][1], color='red', marker='*')

        i = 0
        plt.plot([joint_list[i][0], goal_list[i][0]], [joint_list[i][1], goal_list[i][1]], color='red')
        plt.plot([joint_list[i][0], arm.base[0]], [joint_list[i][1], arm.base[1]], color='red')

        for i in range(1, 4):
            plt.plot([joint_list[i][0], goal_list[i][0]], [joint_list[i][1], goal_list[i][1]], color='green')
            plt.plot([joint_list[i][0], arm.base[0]], [joint_list[i][1], arm.base[1]], color='green')

        plt.show()
    except:
        print('error')
