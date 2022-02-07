import numpy as np
import matplotlib.pyplot as plt


#############################basic functions##############################
def x_2_next_x(x, N, p):
    aa = np.random.binomial(n=1, p=p, size=1)
    if aa[0] == 1:
        v_k = 1
    else:
        v_k = -1
    x_k = (x + v_k) % N
    return x_k


def x_2_z(x_k, N, e, L):
    theta_k = 2 * np.pi * x_k / N
    w_k = np.random.uniform(-e, e)
    z_k = ((L - np.cos(theta_k)) ** 2 + (np.sin(theta_k)) ** 2) ** 0.5 + w_k
    return z_k


def z_2_x(avg, z_k, L, N, e):
    # deal with noise
    if z_k > avg + e / 2: z_k += np.random.uniform(-e, 0)
    # if z_k<avg-e: z_k+=np.random.uniform(0, e)
    theta = np.arccos((L ** 2 - z_k ** 2 + 1) / (2 * L))
    if theta < 0:
        theta += 2 * np.pi
    if theta > 2 * np.pi:
        theta -= 2 * np.pi
    # we cant tell whether the obj is above zero or below zero
    x_1 = theta * N / (2 * np.pi)
    x_2 = (2 * np.pi - theta) * N / (2 * np.pi)
    return x_1, x_2


######################################params#####################################
sample_num = 10000
k = 10
N = 100
x0 = N / 4
e = 0.5
L = 2
p = 0.5
# parms in question(c)
ee = e
pp = p

#####################################init###########################################
# init estimition distribution using uniform
x_m_distribute = np.empty([sample_num], dtype=int)
for i in range(sample_num):
    x_m_distribute[i] = np.random.uniform(0, N)

# init simulation distribution using x0
x_d_distribute = x0 * np.ones(sample_num, dtype=int)

# plt
plt.hist(x_d_distribute, density=1)
plt.hist(x_m_distribute, bins=N, density=1)
plt.xlim(0, N)
plt.ylim(0, 1)
plt.show(block=False)
plt.pause(1)
plt.clf()

######################start simulation and estimation######################################
for step in range(k):
    # init 2 temp distribution
    x_distribute = np.ones([sample_num], dtype=int)
    z_distribute = np.ones([sample_num], dtype=float)
    for i in range(sample_num):
        # using motion model to calculate the simulation result
        x_d_distribute[i] = x_2_next_x(x_d_distribute[i], N, p)
        # using simulation result and measurement model to get temp z distribution
        z_distribute[i] = x_2_z(x_d_distribute[i], N, e, L)
    # use a little trick to narrow the z distribution by reduce noise(more close to the avg)
    avg = sum(z_distribute) / len(z_distribute)
    # use measurement model to calculate x distribution(using estimation pp adn ee)
    for i in range(sample_num):
        x1, x2 = z_2_x(avg, z_distribute[i], L, N, ee)
        # there are some strange errors, ignore
        try:
            x_distribute[i] = int(x1 if np.random.binomial(n=1, p=0.5, size=1) else x2)
        except:
            x_distribute[i] = 0
    # normalization and bayes
    # transform value distribution to possibility distribution
    p1 = np.zeros([N], dtype=int)
    for i in range(N):
        for j in range(sample_num):
            if x_distribute[j] == i:
                p1[i] += 1
    p2 = np.zeros([N], dtype=int)
    for i in range(N):
        for j in range(sample_num):
            if x_m_distribute[j] == i:
                p2[i] += 1
    for i in range(N):
        p1[i] = p1[i] * p2[i]
    # multiply x estimation distribution and x simulation distribution
    p3 = np.zeros([N], dtype=int)
    for i in range(N):
        p3[i] = (p1[i] / p1.sum()) * sample_num
    # transform possibility distribution to value distribution(for loop)
    x_m_distribute = np.ones([sample_num], dtype=int)
    count = 0
    for i in range(N):
        for j in range(int(p3[i])):
            if count == sample_num: break
            x_m_distribute[count] = i
            count += 1
    # using motion model to calculate next step
    for i in range(N):
        x_m_distribute[i] = x_2_next_x(x_m_distribute[i], N, pp)
    # plt
    plt.hist(x_d_distribute, density=1)
    plt.hist(x_m_distribute, bins=N, density=1)
    plt.xlim(0, N)
    plt.ylim(0, 1)
    plt.show(block=False)
    plt.pause(1)
    plt.clf()
