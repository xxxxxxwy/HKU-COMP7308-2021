import numpy as np
import random
import matplotlib.pyplot as plt

def next_step(x, N, e, L, p):
    aa = np.random.binomial(n=1, p=p, size=1)
    if aa[0] == 1:
        v_k = 1
    else:
        v_k = -1
    x_k = (x + v_k) % N
    theta_k = 2 * np.pi * x_k / N
    w_k = random.uniform(-e, e)
    z_k = ((L - np.cos(theta_k)) ** 2 + (np.sin(theta_k)) ** 2) ** 0.5 + w_k

    theta=np.arccos((L**2-z_k**2+1)/(2*L))
    if theta<0: theta+=2*np.pi
    if theta>2*np.pi: theta-=2*np.pi
    x=theta*N/(2*np.pi)
    z_k=x

    return x_k, z_k

def distribution(k, x0, N, e, L, p):
    z_list = []
    x_list = []
    x = x0
    for i in range(k):
        x_k, z_k = next_step(x, N, e, L, p)
        z_list.append(z_k)
        x_list.append(x_k)
        x = x_k
    return x_list, z_list

N = 100
x0 = N / 4
e = 0.5
L = 2
p = 0.5
k = 10

point_number = 1000

x = []
z = []
for i in range(point_number):
    x_list, z_list = distribution(k, x0, N, e, L, p)
    x += x_list
    z += z_list
# print(result)

for step in range(k):
    data_x = []
    data_z = []
    for i in range(point_number):
        data_x.append(x[i * k + step])
        data_z.append(z[i * k + step])
    #data = List3 = np.multiply(np.array(data_x), np.array(data_z))
    data=data_x
    plt.hist(data,density=1)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.show(block=False)
    plt.pause(1)
    plt.clf()
