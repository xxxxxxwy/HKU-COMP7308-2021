from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt


def diffdrive(x, y, theta, w1, w2, t, l):
    r = 1
    w = ((w1 - w2) * r) / (2 * l)
    if w1 == w2:
        return x + t * cos(theta) * w1 * r, y + t * sin(theta) * w1 * r, theta
    R = l * (w1 + w2) / (w2 - w1)
    icrx = x + R * sin(theta)
    icry = y - R * cos(theta)
    A = np.array(
        [[cos(w * t), -sin(w * t), 0],
         [sin(w * t), cos(w * t), 0],
         [0, 0, 1]]
    )
    B = np.array([[x - icrx], [y - icry], [theta]])
    C = np.array([[icrx], [icry], [w * t]])
    D = np.dot(A, B) + C
    xn = D[0][0]
    yn = D[1][0]
    thetan = D[2][0]

    return xn, yn, thetan


# x1,y1,theta1=diffdrive(1.5,2.0,np.pi/2,0.2,0.3,3,0.25)
# x2,y2,theta2=diffdrive(x1,y1,theta1,-0.1,0.1,1,0.25)
# x3,y3,theta3=diffdrive(x2,y2,theta2,0,0.2,2,0.25)
# print(x3,y3)

def diffdrive_gauss(x, y, theta, w1, w2, t, l, n1, n2, n3):
    dot_number = 20
    r = 1
    w = ((w1 - w2) * r) / (2 * l)
    if w1 == w2:
        return x + t * cos(theta) * w1 * r, y + t * sin(theta) * w1 * r, theta
    R = l * (w1 + w2) / (w2 - w1)
    icrx = x + R * sin(theta)
    icry = y - R * cos(theta)
    A = np.array(
        [[cos(w * t), -sin(w * t), 0],
         [sin(w * t), cos(w * t), 0],
         [0, 0, 1]]
    )
    B = np.array([[x - icrx], [y - icry], [theta]])
    C = np.array([[icrx], [icry], [w * t]])
    D = np.dot(A, B) + C
    xn = D[0][0]
    yn = D[1][0]
    thetan = D[2][0]
    xn_list = []
    yn_list = []
    thetan_list = []
    for i in range(dot_number):
        xn_list.append(xn + np.random.normal(0, n1))
        yn_list.append(yn + np.random.normal(0, n2))
        thetan_list.append(thetan + np.random.normal(0, n3))
    return xn_list, yn_list, thetan_list


def diffdrive_direct(x, y, theta, w1, w2, t, l, n):
    dot_number=20
    xn_list = []
    yn_list = []
    thetan_list = []
    for i in range(dot_number):
        w1+=np.random.normal(0,n)
        w2+=np.random.normal(0,n)
        r = 1
        w = ((w1 - w2) * r) / (2 * l)
        #w += np.random.normal(0, n)
        if w1 == w2:
            return x + t * cos(theta) * w1 * r, y + t * sin(theta) * w1 * r, theta
        R = l * (w1 + w2) / (w2 - w1)
        icrx = x + R * sin(theta)
        icry = y - R * cos(theta)
        A = np.array(
            [[cos(w * t), -sin(w * t), 0],
             [sin(w * t), cos(w * t), 0],
             [0, 0, 1]]
        )
        B = np.array([[x - icrx], [y - icry], [theta]])
        C = np.array([[icrx], [icry], [w * t]])
        D = np.dot(A, B) + C
        xn = D[0][0]
        yn = D[1][0]
        thetan = D[2][0]
        xn_list.append(xn)
        yn_list.append(yn)
        thetan_list.append(thetan)
    return xn_list, yn_list, thetan_list


# xlist = []
# ylist = []
# n1 = n2 = 0.01
# n3 = 0.3
# x1, y1, theta1 = diffdrive_gauss(1.5, 2.0, np.pi / 2, 0.2, 0.3, 3, 0.25, n1, n2, n3)
# for i in range(len(x1)):
#     x2, y2, theta2 = diffdrive_gauss(x1[i], y1[i], theta1[i], -0.1, 0.1, 1, 0.25, n1, n2, n3)
#     xlist += x2
#     ylist += y2
#     for j in range(len(x2)):
#         x3, y3, theta3 = diffdrive_gauss(x2[j], y2[j], theta2[j], 0, 0.2, 2, 0.25, n1, n2, n3)
#         xlist += x3
#         ylist += y3
#
# xlist.append(1.5)
# ylist.append(2.0)
# plt.scatter(xlist, ylist)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

xlist = []
ylist = []
n=0.002
x1, y1, theta1 = diffdrive_direct(1.5, 2.0, np.pi / 2, 0.2, 0.3, 3, 0.25, n)
for i in range(len(x1)):
    x2, y2, theta2 = diffdrive_direct(x1[i], y1[i], theta1[i], -0.1, 0.1, 1, 0.25, n)
    xlist += x2
    ylist += y2
    for j in range(len(x2)):
        x3, y3, theta3 = diffdrive_direct(x2[j], y2[j], theta2[j], 0, 0.2, 2, 0.25, n)
        xlist += x3
        ylist += y3

xlist.append(1.5)
ylist.append(2.0)
plt.scatter(xlist, ylist,s=0.1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()