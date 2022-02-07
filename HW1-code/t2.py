# import math
# import numpy as np
# import matplotlib.pyplot as plt
# scan = np.loadtxt('laserscan.dat')
# angle = np.linspace(-math.pi/3, math.pi/3, np.shape(scan)[0], endpoint='true')
# print(angle)
# x=[]
# y=[]
# for i in range(len(scan)):
#     l=scan[i]
#     a=angle[i]
#     x.append(l*math.cos(a))
#     x.append(0)
#     y.append(l * math.sin(a))
#     y.append(0)
#
#
#
#
# #plt.scatter(x,y)
# plt.plot(x,y)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
import math
import numpy as np
import matplotlib.pyplot as plt

scan = np.loadtxt('laserscan.dat')
angle = np.linspace(-math.pi / 3, math.pi / 3, np.shape(scan)[0], endpoint='true')
print(angle)
x = []
y = []
for i in range(len(scan)):
    l = scan[i]
    a = angle[i]

    x1 = l * math.cos(a)
    y1 = l * math.sin(a)
    y2=x1
    x2=-y1+0.2
    x3=0.7071*x2-0.7071*y2+1
    y3 = 0.7071 * x2 + 0.7071 * y2 + 0.5



    x.append(x3)
    y.append(y3)

plt.scatter(x, y, color='red')
#yuan dian
plt.scatter(0, 0, color='yellow')
#robot
plt.scatter(1, 0.5, color='blue')
#sensor
plt.scatter(1+0.0707, 0.5+0.0707, color='green')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
