def v2(x, u):
    if u + x == 5 or u + x == 0:
        return 2 * x ** 2 + 2 * u ** 2 + 2 * x * u
    else:
        return 2 * x ** 2 + 2 * u ** 2 + 2 * x * u + 1

for x in range(0, 6):
    v_list = []
    for u in range(-5, 6):
        if ((x + u) < 0) or ((x + u) > 5): continue
        print(x, u, v2(x, u))
        v_list.append(v2(x, u))
    print('v2---min:---', min(v_list))

def v2_res(x):
    if x==0: return 0
    if x == 1: return 2
    if x == 2: return 7
    if x == 3: return 15
    if x == 4: return 25
    if x == 5: return 39

def v1(x, u):
    if u + x == 5 or u + x == 0:
        return x ** 2 + u ** 2 + v2_res(x+u)
    else:
        return x** 2 + u ** 2 + 0.5*v2_res(x+u+1)+0.5*v2_res(x+u-1)

for x in range(0, 6):
    v_list = []
    for u in range(-5, 6):
        if ((x + u) < 0) or ((x + u) > 5): continue
        print(x, u, v1(x, u))
        v_list.append(v1(x, u))
    print('v1---min:---', min(v_list))

def v1_res(x):
    if x==0: return 0
    if x == 1: return 2
    if x == 2: return 8
    if x == 3: return 16.5
    if x == 4: return 28.5
    if x == 5: return 42.5

def v0(x, u):

    if u + x == 5 or u + x == 0:
        return x ** 2 + u ** 2 + v1_res(x+u)
    else:
        return x** 2 + u ** 2 + 0.5*v1_res(x+u+1)+0.5*v1_res(x+u-1)

v_list = []
x=5
for u in range(-5, 6):
    if ((x + u) < 0) or ((x + u) > 5): continue
    print(x, u, v0(x, u))
    v_list.append(v0(x, u))
print('v0---min:---', min(v_list))