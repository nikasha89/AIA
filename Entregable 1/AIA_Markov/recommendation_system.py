import math as ma

r = [
    [1,1,1],
    [2,2,2],
    [0,0,4],
    ]

def mean_row(r):
    total = 0
    for i in r:
        total += i
    return total / len(r)
print(mean_row(r[0]))

def mean_col(r, val):
    total = 0
    for i in range(len(r)):
        for j in range(len(r[0])):
            total += r[j][val]
    return total / len(r[val])
print(mean_col(r,2))


def sim_user(a, b):
    total1 = 0
    total2 = 0
    total3 = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            total1 += (a[i][j] - mean_row(a[i])) * (b[i][j] - mean_row(b[i]))
            total2 += ((a[i][j]) - mean_row(a[i])) ** 2
            total3 += ((b[i][j]) - mean_row(b[i])) ** 2
    return total1 / (ma.sqrt(total2) * ma.sqrt(total3))


def pred_user(a, p):
    total_mean = 0
    total1 = 0
    total2 = 0
    total3 = 0
    for i in range(len(a)):
        total_mean = mean_row(a[i])
        for j in range(len(a)):
            if i != j:
                total1 += sim_user(a[i], a[j]) * (a[j][p] - mean_row(a[j]))
                total2 += (a[i][p] - mean_row(a[i])) ** 2
                total3 += (a[j][p] - mean_row(a[j])) ** 2
    return total_mean + (total1 / (ma.sqrt(total2) * ma.sqrt(total3)))


def sim_item(p, q):
    total1 = 0
    total2 = 0
    total3 = 0
    for i in range(len(p)):
        for j in range(len(p[i])):
            total1 += (p[j][i] - mean_col(p, j)) * (q[j][i] - mean_col(q, j))
            total2 += ((p[j][i]) - mean_col(p, j)) ** 2
            total3 += ((q[j][i]) - mean_col(q, j)) ** 2
    return total1 / (ma.sqrt(total2) * ma.sqrt(total3))


def pred_item(q, p):
    total1 = 0
    total2 = 0
    for i in range(len(q[p])):
        for j in range(len(q[p])):
            if i != j:
                total1 += sim_item(q[i], q[j]) * q[p][j]
                total2 += sim_item(q[i], q[j])
    return total1 / total2
