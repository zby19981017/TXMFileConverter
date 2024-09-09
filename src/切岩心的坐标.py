import pandas as pd
import numpy as np
import 切岩心

x1 = 50
x2 = 230
y1 = 0
y2 = 99
z1 = 0
z2 = 99

xlen = x2 - x1
ylen = y2 - y1
zlen = z2 - z1


data = np.zeros(xlen * ylen * zlen * 3).reshape(xlen * ylen * zlen, 3)
count = 0

for i in range(z1, z2):
    for j in range(y1, y2):
        for k in range(x1, x2):
                data[count, 0] = k
                data[count, 1] = j
                data[count, 2] = i
                count += 1
print(data)
print(len(data))
file = pd.DataFrame(data)
file.to_csv("cutshape.csv", index=False, header=False, na_rep="NULL")

切岩心.a()
