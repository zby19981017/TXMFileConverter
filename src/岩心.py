import numpy as np
import pandas as pd
from vedo import Volume
from vedo.applications import RayCastPlotter

# file_name = 'E:\\备份\\数据\\L95_0_-8-bit.csv'
file_name = 'E:\\备份\\数据\\CT-2023-4\\JY-4\\JY-4_dry_dry_reco23123123.csv'
CSV_file = pd.read_csv(file_name, sep=',')
data = np.array(CSV_file)

flag = 0
x = -1
y = -1
z = -1
v = -1
# print(np.max(data[:, 3]))

for i in range(4):
    a = data[:, i][int(len(data[:, i]) / 2)]
    if (a > 0 and a < 1):
        v = i
        continue
    if (flag == 0):
        x = i
    elif (flag == 1):
        y = i
    elif (flag == 2):
        z = i
    flag += 1

axis_x = np.array(data[:, x], dtype=np.int32)
axis_y = np.array(data[:, y], dtype=np.int32)
axis_z = np.array(data[:, z], dtype=np.int32)
value = np.array(data[:, v], dtype=np.float32)

print(axis_x)

grid_nx = max(axis_x) + 1
grid_ny = max(axis_y) + 1
grid_nz = max(axis_z) + 1

frame = np.zeros(grid_nz * grid_ny * grid_nx).reshape(grid_nx, grid_ny, grid_nz)

frame[axis_x, axis_y, axis_z] = value

vb = Volume(frame).mode(1).c("rainbow").alpha([0, 0.8, 0.85, 0.9, 0.95, 0.95, 1]).addScalarBar()

plt = RayCastPlotter(vb, bg='w', axes=7)  # Plotter instance
plt.show(viewup="y", size=(1200, 750)).close()
