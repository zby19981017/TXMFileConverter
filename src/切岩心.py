import numpy as np
import pandas as pd
from vedo import Volume
from vedo.applications import RayCastPlotter



file_name = 'E:\\备份\\数据\\杨鑫博士论文数据体\\L87_120KV_8W_46UM_recon.csv'

def a():
    CSV_file = pd.read_csv(file_name, sep=',')
    data = np.array(CSV_file)

    CSV_file = pd.read_csv("./cutshape.csv", sep=',', header=None)
    cutdata = np.array(CSV_file)

    axis_x = np.array(data[:, 0], dtype=np.int32)
    axis_y = np.array(data[:, 1], dtype=np.int32)
    axis_z = np.array(data[:, 2], dtype=np.int32)
    value = np.array(data[:, 3], dtype=np.float32)

    x = np.array(cutdata[:, 0], dtype=np.int32)
    y = np.array(cutdata[:, 1], dtype=np.int32)
    z = np.array(cutdata[:, 2], dtype=np.int32)

    grid_nx = max(x) + 1
    grid_ny = max(y) + 1
    grid_nz = max(z) + 1

    v = np.zeros(len(x))
    count = 0
    for i in range(len(axis_x)):
        if (count == len(x)):
            break
        if (x[count] == axis_x[i]):
            if (y[count] == axis_y[i]):
                if (z[count] == axis_z[i]):
                    v[count] = value[i]
                    count += 1

    frame = np.zeros(grid_nx * grid_ny * grid_nz).reshape(grid_nx, grid_ny, grid_nz)

    frame[x, y, z] = v

    vb = Volume(frame).mode(1).c("rainbow").alpha([0, 0.8, 0.85, 0.9, 0.95, 0.95, 1]).addScalarBar()

    plt = RayCastPlotter(vb, bg='w', axes=7)  # Plotter instance
    plt.show(viewup="y", size=(1200, 750)).close()
