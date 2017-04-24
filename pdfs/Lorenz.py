import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
stepCnt = 1000

# Need one more for the initial values
xs = np.empty((stepCnt + 1,))
ys = np.empty((stepCnt + 1,))
zs = np.empty((stepCnt + 1,))

# Setting initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Stepping through "time".
for i in range(stepCnt):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

#new set of boundary conditions
xs2 = np.empty((stepCnt + 1,))
ys2 = np.empty((stepCnt + 1,))
zs2 = np.empty((stepCnt + 1,))

xs2[0], ys2[0], zs2[0] = (0.001, 1.001, 1.051)


for i in range(stepCnt):
    
    # Derivatives of the X, Y, Z state
    
    x_dot2, y_dot2, z_dot2 = lorenz(xs2[i], ys2[i], zs2[i])
    
    xs2[i + 1] = xs2[i] + (x_dot2 * dt)
    ys2[i + 1] = ys2[i] + (y_dot2 * dt)
    zs2[i + 1] = zs2[i] + (z_dot2 * dt)

fig = pl.figure()
pl.rcParams['axes.facecolor'] = 'white'
fig.patch.set_facecolor('white')
ax = fig.gca(projection='3d')

ax.plot(xs2, ys2, zs2,'b', lw=0.1)
ax.plot(xs2[0:1],ys2[0:1],zs2[0:1],'bo', markersize = 10)
ax.plot(xs2[stepCnt-1:stepCnt],ys2[stepCnt-1:stepCnt],zs2[stepCnt-1:stepCnt],'bo', markersize =10)
ax.set_xlabel("X Axis")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")


ax.plot(xs, ys, zs,'r', lw=0.1)
ax.plot(xs[0:1],ys[0:1],zs[0:1],'ro', markersize = 5 )
ax.plot(xs[stepCnt-1:stepCnt],ys[stepCnt-1:stepCnt],zs[stepCnt-1:stepCnt],'ro', markersize=10)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

pl.show()
