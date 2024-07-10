import os
import json
import uuid
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ShallowWater(object):
    def __init__(self, h_ini=1., dx=0.01, dt=0.0001, N=64, px=16, py=16, R=64, Hp=0.1, g=1., b=0.):
        '''
        Args:
            h_ini: initial height of the water
            dx: spatial resolution
            dt: temporal resolution
            N: size of the grid
            px, py: center of the perturbation
            R: radius of the perturbation
            Hp: height of the perturbation in pressure surface
            g: gravity
            b: friction
        '''
        self.time = 0

        # Physical parameters
        self.g = g
        self.b = b

        self.dx = dx  # 1. / N # a changer
        self.dt = dt  # self.dx / 100.

        # limits for h,u,v
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))

        x, y = np.mgrid[:N, :N]
        rr = (x-px)**2 + (y-py)**2

        self.h = h_ini * np.ones((N, N))
        self.h[rr < R] = h_ini + Hp  # set initial conditions

        self.lims = [(h_ini - Hp, h_ini + Hp), (-0.02, 0.02), (-0.02, 0.02)]

    def dxy(self, A, axis=0):
        """
        Compute derivative of array A using balanced finite differences
        Axis specifies direction of spatial derivative (d/dx or d/dy)
        dA[i]/dx =  (A[i+1] - A[i-1] )  / 2dx
        """
        return (np.roll(A, -1, axis) - np.roll(A, 1, axis)) / (self.dx*2.)  # roll: shift the array axis=0 shift the horizontal axis

    def d_dx(self, A):
        return self.dxy(A, 1)

    def d_dy(self, A):
        return self.dxy(A, 0)

    def d_dt(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        """
        for x in [h, u, v]:  # type check
            assert isinstance(x, np.ndarray) and not isinstance(x, np.matrix)

        g, b, dx = self.g, self.b, self.dx

        du_dt = -g*self.d_dx(h) - b*u
        dv_dt = -g*self.d_dy(h) - b*v

        H = 0  # h.mean() - our definition of h includes this term
        dh_dt = -self.d_dx(u * (H+h)) - self.d_dy(v * (H+h))

        return dh_dt, du_dt, dv_dt

    def evolve(self):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        dh_dt, du_dt, dv_dt = self.d_dt(self.h, self.u, self.v)
        dt = self.dt

        self.h += dh_dt * dt
        self.u += du_dt * dt
        self.v += dv_dt * dt
        self.time += dt

        return self.h, self.u, self.v


def simulate(save_dir=None):
    N = 64 * 2
    px = random.randint(27 * 2, 37 * 2) * 1.
    py = random.randint(27 * 2, 37 * 2) * 1.
    R = random.randint(40 * 2, 80 * 2) * 1.
    Hp = random.uniform(0.05, 0.2)
    b = random.uniform(0.02, 2)
    step = random.randint(60, 100)
    sequence_length = 200

    # print(f'px: {px}, py: {py}, R: {R}, Hp: {Hp:.4f}, b: {b:.4f}, step: {step}')

    sw = ShallowWater(N=N, px=px, py=py, R=R, Hp=Hp, b=b)

    sequence_data = np.zeros((sequence_length, 3, N, N))

    for i in range(step * sequence_length):
        if i % step == 0:
            index = i // step

            # print(f'index {index}, time {sw.time:.4f}')
            # plt.imshow(sw.h)
            # plt.savefig(f'shallow_water/data/{index}.png')
            # # plt.show()
            # plt.close()

            sample = np.stack([sw.h, sw.u, sw.v], axis=0)
            sequence_data[index] = sample

        sw.evolve()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f'{uuid.uuid4().hex[:16]}.npy'
        np.save(os.path.join(save_dir, file_name), sequence_data.astype(np.float32))
        return {
            file_name: {
                "px": px,
                "py": py,
                "R": R,
                "Hp": Hp,
                "b": b,
                "step": step,
            }
        }


if __name__ == "__main__":
    configs = {}
    for i in tqdm(range(600)):
        config = simulate(save_dir='data/')
        configs.update(config)
    json.dump(configs, open('configs.json', 'w'))
