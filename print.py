from env import Support_v0

from stable_baselines3 import PPO
import torch as th
import numpy as np

from stl import mesh

def extrude(img):
    n_faces = np.count_nonzero(img)*2
    m = mesh.Mesh(np.zeros(n_faces, dtype=mesh.Mesh.dtype))
    n_row, n_col = img.shape

    step = 1/25
    points = []
    for row in range(n_row):
        for col in range(n_col):
            if not img[row, col]:
                continue

            xl = -1+col*step
            xr = -1+(col+1)*step
            yt = 1-row*step
            yb = 1-(row+1)*step

            v0 = xl, yt, 0
            v1 = xl, yb, 0
            v2 = xr, yb, 0
            v3 = xr, yt, 0

            f1, f2 = [], []
            f1.append(v0)
            f1.append(v1)
            f1.append(v2)
            f2.append(v0)
            f2.append(v2)
            f2.append(v3)

            points.append(f1)
            points.append(f2)

    m.vectors = points
    return m

if __name__ == "__main__":
    env = Support_v0()
    model = PPO.load("./rl_model")

    obs = env.reset()
    done = False
    while not done:
        action = model.policy._predict(
            th.FloatTensor(obs).unsqueeze(0).to(th.device("cuda")),
            deterministic=True,
        ).item()
        obs, _, done, _ = env.step(action)
        if done:
            break

    img = np.logical_or(obs[0], obs[1])
    m = extrude(img)
    m.save('mesh.stl')
