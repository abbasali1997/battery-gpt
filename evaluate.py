# Steps:
# 1. Use samples from checklistData to feed the decoder
# 2. Decode the series to (v,i,t) [input that the SOH estimator can read]
# 3. Pre-process the input
# 4. Use the pre-trained CNN_LSTM model to predict SOH curve
# 5. Assess Knee point and EOL and use evaluation metrics like RSME, MAPE, knee point error and EOL error

import os
import re
import pickle
import math
import numpy as np
import torch

from data.battery.MLSOH.MIT.data_preprocess import date_preprocess
from data.battery.MLSOH.MIT.CNN_LSTM import Net


TOKENS_PER_TIMESTEP = 3
TIMESTEPS_PER_CYCLE = 1200
WINDOW_POINTS = 3600          # 180 * 20
NUM = 180
SW_SHAPE = 20
ROW_PITCH = 2
PITCH_NUM = int(math.sqrt(NUM * SW_SHAPE) / ROW_PITCH)  # 30


def load_checklist_sample(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, str):
        tokens = np.array([int(x) for x in re.findall(r"-?\d+", obj)], dtype=np.float32)
    else:
        tokens = np.asarray(obj, dtype=np.float32).reshape(-1)

    tokens = tokens[: len(tokens) // TOKENS_PER_TIMESTEP * TOKENS_PER_TIMESTEP]
    triples = tokens.reshape(-1, 3)

    n_cycles = triples.shape[0] // TIMESTEPS_PER_CYCLE
    triples = triples[: n_cycles * TIMESTEPS_PER_CYCLE]

    return triples.reshape(n_cycles, TIMESTEPS_PER_CYCLE, 3)


def make_one_soh_input(v_win, i_win, t_win):
    v_img = date_preprocess(v_win, NUM * SW_SHAPE, ROW_PITCH, PITCH_NUM)
    i_img = date_preprocess(i_win, NUM * SW_SHAPE, ROW_PITCH, PITCH_NUM)
    t_img = date_preprocess(t_win, NUM * SW_SHAPE, ROW_PITCH, PITCH_NUM)

    cell = torch.tensor([v_img, i_img, t_img], dtype=torch.float32)

    # The original preprocessing duplicates the same tensor as cell1 and cell2.
    x = torch.stack([cell, cell], dim=0).unsqueeze(0)

    return x


def build_soh_batch(cycles, stride=1200):
    flat = cycles.reshape(-1, 3)

    v = flat[:, 0]
    i = flat[:, 1]
    t = flat[:, 2]

    xs = []
    start_indices = []

    for start in range(0, len(v) - WINDOW_POINTS + 1, stride):
        v_win = v[start:start + WINDOW_POINTS]
        i_win = i[start:start + WINDOW_POINTS]
        t_win = t[start:start + WINDOW_POINTS]

        xs.append(make_one_soh_input(v_win, i_win, t_win))
        start_indices.append(start)

    if not xs:
        raise ValueError("Not enough data. Need at least 3600 timesteps, i.e. 3 cycles.")

    return torch.cat(xs, dim=0), np.array(start_indices)


def load_soh_model(ckpt_path, device):
    model = Net(
        cell_num=2,
        input_size=180,
        hidden_dim=25,
        num_layers=3,
        sequencen_len=20,
        n_class=1,
        mode="LSTM",
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_soh(model, x, device, batch_size=128):
    preds = []

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = model(xb).detach().cpu().reshape(-1)
            preds.append(yb)

    return torch.cat(preds).numpy()


def compute_eol(soh, threshold=0.8):
    idx = np.where(soh < threshold)[0]
    return int(idx[0]) if len(idx) else None


def compute_knee_point(soh):
    soh = np.asarray(soh, dtype=np.float64)
    x = np.arange(len(soh), dtype=np.float64)

    dy = np.gradient(soh, x)
    ddy = np.gradient(dy, x)
    curvature = np.abs(ddy) / np.power(1.0 + dy ** 2, 1.5)

    return int(np.argmax(curvature)), curvature


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device.type)

    samples = [1,2,3,4,5,6]
    for i in samples:
        sample_path = f"checklistData\samples_{i}.bin"

        cycles = load_checklist_sample(sample_path)
        print("Loaded cycles:", cycles.shape)  # (num_cycles, 1200, 3)

        x, starts = build_soh_batch(cycles, stride=1200)
        print("CNN-LSTM input:", x.shape)      # (batch, 2, 3, 60, 60)

        ckpt_path = r"Models\MIT_MODEL\CNN_LSTM\epoch_2040.params"

        model = load_soh_model(ckpt_path, device)
        soh = predict_soh(model, x, device)

        knee_idx, curvature = compute_knee_point(soh)
        eol_idx = compute_eol(soh, threshold=0.8)

        # Each prediction corresponds to a 3600-point window.
        # With stride=1200, one prediction step roughly equals one cycle.
        knee_cycle = int(starts[knee_idx] // TIMESTEPS_PER_CYCLE)
        eol_cycle = None if eol_idx is None else int(starts[eol_idx] // TIMESTEPS_PER_CYCLE)

        print(f"\nResults for sample {i}: ")
        print("-------")
        print("SOH predictions:", len(soh))
        print("SOH min/max:", float(np.min(soh)), float(np.max(soh)))
        print("Knee index:", knee_idx)
        print("Knee cycle approx:", knee_cycle)
        print("EOL index:", eol_idx)
        print("EOL cycle approx:", eol_cycle)

        os.makedirs("outputs", exist_ok=True)
        np.savez(
            f"outputs/soh_results_samples_{i}.npz",
            soh=soh,
            curvature=curvature,
            starts=starts,
            knee_idx=knee_idx,
            eol_idx=-1 if eol_idx is None else eol_idx,
        )

        print(f"\nSaved: outputs/soh_results_samples_{i}.npz")

        print('===========================')


if __name__ == "__main__":
    main()