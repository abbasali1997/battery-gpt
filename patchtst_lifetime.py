import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

SOH_FILE = "data/battery/MIT_SOH.db"

# Number of historical cycles provided to the model
START_CYCLES = 30

# Number of cycles used as model context
CONTEXT_LENGTH = 30

# PatchTST parameters
PATCH_LENGTH = 5
PATCH_STRIDE = 2

D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 3
FF_DIM = 128
DROPOUT = 0.1

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

# EOL definition
EOL_THRESHOLD = 0.80

# Train/test split
TRAIN_RATIO = 0.8

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD SOH DATA
# ============================================================

def load_soh(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


# ============================================================
# CLEAN ONE BATTERY
# ============================================================

def clean_series(series):

    series = np.asarray(
        series,
        dtype=np.float32
    )

    series = series[np.isfinite(series)]

    return series


# ============================================================
# SOH SCALER
# ============================================================

class SOHScaler:

    def __init__(self):

        self.mean = None
        self.std = None

    def fit(self, values):

        values = np.asarray(
            values,
            dtype=np.float32
        )

        self.mean = values.mean()
        self.std = values.std()

        if self.std < 1e-8:
            self.std = 1.0

    def transform(self, values):

        return (
            np.asarray(values)
            - self.mean
        ) / self.std

    def inverse_transform(self, values):

        return (
            np.asarray(values)
            * self.std
            + self.mean
        )


# ============================================================
# TRAINING DATASET
# ============================================================

class SOHDataset(Dataset):

    def __init__(
        self,
        series,
        context_length
    ):

        self.series = np.asarray(
            series,
            dtype=np.float32
        )

        self.context_length = context_length

        self.samples = []

        # Example:
        #
        # cycles:
        # 1 2 3 ... 30 31
        #
        # input:
        # 1 ... 30
        #
        # target:
        # 31

        for i in range(
            len(self.series) - context_length
        ):

            x = self.series[
                i:i + context_length
            ]

            y = self.series[
                i + context_length
            ]

            self.samples.append(
                (x, y)
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        x, y = self.samples[index]

        # PatchTST input:
        #
        # [sequence, channels]
        #
        # SOH has one channel.

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(-1)

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        patch_length,
        patch_stride,
        d_model
    ):

        super().__init__()

        self.patch_length = patch_length
        self.patch_stride = patch_stride

        self.projection = nn.Linear(
            patch_length,
            d_model
        )

    def forward(self, x):

        # x:
        #
        # [B, sequence, channels]

        patches = x.unfold(
            dimension=1,
            size=self.patch_length,
            step=self.patch_stride
        )

        # [B, patches, channels, patch_length]

        patches = patches.squeeze(2)

        # [B, patches, patch_length]

        embeddings = self.projection(
            patches
        )

        return embeddings


# ============================================================
# PATCHTST
# ============================================================

class PatchTST(nn.Module):

    def __init__(
        self,
        context_length,
        patch_length,
        patch_stride,
        d_model,
        n_heads,
        num_layers,
        ff_dim,
        dropout
    ):

        super().__init__()

        self.patch_embedding = PatchEmbedding(
            patch_length,
            patch_stride,
            d_model
        )

        self.num_patches = (
            (context_length - patch_length)
            // patch_stride
            + 1
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.head = nn.Linear(
            self.num_patches * d_model,
            1
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        x = (
            x
            + self.position_embedding
        )

        x = self.encoder(x)

        x = self.norm(x)

        # Flatten patches

        x = x.reshape(
            x.shape[0],
            -1
        )

        prediction = self.head(x)

        return prediction.squeeze(-1)


# ============================================================
# BUILD TRAINING DATA
# ============================================================

def build_training_dataset(
    soh_data,
    train_cells,
    scaler
):

    datasets = []

    for cell in train_cells:

        series = clean_series(
            soh_data[cell]
        )

        if len(series) <= CONTEXT_LENGTH:
            continue

        series_scaled = scaler.transform(
            series
        )

        dataset = SOHDataset(
            series_scaled,
            CONTEXT_LENGTH
        )

        datasets.append(dataset)

    if not datasets:

        raise RuntimeError(
            "No training samples found."
        )

    return torch.utils.data.ConcatDataset(
        datasets
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(x)

            loss = criterion(
                prediction,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss += (
                loss.item()
                * x.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                prediction = model(x)

                loss = criterion(
                    prediction,
                    y
                )

                val_loss += (
                    loss.item()
                    * x.size(0)
                )

        val_loss /= len(
            val_loader.dataset
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "checkpoints/patchtst_soh_best.pt"
            )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )


# ============================================================
# PREDICT NEXT CYCLE
# ============================================================

def predict_next_cycle(
    model,
    history,
    scaler
):

    model.eval()

    context = np.asarray(
        history[-CONTEXT_LENGTH:],
        dtype=np.float32
    )

    context_scaled = scaler.transform(
        context
    )

    x = torch.tensor(
        context_scaled,
        dtype=torch.float32
    )

    x = x.unsqueeze(0)
    x = x.unsqueeze(-1)

    x = x.to(DEVICE)

    with torch.no_grad():

        prediction = model(x)

    prediction = prediction.item()

    prediction = scaler.inverse_transform(
        prediction
    )

    return float(prediction)


# ============================================================
# PREDICT BATTERY LIFETIME
# ============================================================

def predict_lifetime(
    model,
    actual_soh,
    start_cycles,
    scaler,
    threshold=0.8,
    max_cycles=2000
):

    # --------------------------------------------------------
    # Start with REAL observations
    # --------------------------------------------------------

    history = list(
        actual_soh[:start_cycles]
    )

    # --------------------------------------------------------
    # Generate one cycle at a time
    # --------------------------------------------------------

    while (
        len(history) < max_cycles
        and history[-1] > threshold
    ):

        next_soh = predict_next_cycle(
            model,
            history,
            scaler
        )

        history.append(
            next_soh
        )

    predicted_curve = np.asarray(
        history,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Find predicted EOL
    # --------------------------------------------------------

    eol_indices = np.where(
        predicted_curve <= threshold
    )[0]

    if len(eol_indices):

        predicted_eol = int(
            eol_indices[0]
        )

    else:

        predicted_eol = None

    return (
        predicted_curve,
        predicted_eol
    )


# ============================================================
# METRICS
# ============================================================

def calculate_rmse(
    actual,
    predicted
):

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def calculate_mape(
    actual,
    predicted
):

    mask = (
        np.abs(actual) > 1e-8
    )

    return float(
        100
        * np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
    )


def find_eol(
    soh,
    threshold=0.8
):

    indices = np.where(
        soh <= threshold
    )[0]

    if len(indices) == 0:

        return None

    return int(
        indices[0]
    )


# ============================================================
# PLOT
# ============================================================

def plot_prediction(
    actual,
    predicted,
    start_cycles,
    cell,
    predicted_eol
):

    plt.figure(
        figsize=(12, 6)
    )

    cycles_actual = np.arange(
        len(actual)
    )

    cycles_predicted = np.arange(
        len(predicted)
    )

    # Real SOH

    plt.plot(
        cycles_actual,
        actual,
        label="Actual SOH"
    )

    # Model prediction

    plt.plot(
        cycles_predicted,
        predicted,
        "--",
        label="PatchTST prediction"
    )

    # Where prediction starts

    plt.axvline(
        start_cycles,
        linestyle=":",
        label="Prediction starts"
    )

    # EOL threshold

    plt.axhline(
        0.8,
        linestyle=":",
        label="EOL threshold"
    )

    if predicted_eol is not None:

        plt.axvline(
            predicted_eol,
            linestyle="--",
            label=f"Predicted EOL: {predicted_eol}"
        )

    plt.xlabel(
        "Cycle"
    )

    plt.ylabel(
        "SOH"
    )

    plt.title(
        f"Battery {cell} - PatchTST SOH Prediction"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.savefig(
        f"outputs/cell_{cell}_prediction.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "PatchTST Battery Lifetime Prediction"
    )

    print(
        "======================================"
    )

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    soh_data = load_soh(
        SOH_FILE
    )

    cells = sorted(
        soh_data.keys()
    )

    print(
        "Total cells:",
        len(cells)
    )

    # --------------------------------------------------------
    # Remove cells with insufficient data
    # --------------------------------------------------------

    valid_cells = []

    for cell in cells:

        series = clean_series(
            soh_data[cell]
        )

        if len(series) > CONTEXT_LENGTH:

            valid_cells.append(
                cell
            )

    cells = valid_cells

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Split by CELL, not by individual cycles.
    #
    # This prevents information leakage.
    # --------------------------------------------------------

    split_index = int(
        len(cells)
        * TRAIN_RATIO
    )

    train_cells = cells[
        :split_index
    ]

    test_cells = cells[
        split_index:
    ]

    print()
    print(
        "Training cells:",
        train_cells
    )

    print(
        "Testing cells:",
        test_cells
    )

    # --------------------------------------------------------
    # Fit scaler ONLY on training cells
    # --------------------------------------------------------

    all_train_soh = []

    for cell in train_cells:

        series = clean_series(
            soh_data[cell]
        )

        all_train_soh.append(
            series
        )

    all_train_soh = np.concatenate(
        all_train_soh
    )

    scaler = SOHScaler()

    scaler.fit(
        all_train_soh
    )

    print()
    print(
        "SOH mean:",
        scaler.mean
    )

    print(
        "SOH std:",
        scaler.std
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = build_training_dataset(
        soh_data,
        train_cells,
        scaler
    )

    # Validation uses the first test cell

    validation_cell = test_cells[0]

    validation_series = clean_series(
        soh_data[validation_cell]
    )

    validation_scaled = scaler.transform(
        validation_series
    )

    validation_dataset = SOHDataset(
        validation_scaled,
        CONTEXT_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print()
    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = PatchTST(
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        patch_stride=PATCH_STRIDE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    )

    model = model.to(
        DEVICE
    )

    print()
    print(model)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print(
        "Starting training..."
    )

    train_model(
        model,
        train_loader,
        validation_loader
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            "checkpoints/patchtst_soh_best.pt",
            map_location=DEVICE
        )
    )

    print()
    print(
        "Best model loaded."
    )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "Testing lifetime prediction"
    )

    print(
        "======================================"
    )

    for cell in test_cells:

        actual = clean_series(
            soh_data[cell]
        )

        if len(actual) <= START_CYCLES:

            print(
                f"Cell {cell}: insufficient cycles"
            )

            continue

        print()
        print(
            f"Testing cell {cell}"
        )

        print(
            f"Total real cycles: {len(actual)}"
        )

        print(
            f"Real cycles provided: {START_CYCLES}"
        )

        # ----------------------------------------------------
        # Predict future lifetime
        # ----------------------------------------------------

        predicted_curve, predicted_eol = (
            predict_lifetime(
                model=model,
                actual_soh=actual,
                start_cycles=START_CYCLES,
                scaler=scaler,
                threshold=EOL_THRESHOLD
            )
        )

        # ----------------------------------------------------
        # Actual EOL
        # ----------------------------------------------------

        actual_eol = find_eol(
            actual,
            EOL_THRESHOLD
        )

        # ----------------------------------------------------
        # Compare overlapping region
        # ----------------------------------------------------

        compare_length = min(
            len(actual),
            len(predicted_curve)
        )

        actual_compare = actual[
            START_CYCLES:compare_length
        ]

        predicted_compare = predicted_curve[
            START_CYCLES:compare_length
        ]

        if len(actual_compare) > 0:

            score_rmse = calculate_rmse(
                actual_compare,
                predicted_compare
            )

            score_mape = calculate_mape(
                actual_compare,
                predicted_compare
            )

        else:

            score_rmse = None
            score_mape = None

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()
        print(
            "--------------------------------------"
        )

        print(
            f"Actual EOL: {actual_eol}"
        )

        print(
            f"Predicted EOL: {predicted_eol}"
        )

        if (
            actual_eol is not None
            and predicted_eol is not None
        ):

            print(
                "EOL error:",
                abs(
                    predicted_eol
                    - actual_eol
                ),
                "cycles"
            )

        print(
            f"RMSE: {score_rmse}"
        )

        print(
            f"MAPE: {score_mape}%"
        )

        print(
            "Predicted total life:",
            predicted_eol
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_prediction(
            actual=actual,
            predicted=predicted_curve,
            start_cycles=START_CYCLES,
            cell=cell,
            predicted_eol=predicted_eol
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        np.savez(
            f"outputs/cell_{cell}_prediction.npz",
            actual=actual,
            predicted=predicted_curve,
            start_cycles=START_CYCLES,
            actual_eol=(
                -1
                if actual_eol is None
                else actual_eol
            ),
            predicted_eol=(
                -1
                if predicted_eol is None
                else predicted_eol
            ),
            rmse=(
                -1
                if score_rmse is None
                else score_rmse
            ),
            mape=(
                -1
                if score_mape is None
                else score_mape
            )
        )


if __name__ == "__main__":

    main()
