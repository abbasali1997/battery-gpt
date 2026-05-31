import os
import pickle
import torch
from torch.utils.data import DataLoader

from batterygpt_core import BatteryGPT, BatteryGPTConfig
from dataset_battery import BatterySequenceDataset
from data.battery.batteryData import loadDataFile

# -------------------------
# Config
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "out-battery"
os.makedirs(out_dir, exist_ok=True)

block_size = 256
batch_size = 16
max_epochs = 10
learning_rate = 1e-3
weight_decay = 0.1
eval_every = 500

# -------------------------
# Load data
# -------------------------
raw_data = loadDataFile("data/battery/MIT_lenth.db")

with open("data/battery/meta.pkl", "rb") as f:
    meta = pickle.load(f)

meta_vocab_size = int(meta["vocab_size"])
data_max_token = max(max(int(x) for x in seq) for seq in raw_data)
vocab_size = max(meta_vocab_size, data_max_token + 1)

print("meta vocab_size:", meta_vocab_size, flush=True)
print("data max token:", data_max_token, flush=True)
print("using vocab_size:", vocab_size, flush=True)

train_ds = BatterySequenceDataset(raw_data, block_size=block_size, split="train")
val_ds = BatterySequenceDataset(raw_data, block_size=block_size, split="val")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

print("train windows:", len(train_ds))
print("val windows:", len(val_ds))

# -------------------------
# Build model
# -------------------------
config = BatteryGPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=24,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True,
)

model = BatteryGPT(config).to(device)
optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(0.9, 0.95),
)

scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

# -------------------------
# Eval helper
# -------------------------
@torch.no_grad()
def estimate_loss(loader, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# -------------------------
# Train loop
# -------------------------
step = 0
best_val = float("inf")

for epoch in range(max_epochs):
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"epoch={epoch} step={step} train_loss={loss.item():.4f}")

        if step % eval_every == 0:
            train_loss = estimate_loss(train_loader, max_batches=20)
            val_loss = estimate_loss(val_loader, max_batches=20)
            print(f"[eval] step={step} train={train_loss:.4f} val={val_loss:.4f}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": config.__dict__,
                "step": step,
                "epoch": epoch,
                "val_loss": val_loss,
            }

            torch.save(ckpt, os.path.join(out_dir, "current_ckpt.pt"))

            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, os.path.join(out_dir, "best_ckpt.pt"))
                print(f"saved best checkpoint to {out_dir}/best_ckpt.pt")

        step += 1

print("training complete")
