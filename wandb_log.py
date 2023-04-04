import wandb
import random
from config import Args
import numpy as np

args = Args()
args.load("config/train_set_official.yml")
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="SER_depression",
    name="jsd",
    # track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "architecture": "set_official",
        "dataset": "MODMA",
        "epochs": args.epochs,
    }
)
train_metric = np.load(
    "results/data/SET_official_train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0008_pretrainFalse_train_metric.npy",
    allow_pickle=True).item()
# simulate training
train_acc = train_metric['train_acc']
train_loss = train_metric['train_loss']
val_acc = train_metric['val_acc']
val_loss = train_metric['val_loss']
for acc, loss, acc_v, loss_v in zip(train_acc, train_loss, val_acc, val_loss):
    wandb.log({'train accuracy': acc, "train loss": loss, "val accuracy": acc_v, "val loss": loss_v})
art = wandb.Artifact("my-object-detector", type="model")
art.add_file("models/train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0002_pretrainFalse_best.pt")
wandb.log_artifact(art)
# wandb.save("models/train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0002_pretrainFalse_best.pt")
# wandb.log({'model name': "set_official"})
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
