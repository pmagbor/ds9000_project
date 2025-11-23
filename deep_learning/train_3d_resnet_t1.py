import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, CropForegroundd, Resized,
    RandFlipd, RandAffineD,
)
from monai.networks.nets import resnet18

import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ---------- 1. Dataset / transforms ----------

class MotumT1Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {"image": row["path"], "label": row["label"]}
        if self.transforms is not None:
            data = self.transforms(data)
        return data


def get_transforms(train=True):
    # Images already registered to MNI & 1mm; we standardize intensity & size.
    base = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),        # [1, H, W, D]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0, a_max=1.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),  # fixed size
    ]

    if train:
        aug = [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandAffineD(
                keys=["image"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
        ]
        return Compose(base + aug)
    else:
        return Compose(base)


# ---------- 2. Model ----------

def get_model(in_channels=1):
    model = resnet18(
        spatial_dims=3,                 # 3D model
        n_input_channels=in_channels,   # 1 for T1-only
        num_classes=2,                  # HGG vs BM
        pretrained=False,
    )
    return model


# ---------- 3. Helper plotting functions ----------

def plot_confusion_matrix(labels, preds, class_names=("BM", "HGG"),
                          title="3D ResNet T1 – Confusion Matrix",
                          fname="confusion_matrix_3dresnet_t1.png"):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)


def plot_roc_curve(labels, probs,
                   title="3D ResNet T1 – ROC Curve",
                   fname="roc_curve_3dresnet_t1.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)


# ---------- 4. Training one fold, with metrics ----------

def train_one_fold(df, train_idx, val_idx, fold, epochs=60, batch_size=2, lr=1e-4):
    print(f"\n===== Fold {fold} =====")

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = MotumT1Dataset(df_train, transforms=get_transforms(train=True))
    val_ds   = MotumT1Dataset(df_val,   transforms=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -np.inf
    best_state_dict = None

    # -------- training loop with AUC-based model selection --------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].long().to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_train_loss = running_loss / len(train_ds)

        # ---- validation for AUC ----
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].long().to(DEVICE)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(HGG)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = float("nan")

        print(
            f"Epoch {epoch:03d} | Train loss: {avg_train_loss:.4f} | "
            f"Val AUC: {val_auc:.3f}"
        )

        # update best checkpoint if AUC improved (and is not NaN)
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = model.state_dict().copy()
            torch.save(
                best_state_dict,
                f"best_resnet_t1_fold{fold}.pt",
            )

    print(f"Best val AUC (fold {fold}): {best_val_auc:.3f}")

    # ---------- Final metrics on this fold using best model ----------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        # fallback (shouldn't usually happen)
        best_state_dict = model.state_dict()

    model.eval()
    fold_probs = []
    fold_labels = []
    fold_preds = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].long().to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(HGG)
            preds = (probs >= 0.5).long()               # threshold at 0.5

            fold_probs.append(probs.cpu().numpy())
            fold_labels.append(labels.cpu().numpy())
            fold_preds.append(preds.cpu().numpy())

    fold_probs = np.concatenate(fold_probs)
    fold_labels = np.concatenate(fold_labels)
    fold_preds = np.concatenate(fold_preds)

    # Metrics: AUC, accuracy, precision, recall
    fold_auc = roc_auc_score(fold_labels, fold_probs)
    fold_acc = accuracy_score(fold_labels, fold_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        fold_labels, fold_preds, average="binary", pos_label=1
    )

    print(
        f"Fold {fold} metrics: "
        f"AUC={fold_auc:.3f}, Acc={fold_acc:.3f}, "
        f"Prec={prec:.3f}, Rec={rec:.3f}"
    )

    # return metrics AND per-sample predictions
    return fold_auc, fold_acc, prec, rec, fold_labels, fold_probs, fold_preds


# ---------- 5. main: 5-fold CV and aggregate metrics ----------

def main():
    csv_path = "/home/btchatch/links/scratch/mri/BrainIAC/data/csvs/motum_t1_3d_hgg_vs_bm.csv"
    df = pd.read_csv(csv_path)
    print("Total subjects:", len(df), "Class counts:\n", df["label"].value_counts())

    X = np.arange(len(df))
    y = df["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, accs, precs, recs = [], [], [], []

    # lists to accumulate per-fold predictions (for pooled confusion matrix & ROC)
    all_labels = []
    all_probs = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_auc, fold_acc, fold_prec, fold_rec, fold_labels, fold_probs, fold_preds = train_one_fold(
            df, train_idx, val_idx, fold=fold,
            epochs=60, batch_size=2, lr=1e-4
        )
        aucs.append(fold_auc)
        accs.append(fold_acc)
        precs.append(fold_prec)
        recs.append(fold_rec)

        all_labels.append(fold_labels)
        all_probs.append(fold_probs)
        all_preds.append(fold_preds)

    print("\nPer-fold AUCs: ", aucs)
    print("Per-fold Accs:", accs)
    print("Per-fold Prec:", precs)
    print("Per-fold Rec: ", recs)

    print("\nMean ± SD (5-fold CV):")
    print(f"AUC     : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall  : {np.mean(recs):.3f} ± {np.std(recs):.3f}")

    # --------- aggregate across folds for figures ---------
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    all_preds  = np.concatenate(all_preds)

    # save them if you like
    np.save("all_labels_3dresnet_t1.npy", all_labels)
    np.save("all_probs_3dresnet_t1.npy", all_probs)
    np.save("all_preds_3dresnet_t1.npy", all_preds)

    # now make confusion matrix + ROC plots
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(all_labels, all_probs)


if __name__ == "__main__":
    main()
