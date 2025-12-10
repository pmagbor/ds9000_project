import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, CropForegroundd, Resized,
    RandFlipd, RandAffineD, ConcatItemsd,
)
from monai.networks.nets import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ---------- 1. Dataset / transforms ----------

class MotumMultiDataset(Dataset):
    """
    One row in df has: subject, t1, t2, flair, label.
    We load all three and stack into a 3-channel volume: [3, H, W, D].
    """
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {
            "t1": row["t1"],
            "t2": row["t2"],
            "flair": row["flair"],
            "label": row["label"],
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return {
            "image": data["image"],           # [3, 128,128,128]
            "label": data["label"],           # scalar
            "subject": row["subject"],        # e.g. "sub-0001"
        }


def get_transforms(train=True):
    """
    Load T1/T2/FLAIR, normalize each, crop foreground based on T1,
    resize to 128^3, then concatenate into a 3-channel tensor.
    """
    base = [
        LoadImaged(keys=["t1", "t2", "flair"]),
        EnsureChannelFirstd(keys=["t1", "t2", "flair"]),   # each -> [1,H,W,D]
        ScaleIntensityRanged(
            keys=["t1", "t2", "flair"],
            a_min=0.0,
            a_max=1.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["t1", "t2", "flair"], source_key="t1"),
        Resized(keys=["t1", "t2", "flair"], spatial_size=(128, 128, 128)),
        ConcatItemsd(keys=["t1", "t2", "flair"], name="image"),  # -> [3,H,W,D]
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

def get_model(in_channels=3):
    model = resnet18(
        spatial_dims=3,
        n_input_channels=in_channels,  # 3 = T1,T2,FLAIR
        num_classes=2,                 # 0 = metastasis, 1 = glioma
        pretrained=False,
    )
    return model


# ---------- 3. Training one CV fold on train_val ----------

def train_one_fold(df, train_idx, val_idx, fold, epochs=60, batch_size=2, lr=1e-4):
    print(f"\n===== Fold {fold} (CV on train+val) =====")

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = MotumMultiDataset(df_train, transforms=get_transforms(train=True))
    val_ds   = MotumMultiDataset(df_val,   transforms=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    model = get_model(in_channels=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)       # [B,3,128,128,128]
            labels = batch["label"].long().to(DEVICE)

            logits = model(images)                   # [B,2]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_train_loss = running_loss / len(train_ds)

        # ---- validation AUC ----
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].long().to(DEVICE)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(glioma)
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

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = model.state_dict().copy()
            torch.save(best_state_dict, f"best_resnet_multi_fold{fold}.pt")

    print(f"Best val AUC (fold {fold}): {best_val_auc:.3f}")

    # ---------- Final metrics on this fold ----------
    model.load_state_dict(best_state_dict)
    model.eval()

    fold_probs = []
    fold_labels = []
    fold_preds = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].long().to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(glioma)
            preds = (probs >= 0.5).long()

            fold_probs.append(probs.cpu().numpy())
            fold_labels.append(labels.cpu().numpy())
            fold_preds.append(preds.cpu().numpy())

    fold_probs = np.concatenate(fold_probs)
    fold_labels = np.concatenate(fold_labels)
    fold_preds = np.concatenate(fold_preds)

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

    return fold_auc, fold_acc, prec, rec, fold_labels, fold_probs, fold_preds


# ---------- 4. Train final model on full train+val and evaluate on test ----------

def train_full_and_eval_test(df_train_val, df_test, epochs=60, batch_size=2, lr=1e-4):
    print("\n===== Training final model on full train+val, evaluating on test =====")

    train_ds = MotumMultiDataset(df_train_val, transforms=get_transforms(train=True))
    test_ds  = MotumMultiDataset(df_test,      transforms=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=2)

    model = get_model(in_channels=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- train on full train+val ---
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
        avg_loss = running_loss / len(train_ds)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Final-model Epoch {epoch:03d} | Train loss: {avg_loss:.4f}")

    # --- evaluate on test set ---
    model.eval()
    test_probs = []
    test_labels = []
    test_preds = []
    test_subjects = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].long().to(DEVICE)
            subjects = batch["subject"]          # list/array of strings

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]   # P(glioma)
            preds = (probs >= 0.5).long()

            test_probs.append(probs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_preds.append(preds.cpu().numpy())
            # subjects is a list/tuple of length 1 here (batch=1)
            test_subjects.extend(list(subjects))

    test_probs  = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)
    test_preds  = np.concatenate(test_preds)

    test_auc = roc_auc_score(test_labels, test_probs)
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", pos_label=1
    )

    print("\n===== Test set performance (final model) =====")
    print(f"Test AUC     : {test_auc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test Precision: {test_prec:.3f}")
    print(f"Test Recall  : {test_rec:.3f}")

    cm = confusion_matrix(test_labels, test_preds, labels=[1, 0])
    print("Test confusion matrix (rows: true [Glioma, Metastasis], cols: pred [Glioma, Metastasis]):")
    print(cm)

    # ---- save per-patient predictions ----
    test_df = pd.DataFrame({
        "subject": test_subjects,
        "true_label": test_labels,       # 0 = metastasis, 1 = glioma
        "pred_label": test_preds,
        "prob_glioma": test_probs,
    })
    test_df.to_csv("test_predictions_3dresnet_multi.csv", index=False)
    print("\nSaved test predictions to test_predictions_3dresnet_multi.csv")

    # save arrays and weights if you like
    np.save("test_labels_3dresnet_multi.npy", test_labels)
    np.save("test_probs_3dresnet_multi.npy",  test_probs)
    np.save("test_preds_3dresnet_multi.npy",  test_preds)
    torch.save(model.state_dict(), "final_resnet_multi_trainval.pt")

    return test_auc, test_acc, test_prec, test_rec, cm


# ---------- 5. main: Split -> CV on train+val -> final model on test ----------

def main():
    csv_path = "/home/btchatch/links/scratch/mri/BrainIAC/data/csvs/motum_t1_t2_flair_3d_hgg_vs_bm.csv"
    df = pd.read_csv(csv_path)
    print("Total subjects:", len(df), "Class counts:\n", df["label"].value_counts())

    # --- train_val vs test split (subject-level, stratified) ---
    X_all = np.arange(len(df))
    y_all = df["label"].values

    train_val_idx, test_idx = train_test_split(
        X_all,
        test_size=0.2,           # 80% train+val, 20% test
        stratify=y_all,
        random_state=42,
    )

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test      = df.iloc[test_idx].reset_index(drop=True)

    print("\nTrain+Val subjects:", len(df_train_val), "Class counts:\n", df_train_val["label"].value_counts())
    print("Test subjects:", len(df_test), "Class counts:\n", df_test["label"].value_counts())

    # --- 5-fold CV ONLY on train+val ---
    X_tv = np.arange(len(df_train_val))
    y_tv = df_train_val["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, accs, precs, recs = [], [], [], []
    all_labels = []
    all_probs = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tv, y_tv), 1):
        fold_auc, fold_acc, fold_prec, fold_rec, fold_labels, fold_probs, fold_preds = train_one_fold(
            df_train_val, train_idx, val_idx, fold=fold,
            epochs=60, batch_size=2, lr=1e-4
        )
        aucs.append(fold_auc)
        accs.append(fold_acc)
        precs.append(fold_prec)
        recs.append(fold_rec)

        all_labels.append(fold_labels)
        all_probs.append(fold_probs)
        all_preds.append(fold_preds)

    print("\n=== 5-fold CV on train+val ===")
    print("Per-fold AUCs: ", aucs)
    print("Per-fold Accs:", accs)
    print("Per-fold Prec:", precs)
    print("Per-fold Rec: ", recs)

    print("\nMean ± SD (5-fold CV on train+val):")
    print(f"AUC     : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall  : {np.mean(recs):.3f} ± {np.std(recs):.3f}")

    # aggregate across folds for potential CV plots
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    all_preds  = np.concatenate(all_preds)
    np.save("cv_labels_3dresnet_multi.npy", all_labels)
    np.save("cv_probs_3dresnet_multi.npy",  all_probs)
    np.save("cv_preds_3dresnet_multi.npy",  all_preds)

    # --- final model on full train+val evaluated on test ---
    train_full_and_eval_test(df_train_val, df_test, epochs=60, batch_size=2, lr=1e-4)


if __name__ == "__main__":
    main()
