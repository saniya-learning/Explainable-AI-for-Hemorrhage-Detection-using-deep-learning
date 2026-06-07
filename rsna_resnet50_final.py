# ============================================================
# Cell 1 — Install & Imports
# ============================================================
# !pip install pydicom -q

import os, random, warnings, time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report,
                              balanced_accuracy_score,
                              confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Add via: Notebook → Add Data → rsna-intracranial-hemorrhage-detection
RSNA_PATH   = Path("/kaggle/input/rsna-intracranial-hemorrhage-detection")
TRAIN_DIR   = RSNA_PATH / "stage_2_train"
TRAIN_CSV   = RSNA_PATH / "stage_2_train.csv"
OUTPUT_PATH = Path("/kaggle/working")

CLASS_NAMES = ["Normal", "Epidural", "Intraparenchymal",
               "Intraventricular", "Subarachnoid", "Subdural"]
NUM_CLASSES  = len(CLASS_NAMES)

# ── Hyperparameters ───────────────────────────────────────────────────────────
CONFIG = {
    "image_size"      : 224,   # ResNet50 canonical input size
    "batch_size"      : 32,
    "num_epochs"      : 50,
    "learning_rate"   : 3e-4,
    "dropout"         : 0.5,
    "patience"        : 10,
    "weight_decay"    : 1e-4,
    "warmup_epochs"   : 5,     # freeze backbone for first N epochs
    "max_oversample"  : 4.0,   # cap sampler weight vs Normal (prevents collapse)

    # ── Undersampling (paper-backed) ──────────────────────────────────────────
    # Hssayeni et al. 2020 (Intracranial Hemorrhage Segmentation, Data journal)
    # used random undersampling of the no-hemorrhage class to balance training.
    # Burduja et al. 2020 used stratified sampling to build their training set.
    #
    # RSNA raw class distribution (approximate, from the full 752k dataset):
    #   Normal         : ~592,000  (~78.6%)
    #   Epidural       :  ~13,100  ( ~1.7%)
    #   Intraparenchym :  ~44,500  ( ~5.9%)
    #   Intraventricul :  ~32,500  ( ~4.3%)
    #   Subarachnoid   :  ~52,400  ( ~7.0%)
    #   Subdural       :  ~46,800  ( ~6.2%)  (slices with that as dominant label)
    #
    # Strategy: undersample Normal to N × rarest_hemorrhage_class
    #   where N = UNDERSAMPLE_RATIO.
    # This directly mirrors the paper approach while keeping hemorrhage
    # classes at their natural (full) counts.
    # After undersampling: Normal ≈ UNDERSAMPLE_RATIO × Epidural count.
    "undersample_ratio" : 3,   # Normal = 3 × epidural count after sampling
    "max_hemorrhage_per_class": 12000,  # cap each hemorrhage class (memory)
}

print(f"\nConfig: {CONFIG}")
print(f"Classes: {CLASS_NAMES}")


# ============================================================
# Cell 2 — Parse RSNA CSV → single-label DataFrame
# ============================================================
#
# RSNA is multi-label (one slice can have multiple hemorrhage types).
# We convert to single-label using the following priority order
# (rarest subtype wins when multiple are present — ensures model
#  gets exposure to minority classes):
#
#   Epidural > Intraventricular > Subarachnoid > Intraparenchymal > Subdural
#
# This priority matches the clinical rarity order in the RSNA dataset.

print("Parsing RSNA CSV...")
raw = pd.read_csv(TRAIN_CSV)
print(f"  Raw rows: {len(raw):,}")

raw[['image_id', 'subtype']] = raw['ID'].str.rsplit('_', n=1, expand=True)
pivot = (raw.pivot_table(index='image_id', columns='subtype',
                         values='Label', aggfunc='first')
           .reset_index())
pivot.columns.name = None

cols = ['image_id','any','epidural','intraparenchymal',
        'intraventricular','subarachnoid','subdural']
pivot = pivot[[c for c in cols if c in pivot.columns]].fillna(0)
for c in cols[1:]:
    if c in pivot.columns:
        pivot[c] = pivot[c].astype(int)

# Single-label assignment with rarity priority
PRIORITY = ['epidural','intraventricular','subarachnoid',
            'intraparenchymal','subdural']
LABEL_MAP = {'normal':0,'epidural':1,'intraparenchymal':2,
             'intraventricular':3,'subarachnoid':4,'subdural':5}

def assign_label(row):
    if row.get('any', 0) == 0:
        return 0
    for subtype in PRIORITY:
        if row.get(subtype, 0) == 1:
            return LABEL_MAP[subtype]
    return 0

pivot['label']    = pivot.apply(assign_label, axis=1)
pivot['dcm_path'] = pivot['image_id'].apply(
    lambda x: str(TRAIN_DIR / f"{x}.dcm"))
pivot['exists']   = pivot['dcm_path'].apply(os.path.exists)
pivot = pivot[pivot['exists']].drop(columns=['exists']).reset_index(drop=True)

print(f"\nFull dataset distribution ({len(pivot):,} images):")
for i, name in enumerate(CLASS_NAMES):
    n = (pivot['label'] == i).sum()
    print(f"  {name:<22}: {n:>7,}  ({100*n/len(pivot):.1f}%)")


# ============================================================
# Cell 3 — Undersampling + Scan-level Split
# ============================================================
#
# Step 1: Undersample Normal class
#   - Keep ALL hemorrhage slices (up to max_hemorrhage_per_class)
#   - Undersample Normal to UNDERSAMPLE_RATIO × epidural_count
#   → This is exactly what Hssayeni et al. 2020 describe as
#     "random undersampling of the no-hemorrhage class"
#
# Step 2: Scan-level split using GroupShuffleSplit
#   - Group by scan_id (first 10 chars of image_id = study hash prefix)
#   - Prevents adjacent slices from same CT scan leaking across splits
#   - 80% train / 10% val / 10% test

print("\n── Step 1: Undersampling ────────────────────────────────────────────")

# Cap each hemorrhage class first
hem_parts = []
for label_idx in range(1, NUM_CLASSES):
    cls_df = pivot[pivot['label'] == label_idx]
    n = min(len(cls_df), CONFIG['max_hemorrhage_per_class'])
    hem_parts.append(cls_df.sample(n=n, random_state=SEED))

hem_df   = pd.concat(hem_parts).reset_index(drop=True)
epi_count = (hem_df['label'] == 1).sum()   # epidural = rarest

# Undersample Normal to RATIO × epidural count
normal_df   = pivot[pivot['label'] == 0]
normal_keep = min(len(normal_df), CONFIG['undersample_ratio'] * epi_count)
normal_df   = normal_df.sample(n=normal_keep, random_state=SEED)

balanced = pd.concat([normal_df, hem_df]).reset_index(drop=True)
balanced['scan_id'] = balanced['image_id'].str[:10]

print(f"After undersampling ({len(balanced):,} total):")
for i, name in enumerate(CLASS_NAMES):
    n = (balanced['label'] == i).sum()
    print(f"  {name:<22}: {n:>6,}")

print("\n── Step 2: Scan-level GroupShuffleSplit ─────────────────────────────")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
tr_idx, tmp_idx = next(gss.split(balanced, groups=balanced['scan_id']))
train_df = balanced.iloc[tr_idx].reset_index(drop=True)
temp_df  = balanced.iloc[tmp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['scan_id']))
val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

print("\nTrain class distribution:")
for i, name in enumerate(CLASS_NAMES):
    n = (train_df['label'] == i).sum()
    print(f"  {name:<22}: {n:>6,}")

print("\nVal class distribution:")
for i, name in enumerate(CLASS_NAMES):
    n = (val_df['label'] == i).sum()
    print(f"  {name:<22}: {n:>6,}")


# ============================================================
# Cell 4 — DICOM Windowing & Dataset
# ============================================================
#
# Paper-backed windowing (Burduja et al. 2020, also used by the ensembled
# EfficientNet-B0 paper 2021):
#
#   THREE windows stacked as RGB channels:
#     Channel 0 — Brain window    : WC= 40, WW= 80   → maximises blood vs brain
#     Channel 1 — Subdural window : WC= 80, WW=200   → highlights subdural blood
#     Channel 2 — Bone window     : WC=600, WW=2800   → skull context
#
#   "Head CT images were adjusted to three standard head-CT window settings:
#    brain (WL/WW = 40/80 HU), subdural (80/200 HU), and bone (600/2800 HU),
#    resized to 224×224; and converted into arrays of size 3×224×224."
#   — Ensembled Deep Neural Network for ICH (JAIMS 2021)
#
# This is significantly better than single-channel grayscale because:
#   - Each window highlights different tissue contrast
#   - Fits naturally into 3-channel ResNet without channel surgery
#   - Matches what top-performing RSNA competition models used

WINDOWS = [
    (40,   80),    # Brain
    (80,   200),   # Subdural
    (600,  2800),  # Bone
]

def apply_window(img_hu, wc, ww):
    lo = wc - ww / 2
    hi = wc + ww / 2
    img = np.clip(img_hu, lo, hi)
    img = (img - lo) / (hi - lo)
    return (img * 255).astype(np.uint8)

def dicom_to_3ch(dcm_path, image_size=224):
    """
    Load DICOM → Hounsfield Units → 3-window RGB array (H,W,3) uint8.
    Returns None if file is unreadable.
    """
    try:
        dcm       = pydicom.dcmread(str(dcm_path))
        raw       = dcm.pixel_array.astype(np.float32)
        slope     = float(getattr(dcm, 'RescaleSlope',     1))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        hu        = raw * slope + intercept

        channels = []
        for wc, ww in WINDOWS:
            ch = apply_window(hu, wc, ww)
            ch = cv2.resize(ch, (image_size, image_size),
                            interpolation=cv2.INTER_AREA)
            channels.append(ch)

        return np.stack(channels, axis=2)   # (H, W, 3)
    except Exception:
        return None


class RSNADataset(Dataset):
    """
    Loads RSNA DICOM slices with triple-window preprocessing.

    Train: mild on-the-fly augmentation (paper-backed transforms).
    Val/Test: normalization only. NO horizontal flip anywhere.

    Augmentations used (Burduja et al. 2020 + Wang et al. 2021):
      ✅ rotation ±12°       — patient positioning variation
      ✅ random crop         — zoom/framing differences across scanners
      ✅ brightness/contrast — simulates window-level acquisition variation
      ✅ gaussian blur       — scanner noise simulation
      ❌ horizontal flip     — REMOVED: destroys brain laterality
      ❌ vertical flip       — REMOVED: anatomically meaningless
    """

    def __init__(self, dataframe, image_size=224, split='train'):
        self.df         = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.split      = split

        # ImageNet normalisation — matches ResNet50 pretraining
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        self.val_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row['label'])
        img   = dicom_to_3ch(row['dcm_path'], self.image_size)

        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        tf = self.train_tf if self.split == 'train' else self.val_tf
        return tf(img), label


# ── Build datasets ─────────────────────────────────────────────────────────
print("Building datasets...")
train_dataset = RSNADataset(train_df, CONFIG['image_size'], split='train')
val_dataset   = RSNADataset(val_df,   CONFIG['image_size'], split='val')
test_dataset  = RSNADataset(test_df,  CONFIG['image_size'], split='test')

# ── WeightedRandomSampler ──────────────────────────────────────────────────
# sqrt(1/count) weights, capped at MAX_OVERSAMPLE × Normal weight.
# Lessons learned from vbookshelf experiments:
#   • Full inverse-freq weights → Normal recall collapses to 0
#   • sqrt + cap → stable recall across all classes
#   • 2× num_samples → more minority exposure per epoch
train_labels  = train_df['label'].values
class_counts  = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
sqrt_w        = np.sqrt(1.0 / (class_counts + 1e-6))
cap           = sqrt_w[0] * CONFIG['max_oversample']
capped_w      = np.minimum(sqrt_w, cap)
sample_w      = torch.FloatTensor([capped_w[l] for l in train_labels])

sampler = WeightedRandomSampler(
    weights=sample_w,
    num_samples=2 * len(sample_w),
    replacement=True
)

print(f"\nSampler weights (sqrt, capped at {CONFIG['max_oversample']}× Normal):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<22}: w={capped_w[i]:.4f}  count={int(class_counts[i])}")

# ── DataLoaders ─────────────────────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          sampler=sampler, num_workers=2,
                          pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                          shuffle=False,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'],
                          shuffle=False,  num_workers=2, pin_memory=True)

print(f"\nTrain batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")
print(f"Test  batches : {len(test_loader)}")

# ── Visualise triple-window on 6 samples ───────────────────────────────────
print("\nVisualising triple-window preprocessing on sample images...")
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
win_names  = ['Brain (WC40/WW80)', 'Subdural (WC80/WW200)', 'Bone (WC600/WW2800)']

for col, (_, row) in enumerate(train_df.sample(6, random_state=0).iterrows()):
    img3ch = dicom_to_3ch(row['dcm_path'], 224)
    if img3ch is None:
        continue
    for ch in range(3):
        axes[ch, col].imshow(img3ch[:, :, ch], cmap='gray')
        if col == 0:
            axes[ch, col].set_ylabel(win_names[ch], fontsize=7)
        axes[ch, col].set_title(CLASS_NAMES[row['label']], fontsize=7)
        axes[ch, col].axis('off')

plt.suptitle('Triple-Window DICOM Preprocessing (Brain / Subdural / Bone)',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'triple_window_samples.png', dpi=120)
plt.show()


# ============================================================
# Cell 5 — Focal Loss
# ============================================================
#
# Focal Loss (Lin et al. 2017):
#   gamma=2  → focuses on hard/rare examples; easy Normal examples
#              get down-weighted after a few epochs automatically
#   alpha    → mild per-class weights (clipped at 8×) for initial balance
#
# Combined with capped sampler:
#   Sampler → controls class FREQUENCY per batch (mild, sqrt-capped)
#   Focal   → controls loss MAGNITUDE per sample (hard-example focus)
# Both are mild; neither dominates → no Normal collapse.

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets,
                               weight=self.alpha, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


raw_alpha     = class_counts.sum() / (NUM_CLASSES * class_counts + 1e-6)
clipped_alpha = np.clip(raw_alpha, 1.0, 8.0)
alpha_tensor  = torch.FloatTensor(clipped_alpha).to(DEVICE)
criterion     = FocalLoss(gamma=2.0, alpha=alpha_tensor)

print("Focal Loss alpha (clipped 1-8×):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<22}: {clipped_alpha[i]:.2f}")


# ============================================================
# Cell 6 — ResNet50 Model
# ============================================================
#
# Input: 3-channel triple-windowed CT (not raw grayscale).
# Pretrained ImageNet weights apply directly since input is 3-channel.
#
# Two-stage training:
#   Stage 1 (warmup): backbone frozen → train head only
#     Prevents early noisy gradients corrupting pretrained features.
#   Stage 2 (full):   backbone unfrozen → differential LRs
#     backbone LR = LR/10 (gentle)
#     head    LR = LR     (normal)

class ResNet50ICH(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5):
        super().__init__()
        bb = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features   = nn.Sequential(*list(bb.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Backbone FROZEN   — trainable: {n:,}")

    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Backbone UNFROZEN — trainable: {n:,}")


model = ResNet50ICH(NUM_CLASSES, CONFIG['dropout']).to(DEVICE)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    dummy = torch.randn(4, 3, CONFIG['image_size'], CONFIG['image_size']).to(DEVICE)
    print(f"Forward pass: {model(dummy).shape}")


# ============================================================
# Cell 7 — Optimizer & Scheduler
# ============================================================

def build_optimizer(model, stage=1):
    if stage == 1:
        params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.AdamW(params,
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])
    return optim.AdamW([
        {'params': model.features.parameters(),
         'lr': CONFIG['learning_rate'] / 10},          # backbone: gentle
        {'params': model.classifier.parameters(),
         'lr': CONFIG['learning_rate']},               # head: normal
    ], weight_decay=CONFIG['weight_decay'])


def build_scheduler(optimizer, t_max):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=1e-6)


# ============================================================
# Cell 8 — Train / Evaluate Functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False, desc='  train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds       = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, leave=False, desc='  eval '):
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            preds       = out.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    bal = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), correct / total, bal, all_preds, all_labels


# ============================================================
# Cell 9 — Training Loop
# ============================================================

print("Starting training...\n")
print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | "
      f"{'VLoss':>7} | {'VAcc':>6} | {'BalAcc':>7} | {'t':>5} | Note")
print("─" * 70)

history = {k: [] for k in
           ['train_loss','val_loss','train_acc','val_acc','val_bal_acc']}

best_bal   = 0.0
best_wts   = None
pat_ctr    = 0
stage      = 1

model.freeze_backbone()
optimizer = build_optimizer(model, stage=1)
remaining = CONFIG['num_epochs'] - CONFIG['warmup_epochs']
scheduler = build_scheduler(optimizer, remaining)

for epoch in range(1, CONFIG['num_epochs'] + 1):
    t0 = time.time()

    # ── Stage transition ──────────────────────────────────────────────────
    if stage == 1 and epoch > CONFIG['warmup_epochs']:
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch}: switching to Stage 2 — full fine-tuning")
        print(f"{'='*70}\n")
        stage = 2
        model.unfreeze_backbone()
        optimizer = build_optimizer(model, stage=2)
        scheduler = build_scheduler(optimizer, remaining)

    tr_l, tr_a = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    vl_l, vl_a, bal, vl_p, vl_lb = evaluate(model, val_loader, criterion, DEVICE)

    if stage == 2:
        scheduler.step()

    elapsed = time.time() - t0

    for k, v in zip(['train_loss','val_loss','train_acc','val_acc','val_bal_acc'],
                    [tr_l, vl_l, tr_a, vl_a, bal]):
        history[k].append(v)

    # ── Checkpoint on balanced accuracy (not val loss) ────────────────────
    # Balanced acc = mean per-class recall → directly penalises minority failure
    if bal > best_bal:
        best_bal = bal
        best_wts = deepcopy(model.state_dict())
        pat_ctr  = 0
        note = "★ best"
    else:
        pat_ctr += 1
        note = f"({pat_ctr}/{CONFIG['patience']})"

    print(f"{epoch:>3} | {tr_l:>7.4f} | {tr_a:>6.3f} | "
          f"{vl_l:>7.4f} | {vl_a:>6.3f} | {bal:>7.3f} | "
          f"{elapsed:>5.1f}s | {note}")

    if pat_ctr >= CONFIG['patience']:
        print(f"\nEarly stopping — best balanced acc: {best_bal:.4f}")
        break

model.load_state_dict(best_wts)
torch.save(model.state_dict(), OUTPUT_PATH / "best_rsna_resnet50.pth")
print(f"\nBest balanced accuracy : {best_bal:.4f}")
print(f"Model saved → {OUTPUT_PATH / 'best_rsna_resnet50.pth'}")


# ============================================================
# Cell 10 — Training Curves
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
stage_x = CONFIG['warmup_epochs']

for ax, k_tr, k_vl, title in [
    (axes[0], 'train_loss', 'val_loss',    'Loss'),
    (axes[1], 'train_acc',  'val_acc',     'Accuracy'),
    (axes[2], None,         'val_bal_acc', 'Balanced Accuracy (Val)'),
]:
    if k_tr:
        ax.plot(history[k_tr], label='Train', color='steelblue')
    ax.plot(history[k_vl],
            label=('Val' if k_tr else 'Balanced Acc'),
            color=('coral' if k_tr else 'mediumseagreen'))
    ax.axvline(stage_x, color='gray', linestyle='--',
               alpha=0.6, label='Stage 2 start')
    ax.set_title(title); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('ResNet50 — RSNA ICH (Triple-Window + Undersampling)', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'training_curves.png', dpi=150)
plt.show()


# ============================================================
# Cell 11 — Final Evaluation + Confusion Matrix
# ============================================================

def full_report(model, loader, split_name):
    _, acc, bal, preds, labels = evaluate(model, loader, criterion, DEVICE)
    print(f"\n{'='*60}")
    print(f"  {split_name}  —  Acc: {acc:.4f}  |  Balanced Acc: {bal:.4f}")
    print(f"{'='*60}")
    print(classification_report(labels, preds,
                                 target_names=CLASS_NAMES,
                                 zero_division=0))

    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in [
        (axes[0], cm,   'd',    f'{split_name} — Raw Counts'),
        (axes[1], cm_n, '.2f', f'{split_name} — Row-Normalised (Recall)'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f"cm_{split_name.lower()}.png",
                dpi=150, bbox_inches='tight')
    plt.show()

print("Final Validation Report:")
full_report(model, val_loader, "Validation")

print("\nFinal Test Report:")
full_report(model, test_loader, "Test")
