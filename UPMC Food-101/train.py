import os
import math
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
from tqdm.auto import tqdm

import sys

# ==========================================
# 0. Hyperparameters
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_tqdm = sys.stdout.isatty()

IMG_SIZE = 299
MAX_LENGTH = 20
BATCH_SIZE = 64
EPOCHS = 100

TRAIN_CSV = './content/texts/train_titles.csv'
TEST_CSV  = './content/texts/test_titles.csv'
IMAGE_DIR = './content/images/'

SAVE_PATH = 'best_mars_food_model.pth'

# ====== MARS Loss weights ======
lambda_task = 1   #1.0
lambda_lb = 0.05
lambda_distill = 0.5   #1.0
lambda_noise = 0.01

# ====== Discrepancy-guided sampling ======
warmup_epochs = 3     
temperature = 0.05
alpha = 1              # No EMA
min_prob = 0.05

# ==========================================
# 1. Dataset
# ==========================================
class MultimodalFoodDataset(Dataset):
    def __init__(self, csv_file, base_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file, names=['image_path', 'text', 'food'], header=None)
        self.data['image_path'] = self.data['image_path'].str.strip()
        self.data['food'] = self.data['food'].str.strip()
        self.data['text'] = self.data['text'].fillna("").str.strip()

        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.classes = sorted(self.data['food'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        food_label = row['food']

        img_path = os.path.join(self.base_dir, food_label, row['image_path'])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.base_dir, row['image_path'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer.encode_plus(
            str(row['text']),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = torch.tensor(self.class_to_idx[food_label], dtype=torch.long)
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }


# ==========================================
# 2. MARS-style MoE Utilities
# ==========================================
def cv_squared(x: torch.Tensor) -> torch.Tensor:
    # squared coefficient of variation
    eps = 1e-10
    if x.numel() == 1:
        return torch.zeros(1, device=x.device)
    return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)

def softmax_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    logits: [B, N]
    return probs: [B, N] where only top-k entries are nonzero (softmax among top-k)
    """
    topk_vals, topk_idx = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    probs_topk = F.softmax(topk_vals, dim=1)
    probs = torch.zeros_like(logits)
    probs.scatter_(1, topk_idx, probs_topk)
    return probs

def topk_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    # [B, k]
    return torch.topk(logits, k=min(k, logits.size(1)), dim=1).indices


# ==========================================
# 3. Encoders (Image/Text) + Fusion -> z
# ==========================================
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = models.inception_v3(
            weights='DEFAULT',
            aux_logits=True,
            transform_input=True
        )
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, out_dim) # Keras Dense_128
        

    def forward(self, x):
        out = self.cnn(x)
        if hasattr(out, "logits"):
            out = out.logits
        return out

class TextEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=out_dim, batch_first=True)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        _, (h_n, _) = self.lstm(bert_out.last_hidden_state)
        return h_n[-1]  # [B, out_dim]

class Fusion(nn.Module):
    def __init__(self, in_dim_img=128, in_dim_txt=128, z_dim=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(in_dim_img + in_dim_txt, z_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, img_feat, txt_feat):
        z = torch.cat([img_feat, txt_feat], dim=1)
        return self.fuse(z)  # [B, z_dim]


# ==========================================
# 4. MARS-style MoE Module (Dual Router)
# ==========================================
class ResidualRouter(nn.Module):
    def __init__(self, z_dim, n_experts):
        super().__init__()
        self.logit = nn.Linear(z_dim, n_experts)
        self.sigma = nn.Linear(z_dim, n_experts)

    def forward(self, z_res):
        l = self.logit(z_res)                       # [B,N]
        # sigma > 0
        s = F.softplus(self.sigma(z_res)) + 1e-6     # [B,N]
        return l, s

class FeatureRouter(nn.Module):
    def __init__(self, z_dim, n_experts):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(z_dim, n_experts)
        )

    def forward(self, z):
        return self.mlp(z)

class Expert(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(z_dim, z_dim),
            nn.ReLU()
        )

    def forward(self, z):
        return self.net(z)  # [B,z_dim]

class MARSMoE(nn.Module):
    def __init__(self, num_classes, z_dim=256, n_experts=8, top_k=2, noise_eps=1e-6):
        super().__init__()
        self.z_dim = z_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_eps = noise_eps

        self.res_router = ResidualRouter(z_dim, n_experts)
        self.fea_router = FeatureRouter(z_dim, n_experts)
        self.experts = nn.ModuleList([Expert(z_dim) for _ in range(n_experts)])
        self.head = nn.Linear(z_dim, num_classes)

    def aggregate(self, z, probs):
        """
        z: [B,z_dim]
        probs: [B,N]
        """
        # expert output: [B,z_dim]
        outs = []
        for e in self.experts:
            outs.append(e(z).unsqueeze(1))  # [B,1,z_dim]
        outs = torch.cat(outs, dim=1)       # [B,N,z_dim]
        mix = torch.sum(outs * probs.unsqueeze(-1), dim=1)  # [B,z_dim]
        logits = self.head(mix)
        return logits

    def forward_train(self, z_full, z_partial):
        """
        training forward: full/partial
        """
        # ---- FULL path (deployable routing): feature router on z_full
        l_fea_full = self.fea_router(z_full)                 # [B,N]
        p_fea_full = softmax_topk(l_fea_full, self.top_k)    # [B,N]
        y_full = self.aggregate(z_full, p_fea_full)

        # ---- PARTIAL path (privileged routing): residual router on z_res
        z_res = z_full - z_partial
        l_res, sigma = self.res_router(z_res)                # [B,N], [B,N]

        # noisy top-k routing
        noise = torch.randn_like(l_res)
        l_res_noisy = l_res + noise * sigma
        p_res = softmax_topk(l_res_noisy, self.top_k)        # [B,N]
        y_partial = self.aggregate(z_partial, p_res)

        # ---- feature router on partial for distillation + discrepancy
        l_fea_partial = self.fea_router(z_partial)           # [B,N]

        # load/importance from p_res (Eq.9)
        importance = p_res.sum(dim=0)                        # [N]
        load = (p_res > 0).float().sum(dim=0)                # [N]

        # top-k info for discrepancy metric (sampling)
        T_res = topk_indices(l_res, self.top_k)              # [B,K] (clean logits)
        T_fea = topk_indices(l_fea_partial, self.top_k)      # [B,K]

        return {
            "y_full": y_full,
            "y_partial": y_partial,
            "l_res": l_res,
            "sigma": sigma,
            "l_fea_partial": l_fea_partial,
            "importance": importance,
            "load": load,
            "T_res": T_res,
            "T_fea": T_fea,
        }

    def forward_infer(self, z, use_topk=True):
        """
        Inference: feature router only
        """
        l = self.fea_router(z)
        p = softmax_topk(l, self.top_k) if use_topk else F.softmax(l, dim=1)
        y = self.aggregate(z, p)
        return y


# ==========================================
# 5. Whole Model: encoders + fusion + MARS MoE
# ==========================================
class MARSFoodModel(nn.Module):
    """
    modality: image + text (2 modalities)
    """
    def __init__(self, num_classes, z_dim=256, n_experts=8, top_k=2):
        super().__init__()
        img_dim = 512
        text_dim = 512
        self.img_enc = ImageEncoder(out_dim=img_dim)
        self.txt_enc = TextEncoder(out_dim=text_dim)
        self.fusion = Fusion(in_dim_img=img_dim, in_dim_txt=text_dim, z_dim=z_dim)
        self.moe = MARSMoE(num_classes=num_classes, z_dim=z_dim, n_experts=n_experts, top_k=top_k)

    def encode(self, image, input_ids, attention_mask):
        img_feat = self.img_enc(image)                         # [B,128]
        txt_feat = self.txt_enc(input_ids, attention_mask)      # [B,128]
        return img_feat, txt_feat

    def fuse_with_mask(self, img_feat, txt_feat, mask_img, mask_txt):
        """
        mask_img/mask_txt: [B,1] float 0/1
        """
        img_m = img_feat * mask_img
        txt_m = txt_feat * mask_txt
        z = self.fusion(img_m, txt_m)
        return z

    def forward_train(self, image, input_ids, attention_mask, mask_img, mask_txt):
        # full embedding: always both modalities
        img_feat, txt_feat = self.encode(image, input_ids, attention_mask)
        z_full = self.fusion(img_feat, txt_feat)

        # partial embedding: masked
        z_partial = self.fuse_with_mask(img_feat, txt_feat, mask_img, mask_txt)

        return self.moe.forward_train(z_full, z_partial)

    def forward_infer(self, image, input_ids, attention_mask, mask_img, mask_txt):
        img_feat, txt_feat = self.encode(image, input_ids, attention_mask)
        z = self.fuse_with_mask(img_feat, txt_feat, mask_img, mask_txt)
        return self.moe.forward_infer(z)


# ==========================================
# 6. Discrepancy-aware noise regularization (Eq.13)
# ==========================================
def discrepancy_noise_loss(l_res, sigma, l_fea, top_k):
    """
    l_res: [B,N] clean logits (residual router)
    sigma: [B,N] noise std
    l_fea: [B,N] clean logits (feature router)
    """
    B, N = l_res.shape
    p_res = F.softmax(l_res, dim=1)
    p_fea = F.softmax(l_fea, dim=1)

    T_res = topk_indices(l_res, top_k)   # [B,K]
    T_fea = topk_indices(l_fea, top_k)   # [B,K]

    losses = []
    for i in range(B):
        u = torch.unique(torch.cat([T_res[i], T_fea[i]], dim=0))  # [|U|]
        if u.numel() <= 1:
            continue
        m = torch.abs(p_res[i, u] - p_fea[i, u])                  # [|U|]
        # sort by discrepancy desc
        order = torch.argsort(m, descending=True)
        u_sorted = u[order]

        # enforce sigma^2 order aligned with m order
        sig2 = sigma[i, u_sorted] ** 2 + 1e-12
        log_sig2 = torch.log(sig2)

        # want log_sig2[0] >= log_sig2[1] >= ...
        # penalty when next > curr
        diffs = log_sig2[1:] - log_sig2[:-1]
        losses.append(F.softplus(diffs).mean())

    if len(losses) == 0:
        return torch.zeros(1, device=l_res.device)
    return torch.stack(losses).mean()


# ==========================================
# 7. Modality sampling (2-modality => partial combos: IMG-only, TXT-only)
# ==========================================
# partial combinations:
# 0: IMG-only (mask_img=1, mask_txt=0)
# 1: TXT-only (mask_img=0, mask_txt=1)
PARTIAL_COMBOS = [
    (1.0, 0.0),
    (0.0, 1.0),
]

def sample_partial_masks(batch_size, probs, device):
    """
    probs: np.array shape [2] for two partial combos
    return mask_img, mask_txt (each [B,1]), combo_idx [B]
    """
    combo_idx = np.random.choice(len(PARTIAL_COMBOS), size=batch_size, p=probs)
    mask_img = torch.zeros(batch_size, 1, device=device)
    mask_txt = torch.zeros(batch_size, 1, device=device)
    for i, c in enumerate(combo_idx):
        mi, mt = PARTIAL_COMBOS[c]
        mask_img[i, 0] = mi
        mask_txt[i, 0] = mt
    return mask_img, mask_txt, torch.tensor(combo_idx, device=device, dtype=torch.long)


def topk_disagreement(T_res, T_fea, k):
    """
    T_res/T_fea: [B,K] indices
    return mismatch scalar per sample: [B]
    d_i = 1 - |intersection|/K
    """
    B = T_res.size(0)
    out = torch.zeros(B, device=T_res.device)
    for i in range(B):
        inter = len(set(T_res[i].tolist()).intersection(set(T_fea[i].tolist())))
        out[i] = 1.0 - (inter / float(k))
    return out


# ==========================================
# 8. Eval: Accuracy per Modality
# ==========================================
@torch.no_grad()
def evaluate_by_combo(model: MARSFoodModel, loader, device):
    """
    3 combos:
      - IMG-only
      - TXT-only
      - IMG+TXT
    """
    model.eval()

    combos = {
        "IMG": (1.0, 0.0),
        "TXT": (0.0, 1.0),
        "IMG+TXT": (1.0, 1.0),
    }

    correct = {k: 0 for k in combos}
    total = {k: 0 for k in combos}

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch['image'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        B = labels.size(0)

        for name, (mi, mt) in combos.items():
            mask_img = torch.full((B, 1), mi, device=device)
            mask_txt = torch.full((B, 1), mt, device=device)

            # logits = model.forward_infer(images, ids, mask, mask_img, mask_txt)
            logits = model.forward_train(images, ids, mask, mask_img, mask_txt); logits = logits["y_partial"]
            pred = logits.argmax(dim=1)
            correct[name] += (pred == labels).sum().item()
            total[name] += B

    acc = {k: (100.0 * correct[k] / max(1, total[k])) for k in combos}
    avg = sum(acc.values()) / len(acc)
    return acc, avg


# ==========================================
# 9. Training Loop (MARS)
# ==========================================
def run_training():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MultimodalFoodDataset(TRAIN_CSV, os.path.join(IMAGE_DIR, 'train'), tokenizer, train_transform)
    test_dataset  = MultimodalFoodDataset(TEST_CSV,  os.path.join(IMAGE_DIR, 'test'),  tokenizer, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.classes)

    # MoE Hyperparams
    Z_DIM = 1024
    N_EXPERTS = 8
    TOP_K = 3

    model = MARSFoodModel(num_classes=num_classes, z_dim=Z_DIM, n_experts=N_EXPERTS, top_k=TOP_K).to(DEVICE)

    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.txt_enc.bert.parameters(), 'lr': 2e-5},
        {'params': model.img_enc.parameters(),      'lr': 1e-4},
        {'params': model.txt_enc.lstm.parameters(), 'lr': 1e-3},
        {'params': model.fusion.parameters(),       'lr': 1e-3},
        {'params': model.moe.parameters(),          'lr': 1e-3},
    ], weight_decay=0.001)

    criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.1)



    # partial combo sampling prob: [IMG-only, TXT-only]
    sampling_prob = np.array([0.5, 0.5], dtype=np.float64)

    best_avg_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # epoch statistics for sampling update
        # mismatch sum/count per partial combo
        combi_mismatch_sum = np.zeros(len(PARTIAL_COMBOS), dtype=np.float64)
        combi_count = np.zeros(len(PARTIAL_COMBOS), dtype=np.float64)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        running_loss = 0.0
        running_acc = 0
        running_total = 0

        for batch in pbar:
            images = batch['image'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            amask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            B = labels.size(0)

            # --- sample partial modality masks for this batch
            if epoch < warmup_epochs:
                probs = np.array([0.5, 0.5], dtype=np.float64)
            else:
                probs = sampling_prob

            mask_img, mask_txt, combo_idx = sample_partial_masks(B, probs, DEVICE)

            optimizer.zero_grad()

            out = model.forward_train(images, ids, amask, mask_img, mask_txt)

            y_full = out["y_full"]
            y_partial = out["y_partial"]

            # Task loss (full + partial)
            loss_task = criterion(y_full, labels) + criterion(y_partial, labels)

            # Load balance (importance/load from residual routing)
            importance = out["importance"]
            load = out["load"]
            loss_lb = cv_squared(importance) + cv_squared(load)

            # Distillation KL (feature partial -> residual)
            l_fea = out["l_fea_partial"]
            l_res = out["l_res"].detach()
            loss_distill = F.kl_div(
                F.log_softmax(l_fea, dim=1),
                F.softmax(l_res, dim=1),
                reduction='batchmean'
            )

            # Discrepancy-aware noise regularization
            loss_noise = discrepancy_noise_loss(
                l_res=out["l_res"],
                sigma=out["sigma"],
                l_fea=out["l_fea_partial"],
                top_k=TOP_K
            )

            loss = (lambda_task * loss_task
                    + lambda_lb * loss_lb
                    + lambda_distill * loss_distill
                    + lambda_noise * loss_noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            # train accuracy using full path logits
            pred = y_full.argmax(dim=1)
            running_total += B
            running_acc += (pred == labels).sum().item()

            # mismatch measure (for sampling update): top-k overlap
            mismatch_vec = topk_disagreement(out["T_res"], out["T_fea"], TOP_K)  # [B]
            for i in range(B):
                c = int(combo_idx[i].item())
                combi_mismatch_sum[c] += float(mismatch_vec[i].item())
                combi_count[c] += 1.0

            pbar.set_postfix(
                loss=f"{running_loss / max(1, (running_total / BATCH_SIZE)):.4f}",
                acc=f"{100.0 * running_acc / max(1, running_total):.2f}%",
                p_img=f"{sampling_prob[0]:.2f}",
                p_txt=f"{sampling_prob[1]:.2f}",
            )

        # --- update sampling_prob after warmup
        if epoch >= warmup_epochs:
            # Average mismatch
            avg_mismatch = np.where(combi_count > 0, combi_mismatch_sum / combi_count, 0.0)
            # softmax( d / tau )
            new_prob = np.exp(avg_mismatch / max(1e-8, temperature))
            new_prob = new_prob / (new_prob.sum() + 1e-12)

            # EMA + floor
            updated = alpha * new_prob + (1.0 - alpha) * sampling_prob
            updated = np.maximum(updated, min_prob)
            sampling_prob = updated / updated.sum()

        # ===== Validation: print accuracy per modality combination
        acc_dict, avg_acc = evaluate_by_combo(model, test_loader, DEVICE)
        print(f"\n[Epoch {epoch+1}] Combo Acc: "
              f"IMG={acc_dict['IMG']:.2f} | TXT={acc_dict['TXT']:.2f} | IMG+TXT={acc_dict['IMG+TXT']:.2f} | AVG={avg_acc:.2f}")

        scheduler.step(avg_acc)

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Best updated! saved to {SAVE_PATH}")

    print(f"\nTraining done. Best AVG Acc = {best_avg_acc:.2f}%")


if __name__ == "__main__":
    run_training()
