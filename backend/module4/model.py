"""
backend/module4/model.py — NutriIngredientNet v6 inference.

The model, preprocessing and analysis pipeline from `module4/app.py`, lifted
out of that file's FastAPI app so the routes can live on the shared server.
The maths is transcribed unchanged — pseudo-depth (Depth Anything V2, global
scale) -> 4-ch ConvNeXt -> 3 heads -> temperature calibration -> blend ->
recalibration -> Atwater-exact reconciliation — because any edit here would
change reported nutrition numbers.

The one behavioural change is WHEN the weights load. `module4/app.py` loaded
them in an `@app.on_event("startup")` hook; doing that here would block the
shared server's boot on a ~200 MB Hugging Face download, leaving the Module 1
+ Module 2 workout pages unreachable until it finished (and failing the whole
server if the machine were offline). `ensure_loaded()` instead loads on first
use, under a lock so concurrent first requests load exactly once.
"""

import json
import threading

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from backend.module4.settings import HF_REPO, HF_TOKEN, USE_DEPTH, USE_TTA, env_str

_forced = env_str("DEVICE", "auto").lower()
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _forced == "auto" else _forced

IMN_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMN_STD = torch.tensor([0.229, 0.224, 0.225])

S = {}                      # global model state, populated by load_artifacts()
_load_lock = threading.Lock()


# ---------------------------------------------------------------- model
class NutriNet(nn.Module):
    """Must match the training class so state_dict keys line up."""

    def __init__(self, backbone, in_ch, K, mean_gram, mean_tot):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=False,
                                     num_classes=0, global_pool="avg")
        feat = self.enc.num_features
        if in_ch != 3:
            self._inflate_first_conv(in_ch)
        self.pres = nn.Linear(feat, K)
        self.gram = nn.Linear(feat, K)
        self.tot = nn.Linear(feat, 4)
        self.register_buffer("prior_g", torch.tensor(mean_gram, dtype=torch.float32))
        self.register_buffer("prior_t", torch.tensor(mean_tot, dtype=torch.float32))

    def _inflate_first_conv(self, in_ch):
        for name, mod in self.enc.named_modules():
            if isinstance(mod, nn.Conv2d):
                w = mod.weight.data
                new = nn.Conv2d(in_ch, mod.out_channels, mod.kernel_size, mod.stride,
                                mod.padding, bias=mod.bias is not None)
                with torch.no_grad():
                    new.weight[:, :3] = w
                    new.weight[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, in_ch - 3, 1, 1)
                    if mod.bias is not None:
                        new.bias.copy_(mod.bias)
                parent = self.enc
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, new)
                break

    def forward(self, x):
        h = self.enc(x)
        g = F.softplus(self.gram(h)) * self.prior_g
        t = F.softplus(self.tot(h)) * self.prior_t
        return self.pres(h), g, t


# ---------------------------------------------------------------- load
def load_artifacts():
    from huggingface_hub import hf_hub_download

    cfg = json.load(open(hf_hub_download(HF_REPO, "config.json", token=HF_TOKEN)))
    wts = hf_hub_download(HF_REPO, "state_dict.pt", token=HF_TOKEN)

    vocab = cfg["vocab"]
    K = len(vocab)
    dens = cfg["density"]
    model = NutriNet(cfg["backbone"], cfg["in_ch"], K,
                     np.asarray(cfg["mean_gram"], np.float32),
                     np.asarray(cfg["mean_tot"], np.float32))
    model.load_state_dict(torch.load(wts, map_location="cpu"))
    model.eval().to(DEVICE)

    S.update(
        cfg=cfg, model=model, K=K, vocab=vocab,
        names=[cfg["names"].get(i, i) for i in vocab],
        cal_pg=torch.tensor([dens[i]["cal_pg"] for i in vocab], dtype=torch.float32, device=DEVICE),
        fat_pg=torch.tensor([dens[i]["fat_pg"] for i in vocab], dtype=torch.float32, device=DEVICE),
        carb_pg=torch.tensor([dens[i]["carb_pg"] for i in vocab], dtype=torch.float32, device=DEVICE),
        prot_pg=torch.tensor([dens[i]["prot_pg"] for i in vocab], dtype=torch.float32, device=DEVICE),
        depth_pipe=None,
    )
    want_depth = bool(cfg.get("use_pseudo_depth")) and cfg.get("in_ch", 3) == 4
    if want_depth and USE_DEPTH:
        try:
            from transformers import pipeline
            S["depth_pipe"] = pipeline(
                "depth-estimation",
                model=cfg.get("depth_model",
                              "depth-anything/Depth-Anything-V2-Small-hf"),
                device=0 if DEVICE == "cuda" else -1)
            print("[NUTRI] Depth Anything V2 loaded (pseudo-depth ON).")
        except Exception as e:
            print("[NUTRI] warn: depth model unavailable; padding a neutral channel:", e)
    print(f"[NUTRI] Loaded {HF_REPO}: {cfg['backbone']}, K={K}, in_ch={cfg['in_ch']}, "
          f"T={cfg['temperature']:.3f}, thr={cfg['threshold']:.2f}, "
          f"lambda={cfg['crc_lambda']:.2f}, w={cfg['w_blend']:.2f}, "
          f"recal a={cfg.get('recal_a', 1.0):.2f} b={cfg.get('recal_b', 0.0):+.0f}, "
          f"TTA={USE_TTA}, device={DEVICE}")


def is_loaded() -> bool:
    return "model" in S


def ensure_loaded() -> None:
    """
    Load the weights on first use.

    Double-checked under a lock: two requests arriving together must not both
    start a ~200 MB download and race to populate `S`.
    """
    if is_loaded():
        return
    with _load_lock:
        if not is_loaded():
            load_artifacts()


# ---------------------------------------------------------------- preprocess
def build_input(img: Image.Image) -> torch.Tensor:
    cfg = S["cfg"]
    sz = cfg["img_size"]
    im = img.convert("RGB").resize((sz, sz), Image.BILINEAR)
    x = TF.to_tensor(im)
    x = (x - IMN_MEAN[:, None, None]) / IMN_STD[:, None, None]
    if cfg.get("in_ch", 3) == 4:
        if S.get("depth_pipe") is not None:
            d = np.asarray(S["depth_pipe"](im)["depth"], np.float32)
            lo = float(cfg.get("depth_lo", 0.0))
            hi = float(cfg.get("depth_hi", 1.0))
            d = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)     # GLOBAL scale (v6)
            d = torch.from_numpy(d)[None]
        else:
            d = torch.full((1, sz, sz), 0.5)                          # neutral fallback
        x = torch.cat([x, (d - 0.5) / 0.25], 0)
    return x.unsqueeze(0)


# ---------------------------------------------------------------- predict
@torch.no_grad()
def analyze(img: Image.Image) -> dict:
    cfg, model = S["cfg"], S["model"]
    x = build_input(img).to(DEVICE)

    if USE_TTA:
        L = G = T_ = None
        for k in range(4):
            pl, gm, td = model(torch.rot90(x, k, dims=[2, 3]))
            L = pl if L is None else L + pl
            G = gm if G is None else G + gm
            T_ = td if T_ is None else T_ + td
        pl, gm, td = L / 4, G / 4, T_ / 4
    else:
        pl, gm, td = model(x)

    prob = torch.sigmoid(pl / cfg["temperature"])[0]
    gm, td = gm[0], td[0]

    soft_g = gm * prob
    soft_cal = float((soft_g * S["cal_pg"]).sum())
    direct = {"cal": float(td[0]), "fat": float(td[1]),
              "carb": float(td[2]), "prot": float(td[3])}

    w = float(cfg["w_blend"])
    blend = w * soft_cal + (1 - w) * direct["cal"]
    blend = max(1.0, float(cfg.get("recal_a", 1.0)) * blend + float(cfg.get("recal_b", 0.0)))

    lo, hi = cfg.get("s_clip", [0.4, 2.5])
    s = float(np.clip(blend / max(soft_cal, 1e-6), lo, hi))
    recon_g = soft_g * s

    fat = float((recon_g * S["fat_pg"]).sum())
    carb = float((recon_g * S["carb_pg"]).sum())
    prot = float((recon_g * S["prot_pg"]).sum())
    A = cfg.get("atwater", {"fat": 9.0, "carb": 4.0, "prot": 4.0})
    cal = A["fat"] * fat + A["carb"] * carb + A["prot"] * prot   # exact by construction

    lam, thr = float(cfg["crc_lambda"]), float(cfg["threshold"])
    rows = []
    pr = prob.cpu().numpy()
    rg = recon_g.cpu().numpy()
    for k in range(S["K"]):
        if pr[k] <= lam and pr[k] < thr:
            continue
        g = float(rg[k])
        rows.append(dict(
            name=S["names"][k],
            status="certain" if pr[k] > thr else "possible",
            confidence=round(float(pr[k]), 3),
            grams=round(g, 1),
            kcal=round(g * float(S["cal_pg"][k]), 1),
            fat=round(g * float(S["fat_pg"][k]), 1),
            carb=round(g * float(S["carb_pg"][k]), 1),
            protein=round(g * float(S["prot_pg"][k]), 1),
        ))
    rows.sort(key=lambda r: -r["kcal"])

    return dict(
        # ---- TOTALS FIRST (this is the headline answer) ----
        totals=dict(calories=round(cal, 1), fat_g=round(fat, 1),
                    carbs_g=round(carb, 1), protein_g=round(prot, 1),
                    mass_g=round(float(recon_g.sum()), 1)),
        # ---- then the itemised breakdown ----
        ingredients=rows,
        atwater_check=dict(
            from_macros=round(A["fat"] * fat + A["carb"] * carb + A["prot"] * prot, 1),
            calories=round(cal, 1), consistent=True),
        diagnostics=dict(direct_head=direct, bottom_up_kcal=round(soft_cal, 1),
                         blend_kcal=round(blend, 1), scale_factor=round(s, 3),
                         tta=USE_TTA, pseudo_depth=S.get("depth_pipe") is not None),
        accuracy=dict(
            test_kcal_mae=73.5, test_kcal_mae_pct=29.1,
            note=("Mean absolute error on the official Nutrition5k RGB test split. "
                  "A single photo cannot resolve hidden or stacked food, so treat the "
                  "numbers as a good estimate, not a measurement.")),
        warning=("Trained on a FIXED overhead camera rig (Nutrition5k). Photos taken from "
                 "other angles or unknown distances will be less accurate. Research demo — "
                 "not medical or dietary advice."),
    )
