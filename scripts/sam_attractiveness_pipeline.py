# ============================================================
# SAM Feature Extraction + Attractiveness Analysis
# Integrates with existing Claude scores & human ratings
# Run on Google Colab (GPU recommended)
# ============================================================

# ── Step 0: Install dependencies ────────────────────────────
# !pip install git+https://github.com/facebookresearch/segment-anything.git
# !pip install opencv-python pycocotools matplotlib
# !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── CONFIG ───────────────────────────────────────────────────
URBAN_DIR     = "/content/drive/MyDrive/Capstone/urban_400"
LANDSCAPE_DIR = "/content/drive/MyDrive/Capstone/landscape_400"
CLAUDE_CSV    = "/content/drive/MyDrive/Capstone/all_images_15_properties.csv"
VOTES_TSV     = "/content/drive/MyDrive/Capstone/votes.tsv"
SAM_CKPT      = "/content/sam_vit_h_4b8939.pth"
SAM_FEATURES_CACHE = "/content/drive/MyDrive/Capstone/sam_features.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Step 1: Load SAM model ───────────────────────────────────
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,          # lower = faster, still good coverage
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=500,    # ignore tiny noise segments
)

# ── Step 2: Feature extraction from SAM masks ────────────────

def compute_compactness(mask):
    """4π·area / perimeter² — 1.0 = perfect circle, lower = irregular"""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return 0.0
    return (4 * np.pi * area) / (perim ** 2)

def compute_symmetry(mask):
    """Horizontal + vertical flip overlap as symmetry proxy (0–1)"""
    h_flip = np.fliplr(mask)
    v_flip = np.flipud(mask)
    h_sym = np.logical_and(mask, h_flip).sum() / (mask.sum() + 1e-6)
    v_sym = np.logical_and(mask, v_flip).sum() / (mask.sum() + 1e-6)
    return float((h_sym + v_sym) / 2)

def extract_sam_features(image_path):
    """
    Returns a dict of structural features derived from SAM masks.
    All features are scalar values suitable for regression.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]
    total_px = H * W

    try:
        masks = mask_generator.generate(img_rgb)
    except Exception as e:
        print(f"SAM failed on {image_path}: {e}")
        return None

    if not masks:
        return None

    areas      = np.array([m['area'] for m in masks])
    area_fracs = areas / total_px
    n_masks    = len(masks)

    # --- Complexity / Levels of Scale ---
    n_segments        = n_masks                          # raw segment count
    log_n_segments    = np.log1p(n_masks)               # log-scaled

    # --- Size distribution (Positive Space / Dominant region) ---
    dominant_frac     = area_fracs.max()                 # largest segment fraction
    mean_segment_frac = area_fracs.mean()
    std_segment_frac  = area_fracs.std()
    size_entropy      = -np.sum(area_fracs * np.log(area_fracs + 1e-9))  # diversity

    # --- Coverage (Positive Space / The Void) ---
    combined_mask = np.zeros((H, W), dtype=bool)
    for m in masks:
        combined_mask |= m['segmentation']
    coverage_frac = combined_mask.sum() / total_px       # 1 = no void, 0 = empty

    # --- Boundaries / Edge density ---
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges    = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (255 * total_px)        # fraction of edge pixels

    # --- Shape quality (Good Shape proxy) ---
    compactness_scores = [compute_compactness(m['segmentation']) for m in masks]
    mean_compactness   = np.mean(compactness_scores)
    max_compactness    = np.max(compactness_scores)

    # --- Symmetry (Local Symmetries proxy) ---
    # Use top-3 largest masks for efficiency
    top_masks = sorted(masks, key=lambda m: m['area'], reverse=True)[:3]
    sym_scores = [compute_symmetry(m['segmentation']) for m in top_masks]
    mean_symmetry = np.mean(sym_scores)

    # --- Spatial distribution (Strong Centers) ---
    # Weighted centroid distance from image center
    cx, cy = W / 2, H / 2
    centroid_dists = []
    for m in masks:
        ys, xs = np.where(m['segmentation'])
        if len(xs) == 0:
            continue
        dist = np.sqrt((xs.mean() - cx)**2 + (ys.mean() - cy)**2)
        centroid_dists.append(dist / np.sqrt(cx**2 + cy**2))  # normalised
    center_bias = 1 - np.mean(centroid_dists)  # higher = more centered

    # --- Gradient / Gradients proxy ---
    # Colour gradient magnitude
    img_float = img_rgb.astype(np.float32)
    grad_x = np.gradient(img_float, axis=1)
    grad_y = np.gradient(img_float, axis=0)
    gradient_magnitude = np.sqrt((grad_x**2 + grad_y**2).sum(axis=2)).mean()
    gradient_magnitude_norm = gradient_magnitude / 255.0

    # --- Contrast ---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L_channel = lab[:, :, 0].astype(float)
    rms_contrast = L_channel.std() / 255.0

    return {
        # Raw counts
        'sam_n_segments':          n_segments,
        'sam_log_n_segments':      log_n_segments,
        # Size features
        'sam_dominant_frac':       dominant_frac,
        'sam_mean_segment_frac':   mean_segment_frac,
        'sam_std_segment_frac':    std_segment_frac,
        'sam_size_entropy':        size_entropy,
        # Spatial
        'sam_coverage_frac':       coverage_frac,
        'sam_center_bias':         center_bias,
        # Boundary / texture
        'sam_edge_density':        edge_density,
        'sam_gradient_magnitude':  gradient_magnitude_norm,
        'sam_rms_contrast':        rms_contrast,
        # Shape
        'sam_mean_compactness':    mean_compactness,
        'sam_max_compactness':     max_compactness,
        # Symmetry
        'sam_mean_symmetry':       mean_symmetry,
    }

# ── Step 3: Run SAM over all images ─────────────────────────

def run_extraction():
    records = []
    dirs = [
        (URBAN_DIR,     'urban'),
        (LANDSCAPE_DIR, 'landscape'),
    ]
    for folder, itype in dirs:
        files = sorted(f for f in os.listdir(folder) if f.endswith('.jpg'))
        print(f"\nProcessing {len(files)} {itype} images...")
        for i, fname in enumerate(files):
            img_id = fname.replace('.jpg', '')
            path   = os.path.join(folder, fname)
            feats  = extract_sam_features(path)
            if feats is not None:
                feats['image_name'] = fname
                feats['image_id']   = int(img_id)
                feats['image_type'] = itype
                records.append(feats)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} done")

    sam_df = pd.DataFrame(records)
    sam_df.to_csv(SAM_FEATURES_CACHE, index=False)
    print(f"\nSAM features saved → {SAM_FEATURES_CACHE}")
    return sam_df

# Run (or load from cache if already computed)
if os.path.exists(SAM_FEATURES_CACHE):
    print("Loading cached SAM features...")
    sam_df = pd.read_csv(SAM_FEATURES_CACHE)
else:
    sam_df = run_extraction()

print(f"SAM features shape: {sam_df.shape}")
sam_df.head()

# ── Step 4: Merge with Claude scores & human ratings ────────

claude_df = pd.read_csv(CLAUDE_CSV)
human_df  = pd.read_csv(VOTES_TSV, sep='\t')
claude_df['image_id'] = claude_df['image_name'].str.replace('.jpg','',regex=False).astype(int)
human_df.rename(columns={'ID': 'image_id', 'Average': 'human_score'}, inplace=True)

df = (claude_df
      .merge(human_df[['image_id','human_score']], on='image_id')
      .merge(sam_df.drop(columns=['image_name','image_type'], errors='ignore'), on='image_id'))

print(f"Merged dataset: {df.shape}")

claude_props = [
    'levels_of_scale', 'strong_centers', 'boundaries', 'alternating_repetition',
    'positive_space', 'good_shape', 'local_symmetries', 'deep_interlock_and_ambiguity',
    'contrast', 'gradients', 'roughness', 'echoes', 'the_void',
    'simplicity_and_inner_calm', 'not_separateness'
]
pretty_claude = [
    'Levels of Scale', 'Strong Centers', 'Boundaries', 'Alternating Repetition',
    'Positive Space', 'Good Shape', 'Local Symmetries', 'Deep Interlock & Ambiguity',
    'Contrast', 'Gradients', 'Roughness', 'Echoes', 'The Void',
    'Simplicity & Inner Calm', 'Not-Separateness'
]

sam_feats = [c for c in sam_df.columns if c.startswith('sam_')]
sam_pretty = [f.replace('sam_','').replace('_',' ').title() for f in sam_feats]

all_feats  = claude_props + sam_feats
all_pretty = pretty_claude + sam_pretty

# ── Step 5: Importance analysis helper ──────────────────────

def norm(arr):
    arr = np.abs(np.array(arr, dtype=float))
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / (rng + 1e-9)

def compute_importance(sub_df, features, pretty_names):
    X = sub_df[features].values
    y = sub_df['human_score'].values
    X_sc = StandardScaler().fit_transform(X)

    pearson_r  = [stats.pearsonr(sub_df[f], y)[0] for f in features]
    lasso      = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X_sc, y)
    ridge      = RidgeCV(cv=5).fit(X_sc, y)
    rf         = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_sc, y)
    perm       = permutation_importance(rf, X_sc, y, n_repeats=20, random_state=42, n_jobs=-1)
    gb         = GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X_sc, y)

    res = pd.DataFrame({
        'Feature':          features,
        'Property':         pretty_names,
        'Source':           ['Claude']*len(claude_props) + ['SAM']*len(sam_feats),
        'Pearson_r':        pearson_r,
        'Lasso':            norm(lasso.coef_),
        'Ridge':            norm(ridge.coef_),
        'RandomForest':     norm(rf.feature_importances_),
        'Permutation':      norm(perm.importances_mean),
        'GradientBoosting': norm(gb.feature_importances_),
    })
    res['Composite'] = res[['Lasso','Ridge','RandomForest','Permutation','GradientBoosting']].mean(axis=1)
    return res.sort_values('Composite', ascending=False).reset_index(drop=True)

# ── Step 6: Run importance for each subset ───────────────────

results = {}
for subset in ['all', 'urban', 'landscape']:
    sub = df if subset == 'all' else df[df['image_type'] == subset]
    print(f"Computing importance: {subset} ({len(sub)} images)...")
    results[subset] = compute_importance(sub, all_feats, all_pretty)

# ── Step 7: Print top 5 & top 10 ────────────────────────────

for top_n in [5, 10]:
    print(f"\n{'='*70}")
    print(f"  TOP {top_n} FEATURES (Claude + SAM) BY IMAGE TYPE")
    print(f"{'='*70}")
    for subset in ['urban', 'landscape', 'all']:
        print(f"\n  [{subset.upper()}]")
        for i, row in results[subset].head(top_n).iterrows():
            tag = f"[{row['Source']:6s}]"
            print(f"  {i+1:2d}. {tag} {row['Property']:35s}  "
                  f"composite={row['Composite']:.3f}  pearson_r={row['Pearson_r']:+.3f}")

# ── Step 8: Baseline vs SAM-augmented R² comparison ─────────

from sklearn.model_selection import cross_val_score

def r2_cv(X, y):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    return scores.mean(), scores.std()

print("\n=== Predictive Power: Baseline (Claude only) vs Claude + SAM ===\n")
for subset in ['all', 'urban', 'landscape']:
    sub = df if subset == 'all' else df[df['image_type'] == subset]
    y   = sub['human_score'].values
    X_claude = StandardScaler().fit_transform(sub[claude_props].values)
    X_both   = StandardScaler().fit_transform(sub[all_feats].values)
    r2_base_m, r2_base_s = r2_cv(X_claude, y)
    r2_aug_m,  r2_aug_s  = r2_cv(X_both, y)
    gain = r2_aug_m - r2_base_m
    print(f"  {subset:10s}  Claude only: R²={r2_base_m:.3f}±{r2_base_s:.3f}  "
          f"| Claude+SAM: R²={r2_aug_m:.3f}±{r2_aug_s:.3f}  "
          f"| Gain: {gain:+.3f}")

# ── Step 9: Visualise top 10 — Claude vs SAM coloured ───────

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
COLOR_CLAUDE = '#3498DB'
COLOR_SAM    = '#E67E22'
subset_titles = {'urban': 'Urban', 'landscape': 'Landscape', 'all': 'All Images'}

for ax, subset in zip(axes, ['urban', 'landscape', 'all']):
    top10  = results[subset].head(10)
    colors = [COLOR_SAM if s == 'SAM' else COLOR_CLAUDE for s in top10['Source'][::-1]]

    ax.barh(top10['Property'][::-1], top10['Composite'][::-1],
            color=colors, edgecolor='white', linewidth=0.8, alpha=0.92)

    for j, (_, row) in enumerate(top10[::-1].iterrows()):
        ax.text(row['Composite'] + 0.005, j,
                f"r={row['Pearson_r']:+.2f}",
                va='center', fontsize=8, color='#333333')

    ax.set_xlabel('Composite Importance (normalised)', fontsize=11)
    ax.set_title(f"{subset_titles[subset]}\nTop 10 Features", fontsize=13, fontweight='bold')
    ax.set_xlim(0, top10['Composite'].max() * 1.3)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

patch_c = mpatches.Patch(color=COLOR_CLAUDE, label='Claude Score')
patch_s = mpatches.Patch(color=COLOR_SAM,    label='SAM Feature')
fig.legend(handles=[patch_c, patch_s], fontsize=12, loc='lower center',
           ncol=2, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

plt.suptitle('Feature Importance: Claude Scores vs SAM Structural Features\n'
             '(colour = feature source)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Capstone/feature_importance_sam_vs_claude.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── Step 10: SAM features correlation with Claude properties ─

print("\n=== SAM Feature ↔ Claude Property Correlation Matrix ===\n")
corr_cross = df[sam_feats + claude_props].corr().loc[sam_feats, claude_props]
corr_cross.index = sam_pretty
corr_cross.columns = pretty_claude

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(corr_cross.values, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(pretty_claude)))
ax.set_yticks(range(len(sam_pretty)))
ax.set_xticklabels(pretty_claude, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(sam_pretty, fontsize=9)
ax.set_title('SAM Features vs Claude Property Correlations\n'
             '(shows how objective structure maps to subjective scores)',
             fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Pearson r')
for i in range(len(sam_pretty)):
    for j in range(len(pretty_claude)):
        v = corr_cross.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=6.5, color='white' if abs(v) > 0.6 else 'black')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Capstone/sam_claude_correlation.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nAll done. Outputs saved to Capstone folder on Drive.")
