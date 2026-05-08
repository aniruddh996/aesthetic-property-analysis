# decoding-urban-aesthetics

> **Beyond Black-Box Assessment: Identifying Visual Properties that Predict Aesthetic Attraction in Urban Architecture**  
> Aniruddh Rajagopal · Suraj Ravindra Kapare  
> Department of Data Science, The George Washington University

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [NIMA Baseline](#1-nima-baseline-prior-work)
  - [Claude API Scoring](#2-claude-api-scoring-alexanders-15-properties)
  - [SAM Feature Extraction](#3-sam-structural-feature-extraction)
  - [Feature Importance Analysis](#4-feature-importance-analysis)
- [Results](#results)
  - [Claude vs Human Agreement](#claude-vs-human-agreement)
  - [Top Predictors by Image Type](#top-predictors-by-image-type)
  - [Does SAM Help?](#does-sam-improve-prediction)
- [Alexander's 15 Properties](#christoper-alexanders-15-properties-of-wholeness)
- [Installation & Usage](#installation--usage)
- [Notebooks](#notebooks)
- [References](#references)

---

## Overview

Understanding what makes a place visually attractive has significant implications for **urban planning, architectural design, landscape evaluation, and public wellbeing**. Yet aesthetic quality remains difficult to define objectively — making it challenging to establish evidence-based design guidelines.

This project moves beyond black-box aesthetic scoring (like NIMA) toward an **interpretable, theory-driven framework** based on Christopher Alexander's *15 Fundamental Properties of Wholeness*. By combining:

- **Claude API** for structured visual property scoring
- **Crowd-sourced human ratings** (212,000+ votes) as ground truth
- **Meta's Segment Anything Model (SAM)** for objective structural features

...we identify *which specific visual properties* drive human attractiveness ratings across urban and landscape images — shifting the question from *"is this place beautiful?"* to *"which visual structures make it so?"*

---

## Research Question

> *Which of Christopher Alexander's 15 fundamental properties of wholeness best predict human aesthetic ratings of urban and landscape images?*

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Strongest overall predictor** | Gradients (r = 0.67) — smooth transitions and layered visual change |
| **Top urban driver** | Simplicity & Inner Calm (r = 0.591) — clarity and reduced clutter |
| **Top landscape driver** | Good Shape (r = 0.692) — coherent, well-formed visual structures |
| **Universal predictors** | Positive Space & Gradients appear in top-5 for both image types |
| **Claude vs Human correlation** | r = 0.609 (urban), r = 0.687 (landscape) |
| **Claude bias** | Claude scores ~1.9 pts higher than humans on average |
| **SAM contribution** | Modest overall (+0.019 R²), moderate for landscape (+0.058 R²) |
| **SAM in top-5** | Edge Density only enters landscape top-5 (r = −0.108) |

---

## Repository Structure

```
decoding-urban-aesthetics/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
│
├── data/
│   ├── README.md                      ← Data sources and download instructions
│   ├── all_images_15_properties.csv   ← Claude API scores (800 images × 15 properties)
│   ├── sam_features.csv               ← SAM structural features (800 images × 14 features)
│   └── votes.tsv                      ← Scenic-or-Not human ratings (212,000+ votes)
│
├── notebooks/
│   ├── 01_claude_api_scoring.ipynb    ← Claude API scoring pipeline
│   ├── 02_sam_feature_extraction.ipynb← SAM segmentation & feature extraction
│   ├── 03_eda_visualizations.ipynb    ← Exploratory data analysis & all charts
│   ├── 04_human_vs_claude.ipynb       ← Human vs Claude comparison analysis
│   └── 05_feature_importance.ipynb    ← Feature importance by image type
│
├── scripts/
│   ├── claude_scorer.py               ← Claude API scoring script
│   ├── sam_pipeline.py                ← SAM feature extraction pipeline
│   └── feature_importance_by_type.py  ← Feature importance analysis script
│
├── results/
│   ├── figures/                       ← All generated charts and plots
│   │   ├── score_distributions.png
│   │   ├── scatter_claude_vs_human.png
│   │   ├── feature_importance_by_type.png
│   │   ├── sam_claude_correlation_heatmap.png
│   │   └── r2_comparison.png
│   └── tables/                        ← Summary statistics and result tables
│
└── paper/
    └── Capstone_Final_Paper.pdf       ← Full research paper
```

---

## Dataset

### Images
- **Source:** [Geograph UK](https://www.geograph.org.uk) — geographically referenced photographs of built and natural environments
- **Size:** 800 images total
  - 400 Urban scenes
  - 400 Landscape scenes
- **Download:** Images were fetched programmatically via the Geograph API (see `notebooks/01_claude_api_scoring.ipynb`)

### Human Ratings
- **Source:** [Scenic-or-Not](https://scenicornot.datasciencelab.co.uk)
- **Scale:** 1–10 scenic beauty rating
- **Volume:** 212,000+ crowd-sourced votes
- **Format:** `votes.tsv` — each row contains image ID, average rating, variance, and raw votes

### Generated Features
| File | Description | Dimensions |
|------|-------------|------------|
| `all_images_15_properties.csv` | Claude API scores for all 15 Alexander properties | 800 × 17 |
| `sam_features.csv` | SAM-derived structural features | 800 × 16 |

---

## Methodology

### 1. NIMA Baseline (Prior Work)

Before settling on the Alexander-property framework, we tested **NIMA (Neural Image Assessment)** — a CNN-based aesthetic scorer by Google, trained on the AVA dataset (250,000+ human-rated images).

**Why NIMA failed:**

| Metric | NIMA | Human Ratings |
|--------|------|---------------|
| Std Dev | 0.246 | 1.598 |
| Score Range | 4.6 – 6.0 | 1.0 – 10.0 |
| Interpretability | ❌ Black box | ✅ Explicit votes |

NIMA compressed all judgments into a narrow 1.4-point window and offered **no explanation** of which visual properties drove the scores — making it useless for design-relevant insight. This motivated the shift to an interpretable, property-based approach.

---

### 2. Claude API Scoring — Alexander's 15 Properties

Each of the 800 images was evaluated using the **Claude API** (`claude-sonnet-4-20250514`) against Christopher Alexander's 15 Fundamental Properties of Wholeness. Claude assigned a score from **1–10** for each property per image.

```
Total ratings generated: 800 images × 15 properties = 12,000 ratings
```

**Prompt structure:** Each image was passed with a structured prompt asking Claude to rate the presence and strength of each Alexander property on a 1–10 scale with reasoning.

See [`scripts/claude_scorer.py`](scripts/claude_scorer.py) for the full implementation.

---

### 3. SAM Structural Feature Extraction

Meta's **Segment Anything Model (SAM)** — specifically the `vit_h` checkpoint — was used to extract 14 objective structural features per image:

| SAM Feature | What It Measures | Maps To |
|-------------|-----------------|---------|
| `n_segments` | Number of distinct regions | Levels of Scale |
| `dominant_frac` | Largest segment as % of image | Positive Space / Strong Centers |
| `coverage_frac` | % of image covered by segments | The Void |
| `edge_density` | Fraction of edge pixels | Boundaries & Contrast |
| `mean_compactness` | Average shape regularity (0–1) | Good Shape |
| `mean_symmetry` | Flip-overlap symmetry score | Local Symmetries |
| `gradient_magnitude` | Average colour gradient strength | Gradients |
| `center_bias` | How centered the segments are | Strong Centers |
| `size_entropy` | Diversity of segment sizes | Alternating Repetition |
| `log_n_segments` | Log-scaled segment count | Complexity measure |
| `std_segment_frac` | Variation in segment sizes | Scale distribution |
| `mean_segment_frac` | Average segment size | Density measure |
| `rms_contrast` | RMS contrast of L channel | Contrast |
| `max_compactness` | Most circle-like segment | Shape quality |

SAM was run on GPU (recommended). Features are cached in `data/sam_features.csv` so you don't need to re-run.

See [`scripts/sam_pipeline.py`](scripts/sam_pipeline.py) for the full implementation.

---

### 4. Feature Importance Analysis

Feature importance was evaluated using a **5-method ensemble** to ensure robustness:

| Method | Why Included |
|--------|-------------|
| **Lasso (L1 regression)** | Shrinks irrelevant features to exactly zero — acts as feature selector |
| **Ridge (L2 regression)** | Keeps all features but penalises less important ones — stable ranking |
| **Random Forest** | Captures non-linear relationships and feature interactions naturally |
| **Permutation Importance** | Model-agnostic — shuffles each feature and measures accuracy drop |
| **Gradient Boosting** | Sequential tree ensemble — strong at finding interaction effects |

Each method's importance scores were normalised to [0,1] and averaged into a **composite importance score**. Analysis was run separately for:
- All 800 images combined
- Urban images only (n=400)
- Landscape images only (n=400)

**Why Random Forest as the primary model?**  
Aesthetic attractiveness is non-linear, involves feature interactions (e.g. Positive Space + Simplicity together), and works well with datasets of ~800 samples. Random Forest handles all of this naturally and provides feature importance as a direct output. See the paper for full justification.

---

## Results

### Claude vs Human Agreement

| Subset | Pearson r | p-value |
|--------|-----------|---------|
| All images | 0.656 | < 0.001 |
| Urban | 0.609 | < 0.001 |
| Landscape | 0.687 | < 0.001 |

**Score distributions:**

| Metric | Human | Claude |
|--------|-------|--------|
| Overall mean | 4.72 | 6.62 |
| Urban mean | 3.80 | 6.38 |
| Landscape mean | 5.64 | 6.86 |
| Std deviation | 1.78 | 1.21 |

Claude systematically scores ~1.9 points higher than humans (positive bias) and has a narrower distribution — but the relative *ranking* of images is well preserved, which is what the correlations measure.

---

### Top Predictors by Image Type

**Urban — Top 5:**

| Rank | Property | Pearson r | Source |
|------|----------|-----------|--------|
| 1 | Simplicity & Inner Calm | +0.591 | Claude |
| 2 | Positive Space | +0.618 | Claude |
| 3 | Not-Separateness | +0.605 | Claude |
| 4 | Levels of Scale | +0.584 | Claude |
| 5 | Gradients | +0.529 | Claude |

**Landscape — Top 5:**

| Rank | Property | Pearson r | Source |
|------|----------|-----------|--------|
| 1 | Positive Space | +0.620 | Claude |
| 2 | Good Shape | +0.692 | Claude |
| 3 | Levels of Scale | +0.652 | Claude |
| 4 | Edge Density | −0.108 | **SAM** |
| 5 | Simplicity & Inner Calm | +0.600 | Claude |

> **Key insight:** SAM only contributes one feature (Edge Density) and only for landscape images. Claude's interpretive property scores dominate prediction for both image types — especially urban, where all top-5 are Claude scores.

---

### Does SAM Improve Prediction?

Measured using **5-fold cross-validated R²** with Random Forest:

| Subset | Claude Only | Claude + SAM | Gain | Verdict |
|--------|-------------|--------------|------|---------|
| All Images | 0.468 | 0.487 | +0.019 | Marginal |
| Urban | 0.394 | 0.427 | +0.033 | Small |
| Landscape | 0.518 | 0.576 | +0.058 | Moderate ✓ |

SAM helps most for landscape scenes — natural environments have more objective physical structure (open sky, clear foreground/background) that SAM can genuinely measure. For urban scenes, human aesthetic judgment is more subjective and SAM adds little.

---

## Christopher Alexander's 15 Properties of Wholeness

From *The Nature of Order* (Alexander, 2002) — properties that recur in environments perceived as coherent, beautiful, and "alive":

| # | Property | Description |
|---|----------|-------------|
| 1 | **Levels of Scale** | Multiple nested sizes — detail at every zoom level |
| 2 | **Strong Centers** | Focal elements that organize the visual field |
| 3 | **Boundaries** | Clear edges that define and separate regions |
| 4 | **Alternating Repetition** | Patterns that repeat without becoming monotonous |
| 5 | **Positive Space** | Well-shaped, intentional spatial regions |
| 6 | **Good Shape** | Satisfying, well-formed, recognizable shapes |
| 7 | **Local Symmetries** | Symmetry at small scales, not necessarily the whole |
| 8 | **Deep Interlock & Ambiguity** | Elements that interweave and interpenetrate |
| 9 | **Contrast** | Clear differentiation between different elements |
| 10 | **Gradients** | Smooth transitions in colour, texture, light, or form |
| 11 | **Roughness** | Slight irregularity and variation, not mechanical perfection |
| 12 | **Echoes** | Similar angles, shapes, and patterns recurring throughout |
| 13 | **The Void** | Areas of calm emptiness that give the eye somewhere to rest |
| 14 | **Simplicity & Inner Calm** | Quiet, restful quality — not visually noisy |
| 15 | **Not-Separateness** | Elements belong to their surroundings, not isolated |

---

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pandas
numpy
matplotlib
plotly
scikit-learn
scipy
opencv-python-headless
torch
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
anthropic
jupyter
```

### Running the Claude Scoring Pipeline

```python
# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Run scoring
python scripts/claude_scorer.py \
  --urban_dir /path/to/urban_400 \
  --landscape_dir /path/to/landscape_400 \
  --output data/all_images_15_properties.csv
```

### Running SAM Feature Extraction

> ⚠️ GPU strongly recommended. Download the SAM checkpoint first:
> ```bash
> wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
> ```

```python
python scripts/sam_pipeline.py \
  --urban_dir /path/to/urban_400 \
  --landscape_dir /path/to/landscape_400 \
  --checkpoint sam_vit_h_4b8939.pth \
  --output data/sam_features.csv
```

### Running Feature Importance Analysis

```python
python scripts/feature_importance_by_type.py \
  --claude_csv data/all_images_15_properties.csv \
  --votes_tsv data/votes.tsv \
  --sam_csv data/sam_features.csv \
  --top_n 10
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_claude_api_scoring.ipynb` | Full Claude API pipeline — mounts Drive, downloads images, scores all 15 properties, saves CSV |
| `02_sam_feature_extraction.ipynb` | SAM installation, feature extraction across all 800 images, caches to Drive |
| `03_eda_visualizations.ipynb` | Score distributions, correlation heatmaps, violin plots, radar charts, all 15 property histograms |
| `04_human_vs_claude.ipynb` | Scatter plots, Bland-Altman agreement, rank concordance, per-property correlations |
| `05_feature_importance.ipynb` | 5-method ensemble importance, top-5/top-10 rankings, R² comparison, SAM vs Claude chart |

All notebooks are designed for **Google Colab** with GPU runtime.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{rajagopal2025aesthetics,
  title   = {Beyond Black-Box Assessment: Identifying Visual Properties 
             that Predict Aesthetic Attraction in Urban Architecture},
  author  = {Rajagopal, Aniruddh and Kapare, Suraj Ravindra},
  year    = {2025},
  school  = {The George Washington University},
  note    = {Department of Data Science Capstone Project}
}
```

---

## References

- Alexander, C. (2002). *The Nature of Order*. Center for Environmental Structure.
- Talebi, H., & Milanfar, P. (2018). NIMA: Neural image assessment. *IEEE Transactions on Image Processing*, 27(8), 3998–4011.
- Kirillov, A., et al. (2023). Segment Anything. *ICCV 2023*.
- Salesses, P., Schechtner, K., & Hidalgo, C. A. (2013). The collaborative image of the city. *PLOS ONE*, 8(7).
- Seresinhe, C. I., Preis, T., & Moat, H. S. (2017). Using deep learning to quantify the beauty of outdoor places. *Royal Society Open Science*, 4(7).
- Murray, N., Marchesotti, L., & Perronnin, F. (2012). AVA: A large-scale database for aesthetic visual analysis. *CVPR 2012*.
- Dubey, A., et al. (2016). Deep learning the city. *ECCV 2016*.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*George Washington University · Department of Data Science · 2025*
