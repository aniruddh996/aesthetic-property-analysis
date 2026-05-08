import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ── Load & merge ─────────────────────────────────────────────────────────────
claude = pd.read_csv('all_images_15_properties.csv')
human  = pd.read_csv('votes.tsv', sep='\t')
claude['ID'] = claude['image_name'].str.replace('.jpg', '', regex=False).astype(int)
df = claude.merge(human[['ID', 'Average']], on='ID')
df.rename(columns={'Average': 'human_score'}, inplace=True)

properties = [
    'levels_of_scale', 'strong_centers', 'boundaries', 'alternating_repetition',
    'positive_space', 'good_shape', 'local_symmetries', 'deep_interlock_and_ambiguity',
    'contrast', 'gradients', 'roughness', 'echoes', 'the_void',
    'simplicity_and_inner_calm', 'not_separateness'
]
pretty = [
    'Levels of Scale', 'Strong Centers', 'Boundaries', 'Alternating Repetition',
    'Positive Space', 'Good Shape', 'Local Symmetries', 'Deep Interlock & Ambiguity',
    'Contrast', 'Gradients', 'Roughness', 'Echoes', 'The Void',
    'Simplicity & Inner Calm', 'Not-Separateness'
]

def norm(arr):
    arr = np.abs(arr)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / (rng + 1e-9)

def compute_importance(sub_df):
    X = sub_df[properties].values
    y = sub_df['human_score'].values
    X_scaled = StandardScaler().fit_transform(X)

    pearson_r  = [stats.pearsonr(sub_df[p], y)[0] for p in properties]
    lasso      = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X_scaled, y)
    ridge      = RidgeCV(cv=5).fit(X_scaled, y)
    rf         = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_scaled, y)
    perm       = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=42, n_jobs=-1)
    gb         = GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X_scaled, y)

    res = pd.DataFrame({
        'Property':         pretty,
        'Pearson_r':        pearson_r,
        'Lasso':            norm(lasso.coef_),
        'Ridge':            norm(ridge.coef_),
        'RandomForest':     norm(rf.feature_importances_),
        'Permutation':      norm(perm.importances_mean),
        'GradientBoosting': norm(gb.feature_importances_),
    })
    res['Composite'] = res[['Lasso','Ridge','RandomForest','Permutation','GradientBoosting']].mean(axis=1)
    return res.sort_values('Composite', ascending=False).reset_index(drop=True)

# ── Run for each subset ───────────────────────────────────────────────────────
results = {}
for subset in ['urban', 'landscape', 'all']:
    sub = df if subset == 'all' else df[df['image_type'] == subset]
    print(f"Computing importance for: {subset} ({len(sub)} images)...")
    results[subset] = compute_importance(sub)

# ── Print tables ──────────────────────────────────────────────────────────────
for top_n in [5, 10]:
    print(f"\n{'='*65}")
    print(f"  TOP {top_n} PROPERTIES BY IMAGE TYPE")
    print(f"{'='*65}")
    for subset in ['urban', 'landscape', 'all']:
        print(f"\n  [{subset.upper()}]")
        for i, row in results[subset].head(top_n).iterrows():
            print(f"  {i+1:2d}. {row['Property']:30s}  composite={row['Composite']:.3f}  pearson_r={row['Pearson_r']:+.3f}")

# ── Plot: Top 10 side by side for urban vs landscape ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)

colors_map = {'urban': '#E74C3C', 'landscape': '#2980B9', 'all': '#7F8C8D'}
titles_map  = {'urban': 'Urban', 'landscape': 'Landscape', 'all': 'All Images'}

for ax, subset in zip(axes, ['urban', 'landscape', 'all']):
    top10 = results[subset].head(10)
    cmap_vals = plt.cm.RdYlGn(np.linspace(0.85, 0.35, 10))

    bars = ax.barh(top10['Property'][::-1], top10['Composite'][::-1],
                   color=cmap_vals[::-1], edgecolor='white', linewidth=0.8, alpha=0.92)

    # Annotate pearson r
    for j, (_, row) in enumerate(top10[::-1].iterrows()):
        ax.text(row['Composite'] + 0.005, j,
                f"r={row['Pearson_r']:+.2f}",
                va='center', fontsize=8.5, color='#333333')

    ax.set_xlabel('Composite Importance (normalised)', fontsize=11)
    ax.set_title(f"{titles_map[subset]}\nTop 10 Properties", fontsize=13, fontweight='bold')
    ax.set_xlim(0, top10['Composite'].max() * 1.25)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=9)

plt.suptitle('Which Properties Drive Human Attractiveness?\n(Composite of Lasso, Ridge, Random Forest, Permutation, Gradient Boosting)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_importance_by_type.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Plot: Top 5 head-to-head heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

all_top5 = pd.concat([
    results[s].head(5)[['Property','Composite']].assign(Type=s)
    for s in ['urban', 'landscape', 'all']
])
pivot = all_top5.pivot_table(index='Property', columns='Type', values='Composite', fill_value=0)
pivot = pivot.reindex(columns=['urban', 'landscape', 'all'])
pivot = pivot.sort_values('all', ascending=False)

im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(3))
ax.set_xticklabels(['Urban', 'Landscape', 'All'], fontsize=12)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=11)
ax.set_title('Top-5 Properties Heatmap — Composite Importance by Type', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalised Importance')
for i in range(len(pivot)):
    for j in range(3):
        val = pivot.values[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=10, color='white' if val > 0.4 else 'black', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_top5_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved: feature_importance_by_type.png  |  feature_importance_top5_heatmap.png")
