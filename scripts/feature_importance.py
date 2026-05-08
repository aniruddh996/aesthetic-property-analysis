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

X = df[properties].values
y = df['human_score'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 1. Pearson correlation ────────────────────────────────────────────────────
pearson_r = [stats.pearsonr(df[p], y)[0] for p in properties]

# ── 2. Lasso (L1) coefficients ───────────────────────────────────────────────
lasso = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X_scaled, y)
lasso_coef = np.abs(lasso.coef_)

# ── 3. Ridge coefficients ────────────────────────────────────────────────────
ridge = RidgeCV(cv=5).fit(X_scaled, y)
ridge_coef = np.abs(ridge.coef_)

# ── 4. Random Forest importance ──────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)
rf_imp = rf.feature_importances_

# ── 5. Permutation importance (model-agnostic) ───────────────────────────────
perm = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=42, n_jobs=-1)
perm_imp = perm.importances_mean

# ── 6. Gradient Boosting importance ─────────────────────────────────────────
gb = GradientBoostingRegressor(n_estimators=300, random_state=42)
gb.fit(X_scaled, y)
gb_imp = gb.feature_importances_

# ── Normalise all to [0, 1] for fair comparison ──────────────────────────────
def norm(arr):
    arr = np.abs(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

results = pd.DataFrame({
    'Property':         pretty,
    'Pearson_r':        pearson_r,
    'Lasso':            norm(lasso_coef),
    'Ridge':            norm(ridge_coef),
    'RandomForest':     norm(rf_imp),
    'Permutation':      norm(perm_imp),
    'GradientBoosting': norm(gb_imp),
})
results['Composite'] = results[['Lasso','Ridge','RandomForest','Permutation','GradientBoosting']].mean(axis=1)
results = results.sort_values('Composite', ascending=False).reset_index(drop=True)

print("\n=== Feature Importance Ranking (by Composite score) ===\n")
print(results[['Property','Pearson_r','Composite']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: composite ranking
ax = axes[0]
colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(results)))
bars = ax.barh(results['Property'][::-1], results['Composite'][::-1],
               color=colors[::-1], edgecolor='white', alpha=0.9)
ax.set_xlabel('Composite Importance (normalised)', fontsize=12)
ax.set_title('Overall Feature Importance\n(avg of Lasso, Ridge, RF, Permutation, GBM)', fontsize=12, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# Right: all methods side by side
ax2 = axes[1]
methods = ['Pearson_r', 'Lasso', 'Ridge', 'RandomForest', 'Permutation', 'GradientBoosting']
method_labels = ['Pearson r', 'Lasso', 'Ridge', 'Rand Forest', 'Permutation', 'Grad Boost']
x = np.arange(len(results))
width = 0.13
cmap = plt.cm.tab10.colors

for i, (m, ml) in enumerate(zip(methods, method_labels)):
    vals = norm(results[m].values) if m == 'Pearson_r' else results[m].values
    ax2.bar(x + i * width, vals, width, label=ml, color=cmap[i], alpha=0.85, edgecolor='white')

ax2.set_xticks(x + width * (len(methods) - 1) / 2)
ax2.set_xticklabels(results['Property'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Normalised Importance', fontsize=11)
ax2.set_title('Importance by Method', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_axisbelow(True)

plt.suptitle('Which Property Drives Human Attractiveness Ratings?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 most influential properties:")
for i, row in results.head(5).iterrows():
    print(f"  {i+1}. {row['Property']:30s}  composite={row['Composite']:.3f}  pearson_r={row['Pearson_r']:+.3f}")
