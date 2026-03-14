"""
FICO Score Quantization — Optimal Bucket Construction
JPMorgan Chase – Quantitative Research Virtual Experience Task 4
----------------------------------------------------------------
Two methods implemented:
  1. MSE minimization     – minimizes squared error within buckets
  2. Log-likelihood (DP)  – maximizes default-discriminative log-likelihood
       via dynamic programming
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA  (reuse Task 3 CSV via the previous model's data path)
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv('/mnt/user-data/uploads/Task_3_and_4_Loan_Data.csv')
fico   = df['fico_score'].values.astype(float)
default = df['default'].values.astype(int)

print("═"*60)
print("  FICO SCORE OVERVIEW")
print("═"*60)
print(f"  Records     : {len(fico):,}")
print(f"  FICO range  : {fico.min():.0f} – {fico.max():.0f}")
print(f"  Default rate: {default.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  2.  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def bucket_stats(fico, default, boundaries):
    """
    Given sorted boundary list [b0, b1, ..., bk] (inclusive min / exclusive max
    except last), return per-bucket (n, k_defaults, pd) arrays.
    boundaries must start at or below min(fico) and end above max(fico).
    """
    n_buckets = len(boundaries) - 1
    ns, ks, pds = [], [], []
    for i in range(n_buckets):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask  = (fico >= lo) & (fico < hi) if i < n_buckets - 1 else (fico >= lo) & (fico <= hi)
        n_i   = mask.sum()
        k_i   = default[mask].sum()
        p_i   = k_i / n_i if n_i > 0 else 0.0
        ns.append(n_i); ks.append(k_i); pds.append(p_i)
    return np.array(ns), np.array(ks), np.array(pds)


def log_likelihood(ns, ks, pds, eps=1e-10):
    """
    LL = Σ  [k_i * log(p_i) + (n_i - k_i) * log(1 - p_i)]
    """
    ll = 0.0
    for n, k, p in zip(ns, ks, pds):
        p = np.clip(p, eps, 1 - eps)
        ll += k * np.log(p) + (n - k) * np.log(1 - p)
    return ll


def mse_loss(fico, default, boundaries):
    """MSE: map every score to its bucket's mean FICO, minimise squared error."""
    n_buckets = len(boundaries) - 1
    total_mse = 0.0
    for i in range(n_buckets):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask   = (fico >= lo) & (fico < hi) if i < n_buckets - 1 else (fico >= lo) & (fico <= hi)
        if mask.sum() > 0:
            mu         = fico[mask].mean()
            total_mse += ((fico[mask] - mu) ** 2).sum()
    return total_mse / len(fico)


def rating_label(bucket_idx, n_buckets):
    """Lower rating = better credit (bucket 1 = best FICO scores)."""
    return bucket_idx + 1   # 1 = best (highest FICO), n = worst


# ══════════════════════════════════════════════════════════════════════════════
#  3.  METHOD A — MSE MINIMIZATION  (quantile / equal-frequency binning
#      extended with local boundary search to minimize MSE)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_mse(fico, default, n_buckets, n_restarts=3):
    """
    Find bucket boundaries that minimise MSE using quantile initialisation
    plus greedy boundary search.
    """
    fico_sorted   = np.sort(np.unique(fico.astype(int)))
    candidate_bds = fico_sorted.tolist()      # every unique score is a candidate boundary

    # Initialise from quantiles
    quantiles = np.linspace(0, 100, n_buckets + 1)
    init_inner = np.unique(np.percentile(fico, quantiles[1:-1]).astype(int)).tolist()
    best_inner = init_inner[: n_buckets - 1]

    lo = int(fico.min()) - 1
    hi = int(fico.max()) + 1

    def boundaries_from_inner(inner):
        return sorted([lo] + inner + [hi])

    best_mse  = mse_loss(fico, default, boundaries_from_inner(best_inner))

    # Greedy pass: try shifting each boundary one unique-score step at a time
    improved = True
    while improved:
        improved = False
        for idx in range(len(best_inner)):
            current = best_inner[idx]
            pos     = candidate_bds.index(current) if current in candidate_bds else 0
            for delta in range(-10, 11):
                new_pos = pos + delta
                if new_pos < 1 or new_pos >= len(candidate_bds) - 1:
                    continue
                new_val   = candidate_bds[new_pos]
                trial     = best_inner.copy()
                trial[idx] = new_val
                trial_sorted = sorted(trial)
                if len(set(trial_sorted)) < len(trial_sorted):
                    continue   # boundaries would collide
                mse = mse_loss(fico, default, boundaries_from_inner(trial_sorted))
                if mse < best_mse - 1e-6:
                    best_mse   = mse
                    best_inner = trial_sorted
                    improved   = True

    return boundaries_from_inner(best_inner), best_mse


# ══════════════════════════════════════════════════════════════════════════════
#  4.  METHOD B — LOG-LIKELIHOOD  (dynamic programming)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_log_likelihood_dp(fico, default, n_buckets):
    """
    Dynamic programming to maximise log-likelihood across n_buckets.

    State:  dp[i][j] = best LL achievable using j buckets over FICO range
                        [score_min, unique_scores[i]]
    Transition: try all possible previous boundary positions.
    """
    scores = np.sort(np.unique(fico.astype(int)))
    N      = len(scores)          # number of unique FICO values
    eps    = 1e-10

    # Pre-compute cumulative n and k indexed by position in `scores`
    n_total = np.zeros(N, dtype=int)
    k_total = np.zeros(N, dtype=int)
    for idx, s in enumerate(scores):
        mask          = fico == s
        n_total[idx]  = mask.sum()
        k_total[idx]  = default[mask].sum()

    cum_n = np.concatenate([[0], np.cumsum(n_total)])
    cum_k = np.concatenate([[0], np.cumsum(k_total)])

    def ll_segment(i_start, i_end):
        """LL for scores[i_start : i_end+1] (inclusive)."""
        n = cum_n[i_end + 1] - cum_n[i_start]
        k = cum_k[i_end + 1] - cum_k[i_start]
        if n == 0:
            return -np.inf
        p = np.clip(k / n, eps, 1 - eps)
        return k * np.log(p) + (n - k) * np.log(1 - p)

    # dp[j][i] = best LL using j buckets covering scores[0..i]
    NEG_INF = -1e18
    dp   = np.full((n_buckets + 1, N), NEG_INF)
    split = np.full((n_buckets + 1, N), -1, dtype=int)  # for back-tracking

    # Base: 1 bucket covering scores[0..i]
    for i in range(N):
        dp[1][i] = ll_segment(0, i)

    # Fill table
    for j in range(2, n_buckets + 1):
        for i in range(j - 1, N):
            for m in range(j - 2, i):          # last boundary after index m
                val = dp[j - 1][m] + ll_segment(m + 1, i)
                if val > dp[j][i]:
                    dp[j][i]    = val
                    split[j][i] = m

    # Back-track to find boundaries
    boundaries_idx = []
    i = N - 1
    for j in range(n_buckets, 1, -1):
        m = split[j][i]
        boundaries_idx.append(m + 1)    # boundary is at start of new bucket
        i = m
    boundaries_idx.reverse()

    # Convert index positions → actual FICO score boundaries
    inner_scores = [int(scores[b]) for b in boundaries_idx]
    lo = int(fico.min()) - 1
    hi = int(fico.max()) + 1
    boundaries = sorted([lo] + inner_scores + [hi])

    best_ll = dp[n_buckets][N - 1]
    return boundaries, best_ll


# ══════════════════════════════════════════════════════════════════════════════
#  5.  BUILD RATING MAP FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_rating_map(fico, default, n_buckets=5, method='log_likelihood'):
    """
    Construct an optimal FICO → credit rating map.

    Parameters
    ----------
    fico       : array of FICO scores
    default    : array of 0/1 default flags
    n_buckets  : number of rating categories
    method     : 'log_likelihood' (DP) | 'mse'

    Returns
    -------
    dict with:
        boundaries   : list of boundary values
        rating_map   : function(score) → rating (1=best)
        bucket_table : DataFrame summary
        score        : LL or MSE of the solution
    """
    # Sort so bucket 1 = best credit (highest FICO scores)
    # We bin in ascending order then reverse the rating labels
    if method == 'log_likelihood':
        boundaries, score = optimize_log_likelihood_dp(fico, default, n_buckets)
        metric_name = 'Log-Likelihood'
    elif method == 'mse':
        boundaries, score = optimize_mse(fico, default, n_buckets)
        metric_name = 'MSE'
    else:
        raise ValueError(f"Unknown method: {method}")

    ns, ks, pds = bucket_stats(fico, default, boundaries)

    # Rating: bucket with LOWEST FICO scores gets HIGHEST rating number (worst credit)
    rows = []
    for i in range(n_buckets):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        # Rating 1 = best = highest FICO bucket (last bucket in ascending order)
        rating = n_buckets - i        # reverse: bucket 0 (lowest FICO) → rating n
        rows.append({
            'Rating':       rating,
            'FICO Low':     lo + 1 if i > 0 else lo,
            'FICO High':    hi if i < n_buckets - 1 else hi,
            'Count':        int(ns[i]),
            'Defaults':     int(ks[i]),
            'PD (%)':       round(pds[i] * 100, 2),
        })

    table = pd.DataFrame(rows).sort_values('Rating').reset_index(drop=True)

    def rating_map_fn(score):
        for i in range(n_buckets):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            in_bucket = (score >= lo and score < hi) if i < n_buckets - 1 else (score >= lo and score <= hi)
            if in_bucket:
                return n_buckets - i
        return n_buckets   # fallback

    return {
        'boundaries':   boundaries,
        'rating_map':   rating_map_fn,
        'bucket_table': table,
        'score':        score,
        'metric':       metric_name,
        'method':       method,
        'n_buckets':    n_buckets,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6.  RUN FOR MULTIPLE BUCKET COUNTS — COMPARE METHODS
# ══════════════════════════════════════════════════════════════════════════════
bucket_counts = [5, 7, 10]
all_results   = {}

for n in bucket_counts:
    print(f"\n{'═'*60}")
    print(f"  {n} BUCKETS")
    print('═'*60)
    for method in ['log_likelihood', 'mse']:
        res = build_rating_map(fico, default, n_buckets=n, method=method)
        all_results[(n, method)] = res
        print(f"\n  Method: {res['metric']}")
        print(f"  Boundaries: {res['boundaries']}")
        print(f"  {res['metric']}: {res['score']:.4f}")
        print(res['bucket_table'].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
#  7.  DEMO: rate a new borrower
# ══════════════════════════════════════════════════════════════════════════════
best_res = all_results[(5, 'log_likelihood')]

print(f"\n{'═'*60}")
print("  DEMO — RATING NEW BORROWERS (5-bucket log-likelihood model)")
print('═'*60)
sample_scores = [450, 550, 620, 680, 720, 780, 820]
for s in sample_scores:
    r = best_res['rating_map'](s)
    row = best_res['bucket_table'][best_res['bucket_table']['Rating'] == r].iloc[0]
    print(f"  FICO {s:>3}  →  Rating {r}  |  Bucket PD = {row['PD (%)']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  8.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
BLUE, RED, GREEN, ORANGE, PURPLE = '#1F77B4','#D62728','#2CA02C','#FF7F0E','#9467BD'
COLORS = [BLUE, GREEN, ORANGE, RED, PURPLE, '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#AEC7E8']

fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor('#F7F9FC')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32,
                        left=0.07, right=0.97, top=0.94, bottom=0.04)
fig.text(0.5, 0.97, 'FICO Score Quantization — Optimal Bucket Construction',
         ha='center', fontsize=14, fontweight='bold')

# ── Panel 1: raw FICO distribution with default overlay ──────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#F7F9FC')
bins = np.arange(int(fico.min()), int(fico.max()) + 5, 5)
ax1.hist(fico[default == 0], bins=bins, alpha=0.6, color=GREEN,  label='No Default', density=False)
ax1.hist(fico[default == 1], bins=bins, alpha=0.6, color=RED,    label='Default',    density=False)

# Overlay 5-bucket LL boundaries
res5 = all_results[(5, 'log_likelihood')]
for b in res5['boundaries'][1:-1]:
    ax1.axvline(b, color='navy', lw=1.5, ls='--', alpha=0.8)
ax1.text(res5['boundaries'][1] + 2, ax1.get_ylim()[1] * 0.85 if ax1.get_ylim()[1] > 0 else 100,
         '← DP boundaries', color='navy', fontsize=8)

ax1.set_title('FICO Score Distribution by Default Status\n(dashed lines = 5-bucket DP/LL boundaries)',
              fontsize=11, fontweight='bold')
ax1.set_xlabel('FICO Score'); ax1.set_ylabel('Count')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.25)

# ── Panel 2: PD per bucket — log-likelihood 5 buckets ────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#F7F9FC')
tbl = res5['bucket_table']
bar_labels = [f"R{int(r)}\n{int(lo)}–{int(hi)}"
              for r, lo, hi in zip(tbl['Rating'], tbl['FICO Low'], tbl['FICO High'])]
bar_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(tbl)))[::-1]
bars = ax2.bar(bar_labels, tbl['PD (%)'], color=bar_colors, edgecolor='white', width=0.6, zorder=3)
for bar, val in zip(bars, tbl['PD (%)']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_title('PD per Rating Bucket\n(Log-Likelihood DP, 5 Buckets)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Rating (R1=Best Credit)'); ax2.set_ylabel('Probability of Default (%)')
ax2.grid(True, alpha=0.25, axis='y')

# ── Panel 3: bucket population counts ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#F7F9FC')
ax3.bar(bar_labels, tbl['Count'], color=bar_colors, edgecolor='white', width=0.6, zorder=3)
ax3.bar(bar_labels, tbl['Defaults'], color=[RED]*len(tbl), edgecolor='white',
        width=0.6, alpha=0.5, label='Defaults', zorder=4)
for i, (cnt, dft) in enumerate(zip(tbl['Count'], tbl['Defaults'])):
    ax3.text(i, cnt + 5, f'n={cnt:,}', ha='center', va='bottom', fontsize=8)
ax3.set_title('Population per Bucket\n(grey=total, red=defaults)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Rating'); ax3.set_ylabel('Count')
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25, axis='y')

# ── Panel 4: LL vs MSE boundary comparison for 5 buckets ─────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#F7F9FC')
res_mse = all_results[(5, 'mse')]

x_range = np.linspace(fico.min(), fico.max(), 500)
default_rate_smooth = []
window = 20
for x in x_range:
    mask = (fico >= x - window) & (fico <= x + window)
    if mask.sum() > 5:
        default_rate_smooth.append(default[mask].mean() * 100)
    else:
        default_rate_smooth.append(np.nan)

ax4.plot(x_range, default_rate_smooth, color=PURPLE, lw=2, label='Rolling default rate')
for b in res5['boundaries'][1:-1]:
    ax4.axvline(b, color=BLUE, lw=2, ls='--', alpha=0.9, label='LL boundary' if b == res5['boundaries'][1] else '')
for b in res_mse['boundaries'][1:-1]:
    ax4.axvline(b, color=ORANGE, lw=1.5, ls=':', alpha=0.9, label='MSE boundary' if b == res_mse['boundaries'][1] else '')
ax4.set_title('Boundary Positions: LL (blue dashed) vs MSE (orange dotted)\n(5 buckets each)',
              fontsize=10, fontweight='bold')
ax4.set_xlabel('FICO Score'); ax4.set_ylabel('Default Rate (%)')
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.25)

# ── Panel 5: LL score vs number of buckets ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#F7F9FC')
bucket_range = range(2, 12)
ll_scores  = []
mse_scores = []
for n in bucket_range:
    r_ll  = build_rating_map(fico, default, n_buckets=n, method='log_likelihood')
    r_mse = build_rating_map(fico, default, n_buckets=n, method='mse')
    ll_scores.append(r_ll['score'])
    mse_scores.append(r_mse['score'])

ax5_twin = ax5.twinx()
l1, = ax5.plot(list(bucket_range), ll_scores,  color=BLUE,   lw=2, marker='o', label='Log-Likelihood (↑ better)')
l2, = ax5_twin.plot(list(bucket_range), mse_scores, color=ORANGE, lw=2, marker='s', ls='--', label='MSE (↓ better)')
ax5.set_xlabel('Number of Buckets'); ax5.set_ylabel('Log-Likelihood', color=BLUE)
ax5_twin.set_ylabel('MSE', color=ORANGE)
ax5.tick_params(axis='y', labelcolor=BLUE)
ax5_twin.tick_params(axis='y', labelcolor=ORANGE)
ax5.set_title('Objective Score vs Number of Buckets', fontsize=11, fontweight='bold')
ax5.legend(handles=[l1, l2], fontsize=9, loc='lower right')
ax5.grid(True, alpha=0.25)

plt.savefig('/mnt/user-data/outputs/fico_quantization.png', dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print("\n  Visualization saved ✓")
