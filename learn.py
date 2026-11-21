"""Feature weights + a feedback loop: star ratings from the user slowly re-weight what makes a clip good.

Every candidate clip gets features in 0..1: loud (sudden loudness), shout (raised voice), words (exclamations said),
motion, flash, strobe, cuts, llm (local AI score / 10). Score = weighted sum of the features that are available.
Ratings (1-5 stars) are stored locally in data/feedback.jsonl. With enough ratings the weights are fitted by ridge regression that
pulls back toward the defaults, blended in gradually (alpha = n/60, max 0.85). Nothing is uploaded.
"""
import json
import time
from pathlib import Path
import numpy as np

FEATURES = ['loud', 'shout', 'words', 'motion', 'flash', 'strobe', 'cuts', 'llm', 'chat']
DEFAULT = {'loud': .17, 'shout': .13, 'words': .20, 'motion': .17, 'flash': .08, 'strobe': .04, 'cuts': .03, 'llm': .18, 'chat': .14}
MIN_RATINGS = 8


def _paths(data):
    return Path(data) / 'feedback.jsonl', Path(data) / 'weights.json'


def load_weights(data):
    _, wp = _paths(data)
    try:
        w = json.loads(wp.read_text(encoding='utf-8'))
        return {k: float(w.get(k, DEFAULT[k])) for k in FEATURES}
    except (OSError, ValueError):
        return dict(DEFAULT)


def score(features, weights):
    """Weighted mean over the features that exist (missing ones do not count as zero)."""
    num = den = 0.0
    for k in FEATURES:
        if features.get(k) is None:
            continue
        num += weights[k] * float(features[k])
        den += weights[k]
    return num / den if den else 0.0


def ratings(data):
    fp, _ = _paths(data)
    out = []
    if fp.exists():
        for line in fp.read_text(encoding='utf-8').splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def record(data, file, rating, features, extra=None):
    fp, _ = _paths(data)
    row = {'t': int(time.time()), 'file': file, 'rating': int(rating), 'features': features, **(extra or {})}
    with fp.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return refit(data)


def refit(data):
    """Ridge regression of rating on features, shrunk toward DEFAULT. Returns a summary dict (also saves weights.json)."""
    rows = [r for r in ratings(data) if r.get('features')]
    n = len(rows)
    summary = {'ratings': n, 'weights': dict(DEFAULT), 'fitted': False, 'insight': []}
    if n >= MIN_RATINGS:
        X = np.array([[float(r['features'].get(k) if r['features'].get(k) is not None else 0.0) for k in FEATURES] for r in rows])
        y = np.array([(r['rating'] - 3) / 2.0 for r in rows])      # -1..1
        # correlation per feature (what the data says)
        corr = {}
        for j, k in enumerate(FEATURES):
            if X[:, j].std() > 1e-6 and y.std() > 1e-6:
                corr[k] = float(np.corrcoef(X[:, j], y)[0, 1])
        w0 = np.array([DEFAULT[k] for k in FEATURES])
        lam = 2.0
        # solve min ||X w - y||^2 + lam ||w - w0||^2  (centered features)
        Xc = X - X.mean(axis=0)
        A = Xc.T @ Xc + lam * np.eye(len(FEATURES))
        b = Xc.T @ y + lam * w0
        wf = np.clip(np.linalg.solve(A, b), 0.01, None)
        wf = wf / wf.sum()
        alpha = min(0.85, n / 60.0)
        w = (1 - alpha) * w0 + alpha * wf
        w = w / w.sum()
        summary.update(weights={k: round(float(v), 4) for k, v in zip(FEATURES, w)}, fitted=True, alpha=round(alpha, 2))
        ranked = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:3]
        summary['insight'] = [f"{k}: {'higher' if c > 0 else 'lower'} = better rated ({c:+.2f})" for k, c in ranked]
    _, wp = _paths(data)
    wp.write_text(json.dumps(summary['weights']), encoding='utf-8')
    return summary
