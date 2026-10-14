import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from coevolution_forest import CoevoObliqueForest, datasets, EPS


class BestOfM(CoevoObliqueForest):
    """Same initial species, same fitness, but no breeding.
    With generations=1 each slot just keeps the best of its M random trees."""

    def _next_gen(self, G, fit):
        return G


def make_models(K, G, seed):
    return {
        'coevo': CoevoObliqueForest(K=K, generations=G, seed=seed),
        'best of M, no breeding': BestOfM(K=K, generations=1, seed=seed),
        'RF d4': RandomForestClassifier(K, max_depth=4, random_state=seed),
        'RF full 500': RandomForestClassifier(500, random_state=seed, n_jobs=-1),
        'RF full 500 calibrated': CalibratedClassifierCV(
            RandomForestClassifier(500, random_state=seed, n_jobs=-1),
            method='isotonic', cv=3),
        'ExtraTrees full 500': ExtraTreesClassifier(500, random_state=seed, n_jobs=-1),
    }


def metrics(P, y):
    C = P.shape[1]
    P = np.clip(P, EPS, 1)
    P = P / P.sum(axis=1, keepdims=True)
    onehot = np.eye(C)[y]
    if C == 2:
        auc = roc_auc_score(y, P[:, 1])
    else:
        auc = roc_auc_score(y, P, multi_class='ovr', labels=np.arange(C))
    return {
        'acc': accuracy_score(y, P.argmax(1)),
        'logloss': log_loss(y, P, labels=np.arange(C)),
        'brier': np.mean(np.sum((P - onehot) ** 2, axis=1)),
        'auc': auc,
    }


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return wilcoxon(a, b).pvalue


def run(K=50, G=40, n_repeats=5):
    results = {}  # (dataset, model) -> list of metric dicts, one per fold
    for ds, (X, y) in datasets().items():
        y = LabelEncoder().fit_transform(y)
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=n_repeats, random_state=0)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            for name, mdl in make_models(K, G, fold).items():
                t = time.time()
                mdl.fit(Xtr, y[tr])
                m = metrics(mdl.predict_proba(Xte), y[te])
                m['time'] = time.time() - t
                results.setdefault((ds, name), []).append(m)
            print(f'{ds} fold {fold + 1}/{2 * n_repeats} done', flush=True)
    return results


def report(results):
    cols = ['acc', 'logloss', 'brier', 'auc', 'time']
    for ds in sorted({k[0] for k in results}):
        print(f'\n=== {ds} (mean over folds) ===')
        print(f"{'model':28s}" + ''.join(f'{c:>10s}' for c in cols))
        names = [k[1] for k in results if k[0] == ds]
        for n in names:
            rows = results[(ds, n)]
            print(f'{n:28s}' + ''.join(f'{np.mean([r[c] for r in rows]):10.3f}' for c in cols))

        print('paired Wilcoxon, coevo minus other (negative logloss/brier diff favors coevo)')
        for n in names:
            if n == 'coevo':
                continue
            line = f'  vs {n:26s}'
            for c in ['acc', 'logloss', 'brier', 'auc']:
                a = [r[c] for r in results[(ds, 'coevo')]]
                b = [r[c] for r in results[(ds, n)]]
                line += f' {c} {np.mean(a) - np.mean(b):+.3f} (p={paired_p(a, b):.2f})'
            print(line)


if __name__ == '__main__':
    report(run(K=50, G=40, n_repeats=5))