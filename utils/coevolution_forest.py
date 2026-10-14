import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss, accuracy_score

EPS = 1e-9


class CoevoObliqueForest:
    """Cooperative coevolution: K slots, each a species of M fixed topology oblique trees."""

    def __init__(self, K=25, M=30, depth=4, generations=40, collab_random=0.3,
                 mut_rate=0.1, sigma=0.25, tournament=3, alpha=1.0, seed=0):
        self.K, self.M, self.D, self.G = K, M, depth, generations
        self.n_nodes = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.mut_rate, self.sigma, self.tk, self.alpha = mut_rate, sigma, tournament, alpha
        self.rng = np.random.default_rng(seed)
        self.collab_random = collab_random

    # ---------- tree evaluation (vectorized over a batch of M trees) ----------
    def _route(self, G, X):
        """G: (m, nodes, d+1) genomes. Returns leaf index (m, n)."""
        W, b = G[:, :, :-1], G[:, :, -1]
        Z = np.einsum('nd,mkd->mnk', X, W, optimize=True) - b[:, None, :]
        m, n, _ = Z.shape
        idx = np.zeros((m, n), dtype=np.int64)
        for _ in range(self.D):
            z = np.take_along_axis(Z, idx[:, :, None], axis=2)[:, :, 0]
            idx = 2 * idx + 1 + (z > 0)
        return idx - self.n_nodes

    def _leaf_probs(self, G, X, y, boot):
        """Leaf class distributions from in bag samples. Returns (m, leaves, C)."""
        m = G.shape[0]
        leaf = self._route(G, X)
        flat = (np.arange(m)[:, None] * self.n_leaves + leaf) * self.C + y[None, :]
        cnt = np.bincount(flat.ravel(), weights=np.broadcast_to(boot, leaf.shape).ravel(),
                          minlength=m * self.n_leaves * self.C)
        cnt = cnt.reshape(m, self.n_leaves, self.C) + self.alpha
        return cnt / cnt.sum(axis=2, keepdims=True)

    def _predict_trees(self, G, LP, X):
        """Per tree probabilities for rows of X. (m, n, C)"""
        leaf = self._route(G, X)
        return np.take_along_axis(LP, leaf[:, :, None], axis=1) if False else \
            LP[np.arange(G.shape[0])[:, None], leaf]

    # ---------- initialization ----------
    def _init_slot(self, X, boot):
        n, d = X.shape
        G = np.zeros((self.M, self.n_nodes, d + 1))
        p = min(1.0, 3.0 / d)
        for m in range(self.M):
            idx = np.zeros(n, dtype=np.int64)
            for node in range(self.n_nodes):
                w = self.rng.normal(size=d) * (self.rng.random(d) < p)
                if not w.any():
                    w[self.rng.integers(d)] = self.rng.normal()
                w /= np.linalg.norm(w)
                G[m, node, :-1] = w
            # thresholds: median of in bag projections reaching each node, top down
            idx = np.zeros(n, dtype=np.int64)
            inbag = boot > 0
            for node in range(self.n_nodes):
                pass
            cur = np.zeros(n, dtype=np.int64)
            for level in range(self.D):
                for node in range(2 ** level - 1, 2 ** (level + 1) - 1):
                    sel = inbag & (cur == node)
                    proj = X @ G[m, node, :-1]
                    G[m, node, -1] = np.median(proj[sel]) if sel.sum() > 1 else 0.0
                z = np.einsum('nd,nd->n', X, G[m, cur, :-1]) - G[m, cur, -1]
                cur = 2 * cur + 1 + (z > 0)
        return G

    # ---------- evolution operators (node level crossover) ----------
    def _tournament(self, fit, n):
        cand = self.rng.integers(0, len(fit), size=(n, self.tk))
        return cand[np.arange(n), fit[cand].argmax(axis=1)]

    def _next_gen(self, G, fit):
        M = len(G)
        i1, i2 = self._tournament(fit, M), self._tournament(fit, M)
        mask = self.rng.random((M, self.n_nodes, 1)) < 0.5
        child = np.where(mask, G[i1], G[i2])
        mut = self.rng.random(child.shape) < self.mut_rate
        child = child + mut * self.rng.normal(0, self.sigma, child.shape)
        child[0] = G[fit.argmax()]  # elitism
        return child

    # ---------- ensemble scoring ----------
    @staticmethod
    def _logloss(S, cnt, y):
        ok = cnt > 0
        P = S[ok] / cnt[ok, None]
        return -np.mean(np.log(P[np.arange(ok.sum()), y[ok]] + EPS))

    def fit(self, X, y):
        n, d = X.shape
        self.C = int(y.max()) + 1
        K, M = self.K, self.M
        self.boots = np.stack([np.bincount(self.rng.integers(0, n, n), minlength=n) for _ in range(K)])
        self.oob = self.boots == 0  # (K, n)
        self.pops = [self._init_slot(X, self.boots[k]) for k in range(K)]

        # representatives = first member of each species initially (random oblique forest baseline)
        self.rep_G = np.stack([self.pops[k][0] for k in range(K)])
        self.rep_LP = np.stack([self._leaf_probs(self.rep_G[k:k + 1], X, y, self.boots[k])[0] for k in range(K)])
        self.contrib = np.zeros((K, n, self.C))
        for k in range(K):
            self.contrib[k] = self._predict_trees(self.rep_G[k:k + 1], self.rep_LP[k:k + 1], X)[0] * self.oob[k][:, None]
        self.S = self.contrib.sum(axis=0)
        self.cnt = self.oob.sum(axis=0).astype(float)
        self.init_G, self.init_LP = self.rep_G.copy(), self.rep_LP.copy()
        self.history = [self._logloss(self.S, self.cnt, y)]

        for gen in range(self.G):
            for k in range(K):
                pop = self.pops[k]
                LP = self._leaf_probs(pop, X, y, self.boots[k])
                P = self._predict_trees(pop, LP, X)  # (M, n, C)
                base = self.S - self.contrib[k]  # votes from other slots
                oobk = self.oob[k][:, None]
                fit = np.empty(M)
                for m in range(M):
                    fit[m] = -self._logloss(base + P[m] * oobk, self.cnt, y)
                best = int(fit.argmax())
                self.rep_G[k], self.rep_LP[k] = pop[best], LP[best]
                self.contrib[k] = P[best] * oobk
                self.S = base + self.contrib[k]
                self.pops[k] = self._next_gen(pop, fit)
            self.history.append(self._logloss(self.S, self.cnt, y))
        return self

    def _proba(self, G, LP, X):
        P = self._predict_trees(G, LP, X)
        return P.mean(axis=0)

    def predict_proba(self, X):
        return self._proba(self.rep_G, self.rep_LP, X)

    def predict_proba_initial(self, X):
        return self._proba(self.init_G, self.init_LP, X)


def datasets():
    out = {}
    d = load_breast_cancer(); out['breast_cancer'] = (d.data, d.target)
    d = load_wine(); out['wine'] = (d.data, d.target)
    X, y = make_classification(n_samples=800, n_features=20, n_informative=10, n_redundant=6,
                               n_clusters_per_class=2, flip_y=0.03, class_sep=0.8, random_state=1)
    out['synthetic'] = (X, y)
    return out


def score(P, y):
    return accuracy_score(y, P.argmax(1)), log_loss(y, np.clip(P, EPS, 1), labels=np.arange(P.shape[1]))


if __name__ == '__main__':
    K, G = 100, 40
    rows = []
    for name, (X, y) in datasets().items():
        for seed in range(3):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            t = time.time()
            ga = CoevoObliqueForest(K=K, generations=G, seed=seed).fit(Xtr, ytr)
            tga = time.time() - t
            res = {'gen0 oblique': score(ga.predict_proba_initial(Xte), yte),
                   'coevo oblique': score(ga.predict_proba(Xte), yte)}
            for label, mdl in [('RF d4', RandomForestClassifier(K, max_depth=4, random_state=seed)),
                               ('ExtraTrees d4', ExtraTreesClassifier(K, max_depth=4, random_state=seed))]:
                mdl.fit(Xtr, ytr)
                res[label] = score(mdl.predict_proba(Xte), yte)
            rows.append((name, seed, res, ga.history[0], ga.history[-1], tga))
            print(f"{name} seed{seed} time {tga:.1f}s OOB logloss {ga.history[0]:.3f} -> {ga.history[-1]:.3f}", flush=True)
            for k, (a, l) in res.items():
                print(f"   {k:15s} acc {a:.3f} logloss {l:.3f}", flush=True)

    print("\nMEAN OVER SEEDS (test acc / test logloss)")
    for name in datasets():
        sub = [r for r in rows if r[0] == name]
        print(name)
        for k in sub[0][2]:
            a = np.mean([r[2][k][0] for r in sub]); l = np.mean([r[2][k][1] for r in sub])
            print(f"   {k:15s} acc {a:.3f} logloss {l:.3f}")