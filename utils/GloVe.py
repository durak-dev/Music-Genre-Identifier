import numpy as np
import polars as pl
import torch
from torch import nn


class GloVe(nn.Module):
    """GloVe trained on (word_i, word_j, count) triples stored as .npy files.

    Usage:
        glove = GloVe(vocab, dim=25)
        glove.fit("i.npy", "j.npy", "x.npy", epochs=20)
        glove.most_similar("love")
        glove.save("glove_25d")
    """

    def __init__(self, vocab, dim=25, x_max=100.0, alpha=0.75, device="cpu"):
        super().__init__()
        self.vocab = list(vocab)
        self.index = {word: k for k, word in enumerate(self.vocab)}
        self.dim, self.x_max, self.alpha = dim, x_max, alpha
        self.device = device

        V = len(self.vocab)
        self.w = nn.Embedding(V, dim)      # word vectors
        self.c = nn.Embedding(V, dim)      # context vectors
        self.bw = nn.Embedding(V, 1)       # word biases
        self.bc = nn.Embedding(V, 1)       # context biases
        for emb in (self.w, self.c):
            nn.init.uniform_(emb.weight, -0.5 / dim, 0.5 / dim)
        for emb in (self.bw, self.bc):
            nn.init.zeros_(emb.weight)
        self.to(device)
        self._vectors = None               # cached result after fit

    # ---------- model ----------
    def forward(self, i, j):
        dot = (self.w(i) * self.c(j)).sum(dim=1)
        return dot + self.bw(i).squeeze(1) + self.bc(j).squeeze(1)

    def _loss(self, i, j, counts):
        weight = torch.clamp(counts / self.x_max, max=1.0) ** self.alpha
        return (weight * (self(i, j) - torch.log(counts)) ** 2).mean()

    # ---------- training ----------
    def fit(self, i_path, j_path, x_path, epochs=20, batch_size=1 << 20, lr=0.05):
        # memory-mapped: the arrays stay on disk and are read one batch at a time
        i = np.load(i_path, mmap_mode="r")
        j = np.load(j_path, mmap_mode="r")
        x = np.load(x_path, mmap_mode="r")
        n = len(x)
        starts = np.arange(0, n, batch_size)
        opt = torch.optim.Adagrad(self.parameters(), lr=lr)

        for epoch in range(epochs):
            total = 0.0
            for start in np.random.permutation(starts):
                sl = slice(start, start + batch_size)
                a = self._tensor(i[sl], torch.int64)
                b = self._tensor(j[sl], torch.int64)
                c = self._tensor(x[sl], torch.float32)
                loss = self._loss(a, b, c)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"epoch {epoch + 1}/{epochs}  loss {total / len(starts):.4f}")

        self._vectors = None
        return self

    def _tensor(self, arr, dtype):
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=self.device)

    # ---------- results ----------
    @property
    def vectors(self):
        """(V, dim) float32 array: word + context vectors, the standard GloVe output."""
        if self._vectors is None:
            with torch.no_grad():
                self._vectors = (self.w.weight + self.c.weight).cpu().numpy().astype(np.float32)
        return self._vectors

    def __getitem__(self, word):
        return self.vectors[self.index[word]]

    def most_similar(self, word, k=8):
        v = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        scores = v @ v[self.index[word]]
        best = np.argsort(-scores)[1 : k + 1]
        return [(self.vocab[b], round(float(scores[b]), 3)) for b in best]

    # ---------- persistence ----------
    def save(self, path):
        np.save(f"{path}.npy", self.vectors)
        pl.DataFrame({"word": self.vocab}).write_parquet(f"{path}_vocab.parquet")

    @staticmethod
    def load_vectors(path):
        """Returns (vectors, vocab) without needing torch or retraining."""
        vectors = np.load(f"{path}.npy")
        vocab = pl.read_parquet(f"{path}_vocab.parquet")["word"].to_list()
        return vectors, vocab