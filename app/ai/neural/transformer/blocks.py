"""Transformer blocks with forward AND backward pass support.

Multi-Head Self-Attention, Feed-Forward, LayerNorm — all with proper
backpropagation so the model actually learns.
"""
import math
import random

import numpy as np


def _randn(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    limit = math.sqrt(2.0 / (rows + cols))
    return np.random.normal(0.0, limit, size=(rows, cols)).astype(np.float64)


def _zeros(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.float64)


def _softmax(x: list[float]) -> list[float]:
    max_x = max(x)
    exp_x = [math.exp(v - max_x) for v in x]
    total = sum(exp_x)
    return [v / total for v in exp_x]


def _layer_norm_forward(x: list[float], gamma: list[float], beta: list[float], eps: float = 1e-5):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    x_norm = [(x[i] - mean) / std for i in range(len(x))]
    out = [gamma[i] * x_norm[i] + beta[i] for i in range(len(x))]
    return out, x_norm, mean, std


def _layer_norm_backward(grad_out: list[float], x_norm: list[float], gamma: list[float], std: float):
    n = len(grad_out)
    dx_norm = [grad_out[i] * gamma[i] for i in range(n)]
    dvar = sum(dx_norm[i] * (x_norm[i]) for i in range(n)) * (-0.5) / (std ** 3)
    dmean = sum(dx_norm[i] for i in range(n)) * (-1.0) / std + dvar * sum(-2.0 * x_norm[i] for i in range(n)) / n
    dx = [dx_norm[i] / std + dvar * 2.0 * x_norm[i] / n + dmean / n for i in range(n)]
    dgamma = [grad_out[i] * x_norm[i] for i in range(n)]
    dbeta = list(grad_out)
    return dx, dgamma, dbeta


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_deriv(x: float) -> float:
    """Derivative of GELU."""
    tanh_val = math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))
    sech2 = 1.0 - tanh_val ** 2
    dx = math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dx


def _gelu_deriv_np(x: np.ndarray) -> np.ndarray:
    tanh_val = np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))
    sech2 = 1.0 - tanh_val ** 2
    dx_vec = np.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dx_vec


class Linear:
    """Capa linear con soporte backward."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _randn(in_features, out_features)
        self.bias = np.zeros(out_features, dtype=np.float64) if bias else None
        self._grad_weight = _zeros(in_features, out_features)
        self._grad_bias = np.zeros(out_features, dtype=np.float64) if bias else None
        self._last_input: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        self._last_input = x
        # Vectorizado con NumPy sobre pesos persistentes: [out] = [in] @ [in][out]
        result = np.dot(np.asarray(x, dtype=np.float64), self.weight)
        if self.bias is not None:
            result += self.bias
        return result.tolist()

    def backward(self, grad_output: list[float], lr: float = 0.001) -> list[float]:
        xi = np.asarray(self._last_input, dtype=np.float64)     # [in]
        go = np.asarray(grad_output, dtype=np.float64)          # [out]
        # SGD correcto: dW = outer(xi, go) ; dX = W @ go (con W original)
        self._grad_weight += xi[:, None] * go[None, :]
        d_in = self.weight @ go
        self.weight -= lr * (xi[:, None] * go[None, :])
        if self.bias is not None and self._grad_bias is not None:
            self._grad_bias += go
            self.bias -= lr * go
        return d_in.tolist()


class Embedding:
    """Embedding con soporte backward."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = _randn(vocab_size, embed_dim)
        self._last_ids: list[int] = []

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        self._last_ids = token_ids
        return self.weight[token_ids].tolist()

    def backward(self, grad_output: list[list[float]], lr: float = 0.001) -> None:
        for idx, tid in enumerate(self._last_ids):
            if tid < self.vocab_size:
                self.weight[tid, :] -= lr * np.asarray(grad_output[idx], dtype=np.float64)


class MultiHeadAttention:
    """Multi-Head Self-Attention con backward pass."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self._cache_k: list[list[float]] = []
        self._cache_v: list[list[float]] = []
        self._cache_pos = 0
        self._fwd: dict | None = None

    def forward(self, x: list[list[float]], use_cache: bool = False) -> list[list[float]]:
        seq_len = len(x)
        xa = np.asarray(x, dtype=np.float64)  # [seq, embed]

        Wq = np.asarray(self.q_proj.weight, dtype=np.float64)
        Wk = np.asarray(self.k_proj.weight, dtype=np.float64)
        Wv = np.asarray(self.v_proj.weight, dtype=np.float64)
        Wo = np.asarray(self.out_proj.weight, dtype=np.float64)

        Q = xa @ Wq  # [seq, embed]
        K = xa @ Wk
        V = xa @ Wv

        if use_cache:
            if not self._cache_k:
                self._cache_k = K.tolist()
                self._cache_v = V.tolist()
            else:
                self._cache_k.extend(K.tolist())
                self._cache_v.extend(V.tolist())
            self._cache_pos += seq_len
            K_all = np.asarray(self._cache_k, dtype=np.float64)
            V_all = np.asarray(self._cache_v, dtype=np.float64)
        else:
            K_all = K
            V_all = V

        k_len = K_all.shape[0]

        # Reshape a heads (heads son slices contiguos dentro de cada posición)
        Q_h = Q.reshape(seq_len, self.num_heads, self.head_dim)        # [seq, heads, hd]
        K_h = K_all.reshape(k_len, self.num_heads, self.head_dim)      # [k_len, heads, hd]
        V_h = V_all.reshape(k_len, self.num_heads, self.head_dim)      # [k_len, heads, hd]

        scale = math.sqrt(self.head_dim)

        # scores[heads, i, j] = sum_d Q[i,hd,d]*K[j,hd,d] / scale
        scores = np.einsum('ihd,jhd->hij', Q_h, K_h) / scale

        # Máscara causal
        if use_cache:
            real_i = k_len - seq_len + np.arange(seq_len)
            mask = np.arange(k_len)[None, :] > real_i[:, None]  # [seq, k_len]
        else:
            mask = np.arange(k_len)[None, :] > np.arange(seq_len)[:, None]
        scores[:, mask] = -1e9

        # Softmax sobre el último eje
        scores_max = scores.max(axis=2, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / exp_scores.sum(axis=2, keepdims=True)    # [heads, seq, k_len]

        # out[heads, i, d] = sum_j probs[hd,i,j] * V_h[j,hd,d]
        head_out = np.einsum('hij,jhd->hid', probs, V_h)              # [heads, seq, hd]

        # Concatenar cabezas por posición y proyectar
        out_in = head_out.transpose(1, 0, 2).reshape(seq_len, self.embed_dim)  # [seq, embed]
        output = out_in @ Wo

        self._fwd = {
            "Q": Q.tolist(),
            "K": K_all.tolist(),
            "V": V_all.tolist(),
            "probs": [[[float(p[i][j]) for j in range(k_len)] for i in range(seq_len)] for p in probs],
            "proj_in": [xi[:] for xi in x],
            "seq_len": seq_len,
            "k_len": k_len,
        }
        self._fwd["out_in"] = out_in.tolist()

        return output.tolist()

    def _proj_backward(self, weights, inputs, grads, lr, in_size, out_size):
        """Backprop de projections lineales (SGD correcto), vectorizado.

        weights: [in][out] (ndarray); inputs: n x in; grads: n x out.
        Primero calcula todo el gradiente con los pesos originales y luego
        actualiza una sola vez. Devuelve gradientes de entrada (n x in).
        """
        inputs_a = np.asarray(inputs, dtype=np.float64)   # [n, in]
        grads_a = np.asarray(grads, dtype=np.float64)     # [n, out]
        d_in = grads_a @ weights                          # [n, in]  (weights original)
        weights -= lr * (inputs_a.T @ grads_a)            # [in, out]  dW = inputs.T @ grads
        return d_in.tolist()

    def backward(self, dout: list[list[float]], lr: float = 0.001) -> list[list[float]]:
        """Backprop a través de la atención causal (SGD correcto, vectorizado)."""
        fwd = self._fwd
        Q = np.asarray(fwd["Q"], dtype=np.float64)       # [seq, embed]
        K = np.asarray(fwd["K"], dtype=np.float64)       # [k_len, embed]
        V = np.asarray(fwd["V"], dtype=np.float64)       # [k_len, embed]
        probs = np.asarray(fwd["probs"], dtype=np.float64)  # [heads, seq, k_len]
        seq_len = fwd["seq_len"]
        k_len = fwd["k_len"]
        scale = math.sqrt(self.head_dim)
        do = np.asarray(dout, dtype=np.float64)          # [seq, embed]

        # out_proj backward -> gradiente de la concatenación por posición
        dc = self._proj_backward(
            self.out_proj.weight, fwd["out_in"], dout, lr,
            self.embed_dim, self.embed_dim,
        )
        dc = np.asarray(dc, dtype=np.float64)            # [seq, embed]

        # Dividir por cabezas (slices contiguos) -> [seq, heads, hd] / [k_len, heads, hd]
        dc_h = dc.reshape(seq_len, self.num_heads, self.head_dim)      # [seq, hh, hd]
        Q_h = Q.reshape(seq_len, self.num_heads, self.head_dim)        # [seq, hh, hd]
        K_h = K.reshape(k_len, self.num_heads, self.head_dim)          # [k_len, hh, hd]
        V_h = V.reshape(k_len, self.num_heads, self.head_dim)          # [k_len, hh, hd]

        # dS[hh, i, j] = sum_d dc[i,h,d]*V[j,h,d]
        dS = np.einsum('ihd,jhd->hij', dc_h, V_h)                       # [hh, seq, k_len]

        # dV[hh, j, d] += P[hh, i, j]*dc[i,h,d]
        dV_h = np.einsum('hij,ihd->jhd', probs, dc_h)                   # [k_len, hh, hd]

        # softmax gradiente: da = P*(dS - dot(P, dS))
        dot_p_dS = np.einsum('hij,hij->hi', probs, dS)                  # [hh, seq]
        da = probs * (dS - dot_p_dS[:, :, None])                        # [hh, seq, k_len]

        # dK[hh, j, d] += da[i,j]*Q[i,h,d]/scale ; dQ[hh, i, d] += da[i,j]*K[j,h,d]/scale
        dK_h = np.einsum('hij,ihd->jhd', da, Q_h) / scale               # [k_len, hh, hd]
        dQ_h = np.einsum('hij,jhd->ihd', da, K_h) / scale               # [seq, hh, hd]

        # Reconstruir gradientes densos (des-concatenar cabezas)
        dQ = dQ_h.reshape(seq_len, self.embed_dim)                      # [seq, embed]
        dK = dK_h.reshape(k_len, self.embed_dim)                        # [k_len, embed]
        dV = dV_h.reshape(k_len, self.embed_dim)                        # [k_len, embed]

        proj_in = fwd["proj_in"]
        dq = np.asarray(self._proj_backward(
            self.q_proj.weight, proj_in, dQ.tolist(), lr, self.embed_dim, self.embed_dim), dtype=np.float64)
        dk = np.asarray(self._proj_backward(
            self.k_proj.weight, proj_in, dK.tolist(), lr, self.embed_dim, self.embed_dim), dtype=np.float64)
        dv = np.asarray(self._proj_backward(
            self.v_proj.weight, proj_in, dV.tolist(), lr, self.embed_dim, self.embed_dim), dtype=np.float64)

        return (dq + dk + dv).tolist()

    def clear_cache(self) -> None:
        self._cache_k = []
        self._cache_v = []
        self._cache_pos = 0
        self._fwd = None


class FeedForward:
    """Feed-Forward Network con backward (GELU activation)."""

    def __init__(self, embed_dim: int, ff_mult: int = 4) -> None:
        hidden = embed_dim * ff_mult
        self.fc1 = Linear(embed_dim, hidden)
        self.fc2 = Linear(hidden, embed_dim)

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        self._fc1_in = [xi[:] for xi in x]
        xa = np.asarray(x, dtype=np.float64)                       # [n, embed]
        W1 = np.asarray(self.fc1.weight, dtype=np.float64)         # [embed, hidden]
        W2 = np.asarray(self.fc2.weight, dtype=np.float64)         # [hidden, embed]

        pre_gelu = xa @ W1                                          # [n, hidden]
        gelu_h = _gelu_np(pre_gelu)                                 # [n, hidden]
        result = gelu_h @ W2                                        # [n, embed]

        self._pre_gelu = pre_gelu.tolist()
        self._fc2_in = gelu_h.tolist()
        return result.tolist()

    def backward(self, dout: list[list[float]], lr: float = 0.001) -> list[list[float]]:
        """Backprop: fc2 -> GELU -> fc1 (SGD correcto, vectorizado con NumPy)."""
        dout_a = np.asarray(dout, dtype=np.float64)                  # [n, embed]
        fc2_in = np.asarray(self._fc2_in, dtype=np.float64)          # [n, hidden] (entrada de fc2 = gelu)
        w2 = self.fc2.weight                                         # [hidden, embed]

        # fc2: y = fc2_in @ w2
        dx_gelu = dout_a @ w2.T                                      # [n, hidden]  (w2 original)
        w2 -= lr * (fc2_in.T @ dout_a)                               # correcto: dW2 = fc2_in.T @ dout

        # GELU
        dx1 = _gelu_deriv_np(np.asarray(self._pre_gelu, dtype=np.float64)) * dx_gelu   # [n, hidden]

        # fc1: h = fc1_in @ w1
        fc1_in = np.asarray(self._fc1_in, dtype=np.float64)          # [n, embed]
        w1 = self.fc1.weight                                         # [embed, hidden]
        dx = dx1 @ w1.T                                              # [n, embed]  (w1 original)
        w1 -= lr * (fc1_in.T @ dx1)                                  # correcto: dW1 = fc1_in.T @ dx1

        return dx.tolist()

    def clear_cache(self) -> None:
        pass


class TransformerBlock:
    """Bloque Transformer con LayerNorm + Attention + FeedForward + Residuals."""

    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int = 4) -> None:
        self.embed_dim = embed_dim
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_mult)
        self.ln1_gamma = np.ones(embed_dim, dtype=np.float64)
        self.ln1_beta = np.zeros(embed_dim, dtype=np.float64)
        self.ln2_gamma = np.ones(embed_dim, dtype=np.float64)
        self.ln2_beta = np.zeros(embed_dim, dtype=np.float64)

    def forward(self, x: list[list[float]], use_cache: bool = False) -> list[list[float]]:
        # LayerNorm -> Attention -> Residual
        normed1, self._ln1_xnorm, self._ln1_mean, self._ln1_std = zip(
            *[_layer_norm_forward(xi, self.ln1_gamma, self.ln1_beta) for xi in x]
        )
        self._ln1_xnorm = list(self._ln1_xnorm)
        attn_out = self.attn.forward(list(normed1), use_cache=use_cache)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]

        # LayerNorm -> FeedForward -> Residual
        normed2, self._ln2_xnorm, self._ln2_mean, self._ln2_std = zip(
            *[_layer_norm_forward(xi, self.ln2_gamma, self.ln2_beta) for xi in x]
        )
        self._ln2_xnorm = list(self._ln2_xnorm)
        ff_out = self.ff.forward(list(normed2))
        x = [[x[i][j] + ff_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]

        return x

    def backward(self, dout: list[list[float]], lr: float = 0.001) -> list[list[float]]:
        """Backprop completo a través del bloque transformer."""
        embed_dim = self.embed_dim
        n = len(dout)

        # Rama FeedForward: d(normed2) -> LN2 -> grad. entrada (x post-attn)
        dx_n2_in = self.ff.backward(dout, lr)
        dx2 = self._ln_backward(self._ln2_xnorm, self._ln2_std, self.ln2_gamma, self.ln2_beta, dx_n2_in, lr)

        # Residual: la salida del bloque también fluye directo a su entrada
        d_res = [[dout[i][d] + dx2[i][d] for d in range(embed_dim)] for i in range(n)]

        # Rama Attention: d(normed1) -> Attn -> LN1
        d_normed1 = self.attn.backward(d_res, lr)
        dx1 = self._ln_backward(self._ln1_xnorm, self._ln1_std, self.ln1_gamma, self.ln1_beta, d_normed1, lr)

        return [[d_res[i][d] + dx1[i][d] for d in range(embed_dim)] for i in range(n)]

    def _ln_backward(self, xnorm, stds, gamma, beta, grad_in, lr):
        """LayerNorm backward vectorizado (SGD correcto: acumula y aplica una vez)."""
        g = np.asarray(grad_in, dtype=np.float64)      # [m, embed]
        xn = np.asarray(xnorm, dtype=np.float64)       # [m, embed]
        st = np.asarray(stds, dtype=np.float64)        # [m]
        embed = self.embed_dim
        dGrad_norm = g * gamma[None, :]
        dvar = (dGrad_norm * xn).sum(axis=1) * (-0.5) / (st ** 3)
        dmean = dGrad_norm.sum(axis=1) * (-1.0) / st + dvar * (-2.0 * xn).sum(axis=1) / embed
        dx = dGrad_norm / st[:, None] + (dvar[:, None] * 2.0 * xn) / embed + dmean[:, None] / embed
        dgamma = (g * xn).sum(axis=0)
        dbeta = g.sum(axis=0)
        gamma -= lr * dgamma
        beta -= lr * dbeta
        return dx.tolist()

    def clear_cache(self) -> None:
        self.attn.clear_cache()
        self.ff.clear_cache()
