"""Transformer blocks with forward AND backward pass support.

Multi-Head Self-Attention, Feed-Forward, LayerNorm — all with proper
backpropagation so the model actually learns.
"""
import math
import random


def _randn(rows: int, cols: int, scale: float = 0.02) -> list[list[float]]:
    limit = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, limit) for _ in range(cols)] for _ in range(rows)]


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


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


def _gelu_deriv(x: float) -> float:
    """Derivative of GELU."""
    tanh_val = math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))
    sech2 = 1.0 - tanh_val ** 2
    dx = math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dx


class Linear:
    """Capa linear con soporte backward."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _randn(in_features, out_features)
        self.bias = [0.0] * out_features if bias else None
        self._grad_weight = _zeros(in_features, out_features)
        self._grad_bias = [0.0] * out_features if bias else None
        self._last_input: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        self._last_input = x
        result = [0.0] * self.out_features
        for j in range(self.out_features):
            s = 0.0
            for i in range(self.in_features):
                s += x[i] * self.weight[i][j]
            if self.bias is not None:
                s += self.bias[j]
            result[j] = s
        return result

    def backward(self, grad_output: list[float], lr: float = 0.001) -> list[float]:
        for i in range(self.in_features):
            for j in range(self.out_features):
                self._grad_weight[i][j] = self._last_input[i] * grad_output[j]
                self.weight[i][j] -= lr * self._grad_weight[i][j]
        if self.bias is not None and self._grad_bias is not None:
            for j in range(self.out_features):
                self._grad_bias[j] = grad_output[j]
                self.bias[j] -= lr * self._grad_bias[j]
        grad_input = [0.0] * self.in_features
        for i in range(self.in_features):
            for j in range(self.out_features):
                grad_input[i] += grad_output[j] * self.weight[i][j]
        return grad_input


class Embedding:
    """Embedding con soporte backward."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = _randn(vocab_size, embed_dim)
        self._last_ids: list[int] = []

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        self._last_ids = token_ids
        return [self.weight[tid][:] for tid in token_ids]

    def backward(self, grad_output: list[list[float]], lr: float = 0.001) -> None:
        for idx, tid in enumerate(self._last_ids):
            if tid < self.vocab_size:
                for j in range(self.embed_dim):
                    self.weight[tid][j] -= lr * grad_output[idx][j]


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

        Q = [self.q_proj.forward(xi) for xi in x]
        K = [self.k_proj.forward(xi) for xi in x]
        V = [self.v_proj.forward(xi) for xi in x]

        if use_cache:
            if not self._cache_k:
                self._cache_k = K[:]
                self._cache_v = V[:]
            else:
                self._cache_k.extend(K)
                self._cache_v.extend(V)
            self._cache_pos += seq_len
            K_all = self._cache_k
            V_all = self._cache_v
        else:
            K_all = K
            V_all = V

        def to_heads(arr: list[list[float]]) -> list[list[list[float]]]:
            heads = []
            for i in range(len(arr)):
                h = []
                for hd in range(self.num_heads):
                    start = hd * self.head_dim
                    h.append(arr[i][start:start + self.head_dim])
                heads.append(h)
            return heads

        Q_h = to_heads(Q)
        K_h = to_heads(K_all)
        V_h = to_heads(V_all)

        output_heads = []
        attn_probs = []
        scale = math.sqrt(self.head_dim)
        k_len = len(K_all)

        for hd in range(self.num_heads):
            head_out = []
            head_probs = []
            for i in range(seq_len):
                scores = []
                for j in range(k_len):
                    s = sum(Q_h[i][hd][d] * K_h[j][hd][d] for d in range(self.head_dim)) / scale
                    if use_cache:
                        real_i = k_len - seq_len + i
                        if j > real_i:
                            s = -1e9
                    else:
                        if j > i:
                            s = -1e9
                    scores.append(s)
                attn = _softmax(scores)
                head_probs.append(attn)
                out = [0.0] * self.head_dim
                for j in range(k_len):
                    for d in range(self.head_dim):
                        out[d] += attn[j] * V_h[j][hd][d]
                head_out.append(out)
            output_heads.append(head_out)
            attn_probs.append(head_probs)

        self._fwd = {
            "Q": Q,
            "K": K_all,
            "V": V_all,
            "probs": attn_probs,
            "proj_in": [xi[:] for xi in x],
            "seq_len": seq_len,
            "k_len": k_len,
        }

        output = []
        out_in = []
        for i in range(seq_len):
            concatenated = []
            for hd in range(self.num_heads):
                concatenated.extend(output_heads[hd][i])
            out_in.append(concatenated)
            output.append(self.out_proj.forward(concatenated))

        self._fwd["out_in"] = out_in

        return output

    def _proj_backward(self, weights, inputs, grads, lr, in_size, out_size):
        """Backprop manual de projections lineales por posición.

        weights: [in][out]; inputs: n x in; grads: n x out.
        Devuelve gradientes de entrada (n x in).
        """
        d_in = [[0.0] * in_size for _ in range(len(grads))]
        for i, g in enumerate(grads):
            xi = inputs[i]
            for f in range(in_size):
                for o in range(out_size):
                    d_in[i][f] += g[o] * weights[f][o]
                    weights[f][o] -= lr * g[o] * xi[f]
        return d_in

    def backward(self, dout: list[list[float]], lr: float = 0.001) -> list[list[float]]:
        """Backprop a través de la atención causal (entrenamiento, sin cache)."""
        fwd = self._fwd
        Q = fwd["Q"]
        K = fwd["K"]
        V = fwd["V"]
        probs = fwd["probs"]  # [heads][seq][k_len]
        seq_len = fwd["seq_len"]
        k_len = fwd["k_len"]
        scale = math.sqrt(self.head_dim)

        # out_proj backward -> gradiente de la concatenación por posición
        dc = self._proj_backward(
            self.out_proj.weight, fwd["out_in"], dout, lr,
            self.embed_dim, self.embed_dim,
        )

        dQ = [[0.0] * self.embed_dim for _ in range(seq_len)]
        dK = [[0.0] * self.embed_dim for _ in range(k_len)]
        dV = [[0.0] * self.embed_dim for _ in range(k_len)]

        for hd in range(self.num_heads):
            s = hd * self.head_dim
            for i in range(seq_len):
                dhi = dc[i][s:s + self.head_dim]
                # dS[j] = dhi . V[j] y dV[j] += p_ij * dhi
                dS = [0.0] * k_len
                for j in range(k_len):
                    pj = probs[hd][i][j]
                    if pj <= 0.0:
                        continue
                    for d in range(self.head_dim):
                        dS[j] += dhi[d] * V[j][s + d]
                        dV[j][s + d] += pj * dhi[d]
                # softmax gradiente: da = p*(dS - dot(dS, p))
                dot_p_dS = 0.0
                for j in range(k_len):
                    dot_p_dS += probs[hd][i][j] * dS[j]
                for j in range(k_len):
                    pj = probs[hd][i][j]
                    if pj <= 0.0:
                        continue
                    da = pj * (dS[j] - dot_p_dS)
                    for d in range(self.head_dim):
                        dK[j][s + d] += da * Q[i][s + d] / scale
                        dQ[i][s + d] += da * K[j][s + d] / scale

        proj_in = fwd["proj_in"]
        dq = self._proj_backward(self.q_proj.weight, proj_in, dQ, lr, self.embed_dim, self.embed_dim)
        dk = self._proj_backward(self.k_proj.weight, proj_in, dK, lr, self.embed_dim, self.embed_dim)
        dv = self._proj_backward(self.v_proj.weight, proj_in, dV, lr, self.embed_dim, self.embed_dim)

        return [
            [dq[i][d] + dk[i][d] + dv[i][d] for d in range(self.embed_dim)]
            for i in range(seq_len)
        ]

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
        self._pre_gelu = []
        self._fc2_in = []
        result = []
        for xi in x:
            h = self.fc1.forward(xi)
            self._pre_gelu.append(h)
            gelu_h = [_gelu(v) for v in h]
            self._fc2_in.append(gelu_h)
            result.append(self.fc2.forward(gelu_h))
        return result

    def backward(self, dout: list[list[float]], lr: float = 0.001) -> list[list[float]]:
        """Backprop: fc2 -> GELU -> fc1 (bucles por posición)."""
        n = len(dout)
        fc2_in = self.fc2.in_features      # hidden
        fc2_out = self.fc2.out_features    # embed_dim
        w2 = self.fc2.weight               # [hidden][embed_dim]

        # fc2 backward
        dx_gelu = [[0.0] * fc2_in for _ in range(n)]
        for i in range(n):
            for o in range(fc2_out):
                for h in range(fc2_in):
                    dx_gelu[i][h] += dout[i][o] * w2[h][o]
                    w2[h][o] -= lr * dout[i][o] * self._fc2_in[i][h]

        # GELU
        dx1 = [
            [_gelu_deriv(v) * dx_gelu[i][h] for h, v in enumerate(self._pre_gelu[i])]
            for i in range(n)
        ]

        # fc1 backward
        fc1_in = self.fc1.in_features      # embed_dim
        fc1_out = self.fc1.out_features    # hidden
        w1 = self.fc1.weight               # [embed_dim][hidden]
        dx = [[0.0] * fc1_in for _ in range(n)]
        for i in range(n):
            for o in range(fc1_out):
                for f in range(fc1_in):
                    dx[i][f] += dx1[i][o] * w1[f][o]
                    w1[f][o] -= lr * dx1[i][o] * self._fc1_in[i][f]
        return dx

    def clear_cache(self) -> None:
        pass


class TransformerBlock:
    """Bloque Transformer con LayerNorm + Attention + FeedForward + Residuals."""

    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int = 4) -> None:
        self.embed_dim = embed_dim
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_mult)
        self.ln1_gamma = [1.0] * embed_dim
        self.ln1_beta = [0.0] * embed_dim
        self.ln2_gamma = [1.0] * embed_dim
        self.ln2_beta = [0.0] * embed_dim

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
        out = []
        for i, g in enumerate(grad_in):
            dx, dgamma, dbeta = _layer_norm_backward(g, xnorm[i], gamma, stds[i])
            for d in range(self.embed_dim):
                gamma[d] -= lr * dgamma[d]
                beta[d] -= lr * dbeta[d]
            out.append(dx)
        return out

    def clear_cache(self) -> None:
        self.attn.clear_cache()
        self.ff.clear_cache()
