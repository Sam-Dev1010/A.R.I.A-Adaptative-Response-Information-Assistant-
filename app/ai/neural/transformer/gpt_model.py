"""GPT Language Model with full forward AND backward pass.

Arquitectura GPT-2 simplificada con backpropagation real
para que el modelo realmente aprenda.
"""
import json
import math
import random
from pathlib import Path

import numpy as np

from app.ai.neural.transformer.blocks import (
    Embedding,
    Linear,
    TransformerBlock,
    _layer_norm_backward,
    _layer_norm_forward,
)


def _to_list(w):
    """Convierte ndarray o lista a lista Python (para JSON)."""
    return w.tolist() if isinstance(w, np.ndarray) else w


class GPTModel:
    """Modelo de lenguaje GPT-2 con backpropagation completo."""

    def __init__(
        self,
        vocab_size: int = 8000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 512,
        ff_mult: int = 4,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embed = Embedding(vocab_size, embed_dim)

        rng = np.random.default_rng()
        self.pos_embed_weight = rng.normal(0.0, 0.02, size=(max_seq_len, embed_dim)).astype(np.float64)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_mult)
            for _ in range(num_layers)
        ]

        self.ln_gamma = np.ones(embed_dim, dtype=np.float64)
        self.ln_beta = np.zeros(embed_dim, dtype=np.float64)
        self.lm_head = Linear(embed_dim, vocab_size)

        self._cache_pos = 0
        self._forward_cache: dict[str, any] = {}

    def forward(self, token_ids: list[int], use_cache: bool = False) -> list[list[float]]:
        seq_len = len(token_ids)
        if seq_len > self.max_seq_len:
            token_ids = token_ids[-self.max_seq_len:]
            seq_len = self.max_seq_len

        # Token embeddings
        x = self.token_embed.forward(token_ids)

        # Positional embeddings (vectorizado con NumPy)
        start_pos = self._cache_pos if use_cache else 0
        xa = np.asarray(x, dtype=np.float64)
        pos_weights = np.asarray(self.pos_embed_weight, dtype=np.float64)
        pos_idx = start_pos + np.arange(seq_len)
        # El forward original solo suma la embedding posicional si pos < max_seq_len
        valid = pos_idx < self.max_seq_len
        if valid.all():
            xa += pos_weights[pos_idx]
        else:
            xa[valid] += pos_weights[pos_idx[valid]]

        if use_cache:
            self._cache_pos += seq_len

        x = xa.tolist()

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x, use_cache=use_cache)

        # Final layer norm
        self._forward_cache["pre_ln"] = [row[:] for row in x]
        x_normed = []
        self._forward_cache["ln_xnorm"] = []
        self._forward_cache["ln_std"] = []
        self._forward_cache["ln_out"] = []
        for xi in x:
            out, xnorm, _mean, std = _layer_norm_forward(xi, self.ln_gamma, self.ln_beta)
            x_normed.append(out)
            self._forward_cache["ln_xnorm"].append(xnorm)
            self._forward_cache["ln_std"].append(std)
            self._forward_cache["ln_out"].append(out)

        # LM head → logits
        logits = [self.lm_head.forward(xi) for xi in x_normed]
        self._forward_cache["logits"] = logits
        self._forward_cache["token_ids"] = token_ids

        return logits

    def backward(self, target_ids: list[int], lr: float = 3e-4) -> float:
        """Backward pass completo (backprop real a través del transformer).

        Calcula el gradiente de cross-entropy sobre las posiciones válidas y lo
        propaga: lm_head -> LN final -> bloques transformer -> embeddings.
        """
        logits = self._forward_cache.get("logits", [])
        ln_out = self._forward_cache.get("ln_out", [])
        ln_xnorm = self._forward_cache.get("ln_xnorm", [])
        ln_std = self._forward_cache.get("ln_std", [])

        if not logits or not ln_out:
            return 0.0

        positions = [i for i in range(len(target_ids)) if i < len(logits)]
        if not positions:
            return 0.0

        # 1) Cross-entropy + softmax vectorizado; actualización SGD correcta del lm_head
        #    (primero se calcula TODO el gradiente con los pesos originales y luego se
        #     actualizan una sola vez, sin mezclar propagación con descenso).
        Log = np.asarray([logits[i] for i in positions], dtype=np.float64)      # [P, vocab]
        max_l = Log.max(axis=1, keepdims=True)
        exp_l = np.exp(Log - max_l)
        Pmat = exp_l / exp_l.sum(axis=1, keepdims=True)                          # [P, vocab]
        tgt = np.asarray(target_ids, dtype=np.int64)[positions]                 # [P]
        row_idx = np.arange(Pmat.shape[0])
        total_loss = -np.log(Pmat[row_idx, tgt] + 1e-10).mean()

        Grad = Pmat.copy()                                                       # [P, vocab]
        Grad[row_idx, tgt] -= 1.0
        Grad /= Pmat.shape[0]                                                    # gradiente de la loss media

        Xin = np.asarray([ln_out[i] for i in positions], dtype=np.float64)       # [P, embed]
        d_input = Grad @ self.lm_head.weight.T                                   # [P, embed]
        self.lm_head.weight -= lr * (Xin.T @ Grad)                               # [embed, vocab]

        # 2) LayerNorm final por posición (vectorizado), actualización SGD una sola vez
        Xnorm = np.asarray([ln_xnorm[i] for i in positions], dtype=np.float64)   # [P, embed]
        stds = np.asarray([ln_std[i] for i in positions], dtype=np.float64)      # [P]
        gamma = self.ln_gamma
        dGrad_norm = d_input * gamma[None, :]
        dvar = (dGrad_norm * Xnorm).sum(axis=1) * (-0.5) / (stds ** 3)           # [P]
        dmean = dGrad_norm.sum(axis=1) * (-1.0) / stds + dvar * (-2.0 * Xnorm).sum(axis=1) / self.embed_dim
        dx_final = dGrad_norm / stds[:, None] + (dvar[:, None] * 2.0 * Xnorm) / self.embed_dim + dmean[:, None] / self.embed_dim
        dgamma = (d_input * Xnorm).sum(axis=0)                                   # [embed]
        dbeta = d_input.sum(axis=0)                                              # [embed]
        self.ln_gamma -= lr * dgamma
        self.ln_beta -= lr * dbeta
        dx_final = dx_final.tolist()

        # 3) Bloques transformer en orden inverso (solo entrenamiento, sin cache)
        for block in reversed(self.blocks):
            dx_final = block.backward(dx_final, lr=lr)

        # 4) Embedding de tokens
        self.token_embed.backward(dx_final, lr=lr)

        return total_loss

    def predict_next(self, token_ids: list[int]) -> tuple[int, float]:
        logits = self.forward(token_ids)
        last_logits = logits[-1]
        max_l = max(last_logits)
        exp_l = [math.exp(v - max_l) for v in last_logits]
        total = sum(exp_l)
        probs = [v / total for v in exp_l]
        best_id = max(range(len(probs)), key=lambda i: probs[i])
        return best_id, probs[best_id]

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        vocab_mask_size: int = 0,
        repetition_penalty: float = 1.5,
        min_new_tokens: int = 0,
    ) -> list[int]:
        """Genera tokens; si vocab_mask_size > 0 solo se muestrean tokens reales.

        El llamador debe pasar el tamaño del vocabulario real del tokenizer para
        que el modelo no genere IDs de tokens inexistentes (<unk>).
        """
        generated = list(prompt_ids)
        self.clear_cache()
        self.forward(generated, use_cache=True)

        # Penalización por repetición para evitar bucles (ej: "eeee...")
        log_rep_pen = math.log(repetition_penalty) if repetition_penalty > 1.0 else 0.0
        seen: dict[int, int] = {}
        for tid in generated:
            seen[tid] = seen.get(tid, 0) + 1

        for _ in range(max_new_tokens):
            logits = self.forward(generated[-1:], use_cache=True)
            next_logits = logits[-1]

            # Restringir a tokens que existen en el vocabulario real
            if vocab_mask_size and 0 < vocab_mask_size < len(next_logits):
                for i in range(vocab_mask_size, len(next_logits)):
                    next_logits[i] = float("-inf")
                if vocab_mask_size > 1:
                    # <unk>, <bos>, <user>, <assistant>: tokens de control, no se emiten
                    for ctl_id in (1, 2, 5, 6):
                        if ctl_id < vocab_mask_size:
                            next_logits[ctl_id] = float("-inf")

            if temperature > 0:
                next_logits = [v / temperature for v in next_logits]

            if top_k > 0:
                indexed = sorted(enumerate(next_logits), key=lambda x: -x[1])
                top_k_ids = [idx for idx, _ in indexed[:top_k]]
                next_logits = [next_logits[i] if i in top_k_ids else float("-inf") for i in range(len(next_logits))]

            if top_p < 1.0:
                sorted_indices = sorted(range(len(next_logits)), key=lambda i: -next_logits[i])
                cumulative = 0.0
                cutoff = float("-inf")
                for idx in sorted_indices:
                    val = math.exp(next_logits[idx]) if next_logits[idx] > -1e10 else 0.0
                    cumulative += val
                    if cumulative >= top_p:
                        cutoff = next_logits[idx]
                        break
                next_logits = [v if v >= cutoff else float("-inf") for v in next_logits]

            # Penalizar tokens ya generados (anti-loop)
            if log_rep_pen > 0.0 and seen:
                for tid, count in seen.items():
                    if tid < len(next_logits) and next_logits[tid] > -1e10:
                        next_logits[tid] -= log_rep_pen * count

            if not any(v > -1e10 for v in next_logits):
                break
            max_l = max(v for v in next_logits if v > -1e10)
            exp_l = [math.exp(v - max_l) if v > -1e10 else 0.0 for v in next_logits]
            total = sum(exp_l)
            if total == 0:
                break
            probs = [v / total for v in exp_l]

            r = random.random()
            cumulative = 0.0
            next_id = 0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    next_id = i
                    break

            generated.append(next_id)
            seen[next_id] = seen.get(next_id, 0) + 1
            n_generated = len(generated) - len(prompt_ids)
            if next_id in (0, 3) and n_generated >= min_new_tokens:
                break

        return generated

    def clear_cache(self) -> None:
        self._cache_pos = 0
        for block in self.blocks:
            block.clear_cache()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_seq_len": self.max_seq_len,
            },
            "token_embed": _to_list(self.token_embed.weight),
            "pos_embed": _to_list(self.pos_embed_weight),
            "ln_gamma": _to_list(self.ln_gamma),
            "ln_beta": _to_list(self.ln_beta),
            "lm_head_weight": _to_list(self.lm_head.weight),
        }
        block_states = []
        for block in self.blocks:
            bs = {
                "ln1_gamma": _to_list(block.ln1_gamma),
                "ln1_beta": _to_list(block.ln1_beta),
                "ln2_gamma": _to_list(block.ln2_gamma),
                "ln2_beta": _to_list(block.ln2_beta),
                "attn_q": _to_list(block.attn.q_proj.weight),
                "attn_k": _to_list(block.attn.k_proj.weight),
                "attn_v": _to_list(block.attn.v_proj.weight),
                "attn_out": _to_list(block.attn.out_proj.weight),
                "ff_fc1": _to_list(block.ff.fc1.weight),
                "ff_fc2": _to_list(block.ff.fc2.weight),
            }
            block_states.append(bs)
        state["blocks"] = block_states
        (path / "model.json").write_text(json.dumps(state, ensure_ascii=False))

    def load(self, path: Path | str) -> None:
        path = Path(path)
        state = json.loads((path / "model.json").read_text())
        config = state["config"]
        self.vocab_size = config["vocab_size"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.max_seq_len = config["max_seq_len"]

        self.token_embed.vocab_size = self.vocab_size
        self.token_embed.embed_dim = self.embed_dim
        self.token_embed.weight = np.asarray(state["token_embed"], dtype=np.float64)
        self.pos_embed_weight = np.asarray(state["pos_embed"], dtype=np.float64)
        self.ln_gamma = np.asarray(state["ln_gamma"], dtype=np.float64)
        self.ln_beta = np.asarray(state["ln_beta"], dtype=np.float64)
        self.lm_head.in_features = self.embed_dim
        self.lm_head.out_features = self.vocab_size
        self.lm_head.weight = np.asarray(state["lm_head_weight"], dtype=np.float64)

        self.blocks = [
            TransformerBlock(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ]
        for i, bs in enumerate(state["blocks"]):
            block = self.blocks[i]
            block.ln1_gamma = np.asarray(bs["ln1_gamma"], dtype=np.float64)
            block.ln1_beta = np.asarray(bs["ln1_beta"], dtype=np.float64)
            block.ln2_gamma = np.asarray(bs["ln2_gamma"], dtype=np.float64)
            block.ln2_beta = np.asarray(bs["ln2_beta"], dtype=np.float64)
            block.attn.q_proj.weight = np.asarray(bs["attn_q"], dtype=np.float64)
            block.attn.k_proj.weight = np.asarray(bs["attn_k"], dtype=np.float64)
            block.attn.v_proj.weight = np.asarray(bs["attn_v"], dtype=np.float64)
            block.attn.out_proj.weight = np.asarray(bs["attn_out"], dtype=np.float64)
            block.ff.fc1.weight = np.asarray(bs["ff_fc1"], dtype=np.float64)
            block.ff.fc2.weight = np.asarray(bs["ff_fc2"], dtype=np.float64)

    def count_params(self) -> int:
        params = 0
        params += self.vocab_size * self.embed_dim
        params += self.max_seq_len * self.embed_dim
        params += self.embed_dim * 2
        params += self.embed_dim * self.vocab_size
        for _ in self.blocks:
            params += self.embed_dim * self.embed_dim * 4
            params += self.embed_dim * (self.embed_dim * 4) * 2
            params += self.embed_dim * 4
        return params
