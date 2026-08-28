"""GPT Language Model with full forward AND backward pass.

Arquitectura GPT-2 simplificada con backpropagation real
para que el modelo realmente aprenda.
"""
import json
import math
import random
from pathlib import Path

from app.ai.neural.transformer.blocks import (
    Embedding,
    Linear,
    TransformerBlock,
    _layer_norm_backward,
    _layer_norm_forward,
)


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

        self.pos_embed_weight = []
        for _ in range(max_seq_len):
            self.pos_embed_weight.append([random.gauss(0, 0.02) for _ in range(embed_dim)])

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_mult)
            for _ in range(num_layers)
        ]

        self.ln_gamma = [1.0] * embed_dim
        self.ln_beta = [0.0] * embed_dim
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

        # Positional embeddings
        start_pos = self._cache_pos if use_cache else 0
        for i in range(seq_len):
            pos = start_pos + i
            if pos < self.max_seq_len:
                for j in range(self.embed_dim):
                    x[i][j] += self.pos_embed_weight[pos][j]

        if use_cache:
            self._cache_pos += seq_len

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

        # 1) Gradiente de cross-entropy + softmax por posición y actualizar lm_head
        d_input = [[0.0] * self.embed_dim for _ in positions]
        total_loss = 0.0
        for n, i in enumerate(positions):
            max_l = max(logits[i])
            exp_l = [math.exp(v - max_l) for v in logits[i]]
            total_exp = sum(exp_l)
            probs = [v / total_exp for v in exp_l]
            total_loss -= math.log(probs[target_ids[i]] + 1e-10)

            grad = [probs[j] for j in range(len(probs))]
            grad[target_ids[i]] -= 1.0

            xin = ln_out[i]
            weight = self.lm_head.weight
            # dW += grad . xin ; dX += grad . W
            for f in range(self.embed_dim):
                wf = weight[f]
                grad_input = 0.0
                for v in range(self.vocab_size):
                    grad_input += grad[v] * wf[v]
                    wf[v] -= lr * grad[v] * xin[f]
                d_input[n][f] = grad_input

        total_loss /= len(positions)

        # 2) LayerNorm final por posición
        dx_final = []
        for n, i in enumerate(positions):
            dx, dgamma, dbeta = _layer_norm_backward(
                d_input[n], ln_xnorm[i], self.ln_gamma, ln_std[i]
            )
            for f in range(self.embed_dim):
                self.ln_gamma[f] -= lr * dgamma[f]
                self.ln_beta[f] -= lr * dbeta[f]
            dx_final.append(dx)

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
            "token_embed": self.token_embed.weight,
            "pos_embed": self.pos_embed_weight,
            "ln_gamma": self.ln_gamma,
            "ln_beta": self.ln_beta,
            "lm_head_weight": self.lm_head.weight,
        }
        block_states = []
        for block in self.blocks:
            bs = {
                "ln1_gamma": block.ln1_gamma,
                "ln1_beta": block.ln1_beta,
                "ln2_gamma": block.ln2_gamma,
                "ln2_beta": block.ln2_beta,
                "attn_q": block.attn.q_proj.weight,
                "attn_k": block.attn.k_proj.weight,
                "attn_v": block.attn.v_proj.weight,
                "attn_out": block.attn.out_proj.weight,
                "ff_fc1": block.ff.fc1.weight,
                "ff_fc2": block.ff.fc2.weight,
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
        self.token_embed.weight = state["token_embed"]
        self.pos_embed_weight = state["pos_embed"]
        self.ln_gamma = state["ln_gamma"]
        self.ln_beta = state["ln_beta"]
        self.lm_head.in_features = self.embed_dim
        self.lm_head.out_features = self.vocab_size
        self.lm_head.weight = state["lm_head_weight"]

        self.blocks = [
            TransformerBlock(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ]
        for i, bs in enumerate(state["blocks"]):
            block = self.blocks[i]
            block.ln1_gamma = bs["ln1_gamma"]
            block.ln1_beta = bs["ln1_beta"]
            block.ln2_gamma = bs["ln2_gamma"]
            block.ln2_beta = bs["ln2_beta"]
            block.attn.q_proj.weight = bs["attn_q"]
            block.attn.k_proj.weight = bs["attn_k"]
            block.attn.v_proj.weight = bs["attn_v"]
            block.attn.out_proj.weight = bs["attn_out"]
            block.ff.fc1.weight = bs["ff_fc1"]
            block.ff.fc2.weight = bs["ff_fc2"]

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
