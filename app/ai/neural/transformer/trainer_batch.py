"""Entrenamiento por lotes (mini-batch) del GPT con NumPy puro.

Camino alternativo SOLO para entrenamiento: procesa B secuencias a la vez con
arreglos np [B, seq, embed] en vez de listas por secuencia, lo que amortiza el
overhead de NumPy/Python y hace el entrenamiento mucho más rápido.

Comparte los MISMOS pesos del GPTModel (muta sus ndarrays), así que el
guardado/carga y el camino por secuencia (inferencia/generación) siguen
funcionando igual. Las secuencias se rellenan (pad) a la longitud máxima del
lote y las posiciones inválidas no contribuyen a la pérdida ni a los gradientes.
"""
import math

import numpy as np

from app.ai.neural.transformer.blocks import _gelu_np, _gelu_deriv_np


def _attend(Q_h, K_h, V_h, scale):
    """Atención causal vectorizada. Los heads son slices contiguos de cada
    posición: Q_h [B,L,H,d] (posición-mayor, igual que el camino por secuencia).
    -> (probs [B,L,H,L], out [B,L,H,d])"""
    # Calcula en layout cabeza-mayor [B,H,L,L] con matmul 4D (varios x más rápido
    # que einsum para estas dimensiones).
    Qc = np.ascontiguousarray(Q_h.transpose(0, 2, 1, 3))   # [B,H,L,d]
    Kc = np.ascontiguousarray(K_h.transpose(0, 2, 1, 3))
    scores = np.matmul(Qc, Kc.transpose(0, 1, 3, 2)) / scale   # [B,H,L,L]
    L = Q_h.shape[1]
    mask = np.arange(L)[None, None, :, None] < np.arange(L)[None, None, None, :]
    scores = scores - 1e9 * mask  # j > i (futuro) enmascarado
    maxs = scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(scores - maxs)
    probs_h = exp_s / exp_s.sum(axis=-1, keepdims=True)
    out_h = np.matmul(probs_h, np.ascontiguousarray(V_h.transpose(0, 2, 1, 3)))
    probs = probs_h.transpose(0, 2, 1, 3)   # [B,L,H,L]
    out = out_h.transpose(0, 2, 1, 3)       # [B,L,H,d]
    return probs, out


def _attend_backward(dout_h, Q_h, K_h, V_h, probs, scale):
    dout3 = np.ascontiguousarray(dout_h.transpose(0, 2, 1, 3))  # [B,H,L,d]
    Q3 = np.ascontiguousarray(Q_h.transpose(0, 2, 1, 3))
    K3 = np.ascontiguousarray(K_h.transpose(0, 2, 1, 3))
    V3 = np.ascontiguousarray(V_h.transpose(0, 2, 1, 3))
    probs_h = np.ascontiguousarray(probs.transpose(0, 2, 1, 3))  # [B,H,L,L]
    # dS[b,h,i,k] = sum_d dout[b,i,h,d]*V[b,k,h,d]
    dSc = np.matmul(dout3, V3.transpose(0, 1, 3, 2))
    # gradiente del softmax: da = P*(dS - sum_k P*dS)
    dot = (probs_h * dSc).sum(axis=-1, keepdims=True)
    da_h = probs_h * (dSc - dot)                       # [B,H,L,L]
    # V: einsum 'blhk,blhd->bkhd' (contrato la fila i con probs)
    dV_h = np.matmul(probs_h.transpose(0, 1, 3, 2), dout3).transpose(0, 2, 1, 3)
    dK_h = np.matmul(da_h.transpose(0, 1, 3, 2), Q3).transpose(0, 2, 1, 3) / scale
    dQ_h = np.matmul(da_h, K3).transpose(0, 2, 1, 3) / scale
    return dQ_h, dK_h, dV_h


def _ln_forward(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xnorm = (x - mean) / std
    out = xnorm * gamma[None, None, :] + beta[None, None, :]
    return out, xnorm, std


def _ln_backward(grad, xnorm, std, gamma):
    g = grad
    dnorm = g * gamma[None, None, :]
    st = std[..., 0]
    dvar = (dnorm * xnorm).sum(axis=-1) * (-0.5) / (st ** 3)
    dmean = dnorm.sum(axis=-1) * (-1.0) / st + (
        dvar * (-2.0 * xnorm).sum(axis=-1) / xnorm.shape[-1]
    )
    dgamma = (g * xnorm).sum(axis=(0, 1))
    dbeta = g.sum(axis=(0, 1))
    dx = (
        dnorm / st[..., None]
        + (dvar[..., None] * 2.0 * xnorm) / xnorm.shape[-1]
        + dmean[..., None] / xnorm.shape[-1]
    )
    return dx, dgamma, dbeta


def forward_batched(model, tokens):
    """Forward por lotes. Devuelve (logits, cache). La cache incluye un
    bloque de entradas por cada capa del transformer para el backward."""
    B, L = tokens.shape
    embed = model.embed_dim
    x = model.token_embed.weight[tokens]                      # [B,L,embed]
    x = x + model.pos_embed_weight[:L][None, :, :]          # posiciones 0..L-1

    block_caches = []
    for block in model.blocks:
        b = {}
        # LN1 -> attention -> residual
        normed1, xn1, st1 = _ln_forward(x, block.ln1_gamma, block.ln1_beta)
        b["ln1_xnorm"], b["ln1_std"], b["attn_in"] = xn1, st1, normed1

        xa = normed1
        Wq = block.attn.q_proj.weight; Wk = block.attn.k_proj.weight
        Wv = block.attn.v_proj.weight; Wo = block.attn.out_proj.weight
        Q = xa @ Wq; K = xa @ Wk; V = xa @ Wv                 # [B,L,embed]
        scale = math.sqrt(block.attn.head_dim)
        Hd = block.attn.num_heads
        eh = block.attn.head_dim
        Q_h = Q.reshape(B, L, Hd, eh)
        K_h = K.reshape(B, L, Hd, eh)
        V_h = V.reshape(B, L, Hd, eh)
        probs, attn_out_h = _attend(Q_h, K_h, V_h, scale)
        attn_out = attn_out_h.reshape(B, L, embed)
        out_proj = attn_out @ Wo
        b["attn"] = (Q, K, V, probs, Q_h, K_h, V_h, scale)
        b["attn_out_in"] = attn_out
        x = x + out_proj

        # LN2 -> FF -> residual
        normed2, xn2, st2 = _ln_forward(x, block.ln2_gamma, block.ln2_beta)
        b["ln2_xnorm"], b["ln2_std"], b["ff_in"] = xn2, st2, normed2
        W1 = block.ff.fc1.weight; W2 = block.ff.fc2.weight
        pre = normed2 @ W1
        gelu_h = _gelu_np(pre)
        ff_out = gelu_h @ W2
        b["ff_pre"], b["ff_gelu"] = pre, gelu_h
        x = x + ff_out
        block_caches.append(b)

    # LayerNorm final
    out, xn, st = _ln_forward(x, model.ln_gamma, model.ln_beta)
    cache = {
        "tokens": tokens,
        "pre_ln": x,
        "ln_out": out,
        "ln_xnorm": xn,
        "ln_std": st,
        "logits": out @ model.lm_head.weight,
        "blocks": block_caches,
    }
    return cache["logits"], cache


def backward_batched(model, targets, valid, cache, lr):
    """Backward por lotes de loss = media por secuencia de su CE media (pos. válidas).

    `targets`: np.int64 [B, L]. `valid`: np.bool [B, L].
    Devuelve la loss media (media de las pérdidas por secuencia).
    """
    B, L, V = cache["logits"].shape
    logits = cache["logits"]
    embed = model.embed_dim
    Hd = model.blocks[0].attn.num_heads
    eh = model.blocks[0].attn.head_dim
    tgt = np.where(valid, targets, 0)
    counts = valid.sum(axis=-1, dtype=np.float64)            # [B]
    maxs = logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits - maxs)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    row_b = np.arange(B)[:, None]
    row_l = np.arange(L)[None, :]
    cols = tgt
    lps = -np.log(probs[row_b, row_l, cols] + 1e-10)
    lps = np.where(valid, lps, 0.0)
    losses = (lps.sum(axis=-1) / counts)                     # per-seq mean
    loss = losses.mean()

    # gradiente de la loss media: d/d(logits) = (probs - onehot) * (1/(B*counts))
    onehot = np.zeros_like(probs)
    onehot[row_b, row_l, cols] = 1.0
    grad = probs - onehot
    weight = (1.0 / (B * counts))[:, None, None]
    grad = grad * np.where(valid[:, :, None], weight, 0.0)

    # LM head
    out = cache["ln_out"]
    dW = np.matmul(out.reshape(-1, embed).T, grad.reshape(-1, V))
    d_input = np.matmul(grad, model.lm_head.weight.T)         # [B,L,embed]
    model.lm_head.weight -= lr * dW

    # LayerNorm final
    x, dgamma, dbeta = _ln_backward(d_input, cache["ln_xnorm"], cache["ln_std"],
                                    model.ln_gamma)
    model.ln_gamma -= lr * dgamma
    model.ln_beta -= lr * dbeta

    # Bloques transformer en orden inverso
    for bi in reversed(range(len(model.blocks))):
        block = model.blocks[bi]
        b = cache["blocks"][bi]

        # FF branch
        hidden = block.ff.fc2.weight.shape[0]
        dff = x @ block.ff.fc2.weight.T
        block.ff.fc2.weight -= lr * np.matmul(
            b["ff_gelu"].reshape(-1, hidden).T, x.reshape(-1, embed))
        dx1 = _gelu_deriv_np(b["ff_pre"]) * dff
        dx_ff = dx1 @ block.ff.fc1.weight.T
        block.ff.fc1.weight -= lr * np.matmul(
            b["ff_in"].reshape(-1, embed).T, dx1.reshape(-1, hidden))
        d2, dg2, db2 = _ln_backward(dx_ff, b["ln2_xnorm"], b["ln2_std"],
                                    block.ln2_gamma)
        block.ln2_gamma -= lr * dg2
        block.ln2_beta -= lr * db2
        d_res = x + d2

        # Attention branch
        Qh, Kh, Vh = b["attn"][4], b["attn"][5], b["attn"][6]
        scale = b["attn"][7]
        dc = d_res @ block.attn.out_proj.weight
        block.attn.out_proj.weight -= lr * np.matmul(
            b["attn_out_in"].reshape(-1, embed).T, d_res.reshape(-1, embed))
        dc_h = dc.reshape(B, L, Hd, eh)
        dQ_h, dK_h, dV_h = _attend_backward(dc_h, Qh, Kh, Vh, b["attn"][3], scale)
        dQ = dQ_h.reshape(B, L, embed)
        dK = dK_h.reshape(B, L, embed)
        dV = dV_h.reshape(B, L, embed)
        d_attn = (dQ @ block.attn.q_proj.weight
                  + dK @ block.attn.k_proj.weight
                  + dV @ block.attn.v_proj.weight)
        proj_in = b["attn_in"]
        block.attn.q_proj.weight -= lr * np.matmul(
            proj_in.reshape(-1, embed).T, dQ.reshape(-1, embed))
        block.attn.k_proj.weight -= lr * np.matmul(
            proj_in.reshape(-1, embed).T, dK.reshape(-1, embed))
        block.attn.v_proj.weight -= lr * np.matmul(
            proj_in.reshape(-1, embed).T, dV.reshape(-1, embed))
        d1, dg1, db1 = _ln_backward(d_attn, b["ln1_xnorm"], b["ln1_std"],
                                    block.ln1_gamma)
        block.ln1_gamma -= lr * dg1
        block.ln1_beta -= lr * db1
        x = d_res + d1

    # Embeddings de tokens: scatter de gradientes (igual que el camino por
    # secuencia). Las posiciones se mantienen congeladas como hace el per-seq.
    valid_flat = valid.reshape(-1)
    if valid_flat.all():
        xflat = x.reshape(-1, embed)
    else:
        xflat = np.where(valid_flat[:, None], x.reshape(-1, embed), 0.0)
    np.subtract.at(model.token_embed.weight, cache["tokens"].reshape(-1), lr * xflat)

    return float(loss)