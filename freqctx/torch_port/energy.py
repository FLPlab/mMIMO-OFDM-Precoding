"""Modele d'energie du labo (mMIMO-ART-FPGA, Pipeline/Energy.py), et sa version corrigee.

Formule du labo, par couche (MAC, poids W, activations A, precisions Q_W et Q_A) :
    E_MAC(Q) = 0.857904 (Q / 16)^1.9 ;  E_M = 2 E_MAC ;  E_L = E_MAC
    E = E_MAC (MAC + 3 A) + [E_M W + E_L MAC / sqrt(p_W)] + [2 E_M,A A + E_L,A MAC / sqrt(p_A)],
    sqrt(p) = sqrt(64 Q / 16).
Trois ecarts releves dans Pipeline/Energy.py (2026-11-01) :
  1. les acces memoire (E_M, E_L) sont mis a l'echelle en (Q/16)^1.9 comme un multiplieur, alors
     que lire / ecrire un mot de Q bits coute ~ proportionnellement a Q ;
  2. compute_linear_macs = in x out, sans le nombre de tokens (les activations, elles, le comptent) ;
  3. MultiheadAttention : seq_len = current_size[0] = dimension du lot (1 avec input_size=(1,64,128)).
"corrige" : memoire lineaire en Q (E_M = 2 E_MAC(16) Q/16, E_L = E_MAC(16) Q/16) ; MAC, poids et
activations comptes avec le bon nombre de tokens (compte explicite par couche ci-dessous).
Tout est differentiable en Q (tenseurs), pour la regularisation d'ART.
"""
import math

import torch


def _emac(Q):
    return 0.857904 * (Q / 16) ** 1.9


def layer_energy(mac, w, a, Qw, Qa, corrected=True):
    """Energie (pJ) d'une couche ; Qw, Qa : tenseurs ou nombres."""
    Qw = torch.as_tensor(Qw, dtype=torch.float32)
    Qa = torch.as_tensor(Qa, dtype=torch.float32)
    emac = _emac(Qw)
    if corrected:
        em_w, el_w = 2 * _emac(16.0) * Qw / 16, _emac(16.0) * Qw / 16
        em_a, el_a = 2 * _emac(16.0) * Qa / 16, _emac(16.0) * Qa / 16
    else:
        em_w, el_w = 2 * emac, emac
        em_a, el_a = 2 * _emac(Qa), _emac(Qa)
    sp_w, sp_a = torch.sqrt(64 * Qw / 16), torch.sqrt(64 * Qa / 16)
    ec = emac * (mac + 3 * a)
    ew = em_w * w + el_w * mac / sp_w
    ea = 2 * em_a * a + el_a * mac / sp_a
    return ec + ew + ea


def pair_precoder_layers(model, K):
    """Couches denses du precodeur par paires (nom -> (MAC, poids, activations) pour un echantillon).
    Toutes agissent sur les K^2 tokens de paires."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            T = K * K
            nw = mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0)
            out[name] = (T * mod.in_features * mod.out_features, nw, T * mod.out_features)
    return out


def pair_precoder_fixed(K, M, d, layers):
    """Operations hors couches denses (non quantifiees par ART, comptees a 16 bits) :
    G = h h^H, produit triangulaire, W = h^H C (MAC reelles, complexes = 4 reelles)."""
    mac = 4 * K * K * M + layers * 4 * K ** 3 * d + 4 * M * K * K
    acts = K * K * 2 + layers * K * K * 2 * d + M * K * 2
    return mac, 0, acts


def lab_transformer_layers(model, K):
    """Couches du modele BitAdapt (un token par utilisateur, K tokens), comptees correctement."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            d = mod.embed_dim
            out[name] = (4 * K * d * d + 2 * K * K * d, 4 * d * d + 4 * d, 4 * K * d)
        elif isinstance(mod, torch.nn.Linear) and not isinstance(mod, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            nw = mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0)
            out[name] = (K * mod.in_features * mod.out_features, nw, K * mod.out_features)
    return out


def total_energy(layers, bits, corrected=True, fixed=None):
    """layers : nom -> (MAC, W, A) ; bits : nom -> (Q_W, Q_A). fixed : (MAC, W, A) a 16 bits."""
    e = sum(layer_energy(m, w, a, *bits[n], corrected=corrected) for n, (m, w, a) in layers.items())
    if fixed is not None:
        e = e + layer_energy(*fixed, 16.0, 16.0, corrected=corrected)
    return e
