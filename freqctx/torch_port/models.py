"""Base figee freqctx reecrite en PyTorch (pour la quantification ART et la chaine HLS
du labo, mMIMO-ART-FPGA). Reecriture ligne a ligne de :
    models/delay_domain.py   DelayEstimator (banded, oversample), DelayDomainPrecoder
    models/per_sc.py         Block (attention softmax de l'encodeur)
    models/gram_precoder.py  TriangularBlock (edge_g), GramPrecoder (triangular_g, couches distinctes)
Memes conventions : h [B, F, K, M] complexe, sortie W [B, F, M, K], puissance 1 par SC.
Les poids s'importent depuis Keras (torch_port/export_keras.py -> .npz) ; l'egalite des
sorties est verifiee par torch_port/check_equivalence.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def _real(z):
    return torch.cat([z.real, z.imag], dim=-1)


def _cplx(x):
    d = x.shape[-1] // 2
    return torch.complex(x[..., :d], x[..., d:])


def normalize_power(w):
    p = (w.abs() ** 2).sum(dim=(-2, -1), keepdim=True)
    return w / torch.sqrt(p).to(w.dtype)


class Block(nn.Module):
    """Bloc de l'encodeur : LayerNorm, attention softmax multi-tetes, FFN GELU (2d)."""

    def __init__(self, d, heads, ffn_mult=2):
        super().__init__()
        self.d, self.heads = d, heads
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn1, self.ffn2 = nn.Linear(d, ffn_mult * d), nn.Linear(ffn_mult * d, d)

    def forward(self, x):                                   # x [N, T, d]
        N, T, _ = x.shape
        dh = self.d // self.heads
        q, k, v = (t.reshape(N, T, self.heads, dh).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(self.d, -1))
        a = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5, dim=-1)
        x = x + self.out((a @ v).transpose(1, 2).reshape(N, T, self.d))
        return x + self.ffn2(Fn.gelu(self.ffn1(self.ln2(x))))


class DelayEstimator(nn.Module):
    """Encodeur frequentiel : peigne -> IDFT (grille O fois plus fine) -> 2 blocs d'attention
    entre retards -> filtre complexe sur t-1, t, t+1 -> DFT."""

    def __init__(self, num_sc, num_users, num_bs_ant, d=32, layers=2, heads=2, guard=4, oversample=2):
        super().__init__()
        F, K, M, O = num_sc, num_users, num_bs_ant, oversample
        self.K, self.M = K, M
        N = F // K
        t = np.arange(O * N)
        t_signed = np.where(t < O * N - O * guard, t, t - O * N)
        self.T = T = len(t_signed)
        n = np.arange(N)
        self.register_buffer("pilot_idx", torch.tensor(np.stack([k + K * n for k in range(K)]), dtype=torch.long))
        idft = np.stack([np.exp(2j * np.pi * np.outer(t_signed, k + K * n) / (O * F)) / N for k in range(K)])
        self.register_buffer("idft", torch.tensor(idft, dtype=torch.complex64))          # [K, T, N]
        dft = np.exp(-2j * np.pi * np.outer(np.arange(F), t_signed) / (O * F)) / O
        self.register_buffer("dft", torch.tensor(dft, dtype=torch.complex64))            # [F, T]
        self.embed = nn.Linear(2 * M + 1, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.gain = nn.Linear(d, 6)

    def forward(self, h_ls):                                # [B, F, K, M]
        B, K, T = h_ls.shape[0], self.K, self.T
        hk = h_ls.permute(0, 2, 1, 3)                       # [B, K, F, M]
        y = torch.stack([hk[:, k, self.pilot_idx[k]] for k in range(K)], 1)            # [B, K, N, M]
        g = torch.einsum("ktn,bknm->bktm", self.idft, y)    # [B, K, T, M]
        p = (g.abs() ** 2).mean(-1, keepdim=True)
        feats = torch.cat([g.real, g.imag, torch.log(p + 1e-9) / 10.0], -1)
        x = self.embed(feats.reshape(B * K, T, -1)) + self.pos.weight[None]
        for blk in self.blocks:
            x = blk(x)
        o = self.gain(x).reshape(B, K, T, 6)
        c0 = torch.complex(torch.sigmoid(o[..., 0]), o[..., 1])
        cm, cp = torch.complex(o[..., 2], o[..., 3]), torch.complex(o[..., 4], o[..., 5])
        g_hat = c0[..., None] * g + cm[..., None] * torch.roll(g, 1, 2) + cp[..., None] * torch.roll(g, -1, 2)
        return torch.einsum("ft,bktm->bfkm", self.dft, g_hat)


class TriangularBlock(nn.Module):
    """Attention triangulaire lineaire entre paires, G reinjecte (edge_g)."""

    def __init__(self, d, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d, eps=1e-6)
        self.a = nn.Linear(2 * d + 3, 2 * d, bias=False)
        self.b = nn.Linear(2 * d + 3, 2 * d, bias=False)
        self.f1 = nn.Linear(2 * d + 3 + 2 * d, hidden)
        self.f2 = nn.Linear(hidden, 2 * d)

    def forward(self, e, ef):                               # e [N, K, K, d] complexe, ef [N, K, K, 3] reel
        x = torch.cat([self.norm(_real(e)), ef], -1)
        m = torch.einsum("njlc,nlkc->njkc", _cplx(self.a(x)), _cplx(self.b(x))) / e.shape[1]
        return e + _cplx(self.f2(Fn.gelu(self.f1(torch.cat([x, _real(m)], -1)))))


class PairPrecoder(nn.Module):
    """Transformer par paires d'utilisateurs : W = hp^H C.
    inp (phase 10, dynamique des entrees avec des puissances tres inegales) :
      "raw"  : base figee (G = h h^H / M, aretes [Re G, Im G, |G|]) -- equivalent Keras ;
      "log"  : idem, mais G donnee en module (dB/10) et phase (cos, sin) ;
      "norm" : canal de chaque utilisateur normalise (hp = h_k / ||h_k||, G = correlations, |G| <= 1) ;
               puissance relative de chaque utilisateur (dB/10) donnee a part sur les paires ;
               W = hp^H C reste dans l'espace des canaux."""

    def __init__(self, d=16, hidden=32, layers=8, inp="raw"):
        super().__init__()
        self.inp = inp
        self.init = nn.Linear(7 if inp == "norm" else 5, 2 * d)
        self.blocks = nn.ModuleList([TriangularBlock(d, hidden) for _ in range(layers)])
        self.read = nn.Linear(2 * d, 2)

    def prepare(self, h):
        """Renvoie (hp, extra) : vecteurs utilises pour G et W, et caracteristiques par paire en plus."""
        if self.inp == "norm":
            n2 = (h.abs() ** 2).sum(-1)                                   # [N, K]
            hp = h / torch.sqrt(n2)[..., None].to(h.dtype)
            pdb = 10 * torch.log10(n2 / n2.mean(-1, keepdim=True)) / 10.0  # puissance relative, dB/10
            K = h.shape[1]
            extra = torch.stack([pdb[:, :, None].expand(-1, -1, K), pdb[:, None, :].expand(-1, K, -1)], -1)
            return hp, extra
        return h, None

    def gram(self, hp):
        return hp @ hp.conj().transpose(-1, -2) / (1 if self.inp == "norm" else hp.shape[2])

    def edges(self, g):
        if self.inp == "log":
            a = g.abs()
            ph = g / a.clamp_min(1e-12)
            return torch.stack([ph.real, ph.imag, 10 * torch.log10(a.clamp_min(1e-6)) / 10.0], -1)
        return torch.stack([g.real, g.imag, g.abs()], -1)

    def coeffs_from(self, g, snr_db, extra):
        K = g.shape[1]
        ef = self.edges(g)
        eye = torch.eye(K, device=g.device).expand(g.shape)
        snr = (snr_db / 10.0)[:, None, None].expand(g.shape)
        parts = [ef, eye[..., None], snr[..., None]] + ([extra] if extra is not None else [])
        e = _cplx(self.init(torch.cat(parts, -1)))
        for blk in self.blocks:
            e = blk(e, ef)
        return _cplx(self.read(_real(e)))[..., 0]                          # [N, K, K]

    def coeffs(self, h, snr_db):
        hp, extra = self.prepare(h)
        return self.coeffs_from(self.gram(hp), snr_db, extra)

    def forward(self, h, snr_db):
        hp, extra = self.prepare(h)
        c = self.coeffs_from(self.gram(hp), snr_db, extra)
        return normalize_power(hp.conj().transpose(-1, -2) @ c)          # [N, M, K]


class FreqctxModel(nn.Module):
    """Base figee : encodeur frequentiel + transformer par paires, applique a chaque SC."""

    def __init__(self, num_sc=612, num_users=4, num_bs_ant=8, est_d=32, d=16, hidden=32, layers=8):
        super().__init__()
        self.F, self.K, self.M = num_sc, num_users, num_bs_ant
        self.estimator = DelayEstimator(num_sc, num_users, num_bs_ant, d=est_d)
        self.precoder = PairPrecoder(d, hidden, layers)

    def forward(self, h_ls, snr_db):                        # [B, F, K, M], [B]
        B = h_ls.shape[0]
        h = self.estimator(h_ls)
        w = self.precoder(h.reshape(B * self.F, self.K, self.M), snr_db.repeat_interleave(self.F))
        return w.reshape(B, self.F, self.M, self.K)


def load_keras_npz(model, path):
    """Charge les poids exportes par export_keras.py (noms explicites, noyaux Keras [in, out])."""
    z = np.load(path)
    sd = {}
    for k in z.files:
        v = torch.tensor(z[k])
        sd[k] = v.T.contiguous() if k.endswith(".weight") and v.ndim == 2 and not k.endswith("pos.weight") else v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith(("pilot_idx", "idft", "dft"))]
    assert not missing and not unexpected, (missing, unexpected)
    return model
