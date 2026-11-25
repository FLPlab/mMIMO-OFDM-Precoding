"""Generateur du projet HLS du precodeur par paires quantifie, a partir de la reference entiere (hls/intref.py) :
memes constantes, memes operations, simulation C identique bit a bit.

Une petite classe par bloc du reseau, chacune ecrit ses constantes (weights.h) et son C++ (precoder.cpp) :
  MathGen       rnd_shift, quantification, racine, inverse de racine (Newton), log2, GELU, tables ;
  DenseGen      une couche dense (struct de constantes + appel dense<P> deroule ou dense_rt partage) ;
  QuantGen      un quantificateur d'operande de produit ;
  LayerNormGen  normalisation de couche ;
  NormalizeGen  normalisation par utilisateur (entree en virgule flottante par bloc) et puissances en dB ;
  GramGen       G = h h^H (10 paires, symetrie hermitienne) et |G| ;
  InitGen       couche d'entree sur les 7 caracteristiques de paire ;
  ProjGen       etage 1 d'une couche triangulaire : LayerNorm, projections a et b, quantification ;
  FfnGen        etage 2 : produit triangulaire, FFN (f1, GELU, f2), residuel ;
  ReadGen       couche de lecture -> C quantifie ;
  CombineGen    W = h^H C et normalisation de puissance ;
  Project       assemble le tout selon l'architecture (--arch) : seq, flow, stream.
Les etages sont des processus relies par des hls::stream (un element par token, par groupe d'antennes ou par
sous-porteuse), chacun une boucle aplatie (sous-porteuse, token) a II = 1 : en DATAFLOW sur N_SC sous-porteuses
(stream : 64), le flot ne se vide jamais entre deux sous-porteuses.

Largeurs : operandes quantifies sur b + 1 bits, accumulateurs dimensionnes par bornes exactes, valeurs
intermediaires (unites 2^-F) dimensionnees sur les maxima releves par la reference (marge de 2 bits).

    python -m hls.codegen --arch stream --out /users/sala/Documents/test_projet/hls_work/l2_stream [--stats-n 5000]
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np

from hls.intref import F, LOG10_2, LOG2_TAB, RCP, RS_TAB, IntPairPrecoder, block_float, to_fix
from hls.model import QuantPairPrecoder


# ------------------------------------------------------------------ architectures ------------------------------------


# ------------------------------------------------------------------ architectures ------------------------------------
@dataclass
class Arch:
    """dataflow : etages en processus DATAFLOW relies par des streams (sinon appels sequentiels) ; unrolled : couches
    denses deroulees (poids constants), boucles internes deroulees, boucle (sous-porteuse, token) a II = 1 ;
    ant : antennes par cycle (normalisation, G, W) ; nsc : sous-porteuses par appel (flot continu)."""
    name: str
    dataflow: bool
    unrolled: bool
    ant: int
    nsc: int

    @property
    def const(self):                       # poids en constantes (couches deroulees) ou en ROM (couche partagee)
        return self.unrolled

    def pipe(self):
        return "#pragma HLS PIPELINE II=1\n" if self.unrolled else ""

    def unroll(self):
        """Boucles internes : deroulees, ou seulement pipelinees (seq, boucle des tokens non pipelinee)."""
        return "#pragma HLS UNROLL\n" if self.unrolled else "#pragma HLS PIPELINE II=1\n"

    def stage(self):
        """Etages en processus separes (DATAFLOW) ou inlines dans le haut niveau (seq)."""
        return "#pragma HLS INLINE off\n" if self.dataflow else "#pragma HLS INLINE\n"

    def partition(self, var, dim, factor=None):
        """Partition complete (factor None) ou cyclique (rien si factor = 1)."""
        if not self.unrolled or factor == 1:
            return ""
        if factor is None:
            return f"#pragma HLS ARRAY_PARTITION variable={var} complete dim={dim}\n"
        return f"#pragma HLS ARRAY_PARTITION variable={var} cyclic factor={factor} dim={dim}\n"

    def depth(self, n):
        """Profondeur d'un stream : n en DATAFLOW ; sans DATAFLOW il doit contenir toute la sous-porteuse."""
        return n if self.dataflow else max(n, 64 * self.nsc)


ARCHS = {
    "seq": Arch("seq", dataflow=False, unrolled=False, ant=1, nsc=1),   # couche par couche, dense_rt partagee
    "flow": Arch("flow", dataflow=True, unrolled=True, ant=4, nsc=1),   # etages DATAFLOW, une sous-porteuse par appel
    "stream": Arch("stream", dataflow=True, unrolled=True, ant=4, nsc=64),   # flot continu de 64 sous-porteuses
}


# ------------------------------------------------------------------ utilitaires ---------------------------------------
def bits_s(v):
    """Bits signes pour |x| <= v."""
    return int(v).bit_length() + 1


def bits_u(v):
    return max(int(v).bit_length(), 1)


def carr(name, a, ctype="int", qual="static const"):
    a = np.asarray(a)
    dims = "".join(f"[{s}]" for s in a.shape)
    body = ", ".join(str(int(v)) for v in a.reshape(-1))
    return f"{qual} {ctype} {name}{dims} = {{{body}}};\n"


class Widths:
    """Largeurs des types C++ : bornes exactes pour les entiers de produits, maxima releves (+ marge) ailleurs."""
    MARGIN = 2

    def __init__(self, ip: IntPairPrecoder, peak):
        K, M = ip.K, ip.M
        self.act = bits_s(peak["act"]) + self.MARGIN                        # valeurs en unites 2^-F
        self.hq, self.cq = bits_s(ip.q_h.lim), bits_s(ip.q_c.lim)
        self.n2 = bits_u(M * 2 * (2 ** 15) ** 2)                            # 38
        self.ri_norm = max(bits_u(peak["ri_norm"]) + self.MARGIN, bits_u(1 << 28) )   # n2 >= 2^28 (mantisse de crete >= 2^14)
        self.hp = bits_s(peak["hp"]) + self.MARGIN
        self.n2a = bits_u(peak["n2a_sum"]) + 4
        self.gacc = bits_s(M * 2 * ip.q_h.lim ** 2)
        self.g = bits_s(peak["ef"]) + self.MARGIN                           # Re G, Im G, |G|
        self.g2 = 2 * self.g                                                 # |G|^2 (racine : W.g bits)
        self.wacc = bits_s(K * 2 * ip.q_h.lim * ip.q_c.lim)
        self.p = bits_u(M * K * 2 * (K * 2 * ip.q_h.lim * ip.q_c.lim) ** 2)
        self.ri_pow = bits_u(peak["ri_pow"]) + self.MARGIN
        self.var = bits_u(peak["var"]) + 2 * self.MARGIN                    # variance : marge doublee (entree de l'irsqrt)
        self.inv_ln = bits_u(1 << (3 * F // 2)) + 1                         # borne exacte : <= 2^(3F/2)
        self.rt_in = max(bits_s(L.lim) for L in ip.layers())
        self.rt_w = max(bits_s(np.abs(L.w).max()) for L in ip.layers())
        self.rt_acc = max(bits_s(L.w.shape[1] * L.lim * np.abs(L.w).max()) for L in ip.layers())

    def types(self):
        return (f"typedef ap_int<16> mant_t;            // mantisses du canal\n"
                f"typedef ap_int<8> ex_t;               // exposants (un par utilisateur)\n"
                f"typedef ap_int<{self.act}> act_t;      // valeurs intermediaires, unites 2^-FRAC\n"
                f"typedef ap_int<{self.hq}> hq_t;        // canal normalise quantifie\n"
                f"typedef ap_int<{self.cq}> cq_t;        // coefficients C quantifies\n"
                f"typedef ap_uint<{self.n2}> n2_t;       // puissance d'un utilisateur (mantisses)\n"
                f"typedef ap_uint<{self.n2a}> n2a_t;     // puissances alignees sur l'exposant minimal\n"
                f"typedef ap_uint<{self.ri_norm}> rin_t; // 2^(FRAC+RCP) / sqrt(n2)\n"
                f"typedef ap_int<{self.gacc}> gacc_t;    // accumulateur de G\n"
                f"typedef ap_int<{self.wacc}> wacc_t;    // accumulateur de W = h^H C\n"
                f"typedef ap_uint<{self.p}> pw_t;        // puissance totale de W\n"
                f"typedef ap_uint<{self.ri_pow}> rip_t;  // 2^(FRAC+RCP) / sqrt(p)\n"
                f"typedef ap_uint<{self.var}> var_t;     // variance (LayerNorm)\n"
                f"typedef ap_uint<{self.inv_ln}> inv_t;  // 2^(3FRAC/2) / sqrt(var)\n"
                f"typedef ap_int<{self.rt_in}> rt_in_t; typedef ap_int<{self.rt_w}> rt_w_t; "
                f"typedef ap_int<{self.rt_acc}> rt_acc_t;   // couche dense partagee (arch seq)\n")


# ------------------------------------------------------------------ blocs ---------------------------------------------
class MathGen:
    """Operations elementaires communes et tables (identiques a hls/intref.py)."""

    def __init__(self, ip: IntPairPrecoder):
        self.ip = ip

    def constants(self):
        return (carr("GELU_TAB", self.ip.gelu_tab) + carr("LOG2_TAB", LOG2_TAB) + carr("RS_TAB", RS_TAB, "long long")
                + f"static const int LOG10_2 = {LOG10_2};\n")

    def source(self, a: Arch):
        return r"""
// Arrondi au plus proche (vers le haut) puis decalage : S constant, puis s variable (s <= 0 : decalage a gauche).
template<int S, int W, bool SG> static inline ap_int<(W + 1 - S > 1 ? W + 1 - S : 1)> rsh(const ap_int_base<W, SG>& x) {
    ap_int<W + 1> t = x; t += (ap_int<W + 1>)1 << (S - 1); return t >> S;
}
template<int W, bool SG> static inline ap_int<W + 1> rnd_shift(const ap_int_base<W, SG>& x, int s) {
    ap_int<W + 1> t = x;
    if (s > 0) { t += (ap_int<W + 1>)1 << (s - 1); return t >> s; }
    return t << (-s);
}
// Quantification LSQ : q = clamp(rsh<QS>(x * QM), -LIM, LIM), resultat sur b + 1 bits (type T).
template<int QM, int QS, int LIM, class T, int W, bool SG> static inline T quant(const ap_int_base<W, SG>& x) {
    ap_int<W + 18 - QS> q = rsh<QS>(x * (ap_int<17>)QM);
    return q > LIM ? (T)LIM : (q < -LIM ? (T)(-LIM) : (T)q);
}
template<class T, int W, bool SG> static inline T quant_rt(const ap_int_base<W, SG>& x, int qm, int qs, int lim) {
    ap_int<W + 18> q = rnd_shift(x * (ap_int<17>)qm, qs);
    return q > lim ? (T)lim : (q < -lim ? (T)(-lim) : (T)q);
}
template<int W> static inline int msb_of(ap_uint<W> x) {      // position du bit de tete (x >= 1)
    int msb = 0;
    for (int b = 0; b < W; b++) {
#pragma HLS UNROLL
        if (x[b]) msb = b;
    }
    return msb;
}
template<int W> static inline ap_uint<W / 2> isqrt(ap_uint<W> x) {   // racine entiere exacte, W pair, W/2 tours
    ap_uint<W + 2> v = x, r = 0, b = (ap_uint<W + 2>)1 << (W - 2);
    for (int it = 0; it < W / 2; it++) {
#pragma HLS UNROLL
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return r;
}
// ~ 2^S / sqrt(x) sans division : t = x * 2^-e2 dans [2^30, 2^32) (e2 pair), table RS_TAB interpolee (u = t / 2^32)
// puis un tour de Newton y <- y (3 - u y^2) / 2 en Q30 ; resultat = y * 2^(S - 46 - e2/2), sur RW bits.
template<int W, int S, int RW> static inline ap_uint<RW> irsqrt(ap_uint<W> x0) {
    ap_uint<W> x = x0 < 1 ? (ap_uint<W>)1 : x0;
    int msb = msb_of<W>(x), e2 = msb - 30 - (msb & 1);
    ap_uint<64> xw = x;
    ap_uint<32> t = e2 >= 0 ? (ap_uint<32>)(xw >> e2) : (ap_uint<32>)(xw << (-e2));
    int i = (t >> 24) - 64;
    ap_uint<24> fr = t(23, 0);
    ap_int<33> d = (ap_int<33>)RS_TAB[i + 1] - (ap_int<33>)RS_TAB[i];
    ap_int<64> y = (ap_int<64>)RS_TAB[i] + rsh<24>(d * fr);
    ap_int<64> yy = (y * y) >> 30;
    ap_int<64> p = ((ap_int<64>)(t >> 2) * yy) >> 30;
    y = (y * (((ap_int<64>)3 << 30) - p)) >> 31;
    return rnd_shift(y, 46 + (e2 >> 1) - S);
}
template<int W> static inline ap_int<24> ilog2_fix(ap_uint<W> x) {   // log2(x), unites 2^-FRAC : bit de tete + table
    int msb = msb_of<W>(x);
    ap_uint<64> xw = x;
    ap_uint<31> t = msb >= 30 ? (ap_uint<31>)(xw >> (msb - 30)) : (ap_uint<31>)(xw << (30 - msb));
    ap_uint<30> u = t(29, 0);
    int i = u >> 22;
    ap_uint<22> fr = u(21, 0);
    return (ap_int<24>)msb * ONE + LOG2_TAB[i] + rsh<22>((ap_int<14>)(LOG2_TAB[i + 1] - LOG2_TAB[i]) * fr);
}
static inline act_t gelu(act_t x) {           // table de 257 valeurs, pas 1/16 sur [-8, 8], interpolation lineaire
    const act_t lo = -8 * ONE; const int sh = FRAC - 4;
    if (x >= 8 * ONE) return x;
    if (x <= lo) return 0;
    ap_uint<17> u = x - lo;
    int i = u >> sh; if (i > 255) i = 255;
    ap_uint<8> fr = u - (i << sh);
    return GELU_TAB[i] + rsh<sh>((ap_int<14>)(GELU_TAB[i + 1] - GELU_TAB[i]) * fr);
}
"""


class DenseGen:
    """Couche dense quantifiee : struct de constantes L_<tag> et appel dense<L_tag> (constantes, deroule)
    ou dense_rt<NIN, NOUT> (module partage, poids en ROM)."""

    def __init__(self, L, tag):
        self.L, self.tag = L, tag
        self.nin, self.nout = L.w.shape[1], L.w.shape[0]
        self.acc = bits_s(self.nin * L.lim * np.abs(L.w).max())
        assert np.all(L.out.e - F > 0), f"{tag} : decalage de sortie <= 0"

    @property
    def struct(self):
        return f"L_{self.tag}"

    def constants(self):
        L, s = self.L, self.struct
        bias = L.bias if L.bias is not None else np.zeros(self.nout, dtype=np.int64)
        out = (f"struct {s} {{\n"
               f"    static const int NIN = {self.nin}, NOUT = {self.nout}, QM = {int(L.qin.m)}, QS = {int(L.qin.e) + F},"
               f" LIM = {L.lim}, ACC = {self.acc};\n"
               f"    typedef ap_int<{bits_s(L.lim)}> in_t; typedef ap_int<{bits_s(np.abs(L.w).max())}> w_t;"
               f" typedef ap_int<{self.acc}> acc_t;\n"
               f"    static const int W[{self.nout}][{self.nin}], M[{self.nout}], S[{self.nout}], B[{self.nout}];\n}};\n")
        out += carr(f"{s}::W", L.w, qual="const") + carr(f"{s}::M", L.out.m, qual="const")
        out += carr(f"{s}::S", L.out.e - F, qual="const") + carr(f"{s}::B", bias, qual="const")
        return out

    def call(self, a: Arch, x, y):
        s = self.struct
        if a.const:
            return f"dense<{s}>({x}, {y});"
        return f"dense_rt<{s}::NIN, {s}::NOUT>({x}, {y}, {s}::W, {s}::M, {s}::S, {s}::B, {s}::QM, {s}::QS, {s}::LIM);"

    @staticmethod
    def templates(a: Arch):
        """Les deux formes de la couche dense (emises une fois)."""
        return r"""
// Couche dense deroulee : poids constants (P::W), un token par appel. y = rnd_shift(sum q(x) W, S) + B.
template<class P> static void dense(const act_t x[P::NIN], act_t y[P::NOUT]) {
#pragma HLS INLINE
    typename P::in_t xq[P::NIN];
    for (int i = 0; i < P::NIN; i++) {
#pragma HLS UNROLL
        xq[i] = quant<P::QM, P::QS, P::LIM, typename P::in_t>(x[i]);
    }
    for (int o = 0; o < P::NOUT; o++) {
#pragma HLS UNROLL
        typename P::acc_t acc = 0;
        for (int i = 0; i < P::NIN; i++) {
#pragma HLS UNROLL
            acc += xq[i] * (typename P::w_t)P::W[o][i];
        }
        y[o] = rnd_shift(acc * (ap_int<17>)P::M[o], P::S[o]) + P::B[o];
    }
}
// Couche dense partagee (arch seq) : poids en ROM, une sortie par cycle, une ligne de W lue par cycle.
template<int NIN, int NOUT> static void dense_rt(const act_t *x, act_t *y, const int w[NOUT][NIN], const int *m,
                                                 const int *s, const int *b, int qm, int qs, int lim) {
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable=w complete dim=2
    rt_in_t xq[NIN];
#pragma HLS ARRAY_PARTITION variable=xq complete
    for (int i = 0; i < NIN; i++) {
#pragma HLS UNROLL
        xq[i] = quant_rt<rt_in_t>(x[i], qm, qs, lim);
    }
    for (int o = 0; o < NOUT; o++) {
#pragma HLS PIPELINE II=1
        rt_acc_t acc = 0;
        for (int i = 0; i < NIN; i++) {
#pragma HLS UNROLL
            acc += xq[i] * (rt_w_t)w[o][i];
        }
        y[o] = rnd_shift(acc * (ap_int<17>)m[o], s[o]) + b[o];
    }
}
"""


class QuantGen:
    """Quantificateur d'operande de produit (echelle scalaire) : struct Q_<tag>."""

    def __init__(self, q, tag, ctype):
        self.q, self.tag, self.ctype = q, tag, ctype

    def constants(self):
        q = self.q
        return (f"struct Q_{self.tag} {{ static const int QM = {int(q.qin.m)}, QS = {int(q.qin.e) + F}, LIM = {q.lim}; "
                f"typedef {self.ctype} out_t; }};\n")

    def call(self, x):
        s = f"Q_{self.tag}"
        return f"quant<{s}::QM, {s}::QS, {s}::LIM, {s}::out_t>({x})"


class LayerNormGen:
    """LayerNorm sur les D2 valeurs d'un token : moyenne, variance, inverse de racine (Newton), affine."""

    def __init__(self, blk, i, W: Widths):
        self.w, self.b, self.i, self.W = blk["ln_w"], blk["ln_b"], i, W

    def constants(self):
        return (f"struct LN_{self.i} {{ typedef ap_int<{bits_s(np.abs(self.w).max())}> w_t; "
                f"static const int W[{len(self.w)}], B[{len(self.b)}]; }};\n"
                + carr(f"LN_{self.i}::W", self.w, qual="const") + carr(f"LN_{self.i}::B", self.b, qual="const"))

    def call(self, x, y):
        return f"layernorm<LN_{self.i}>({x}, {y});"

    @staticmethod
    def template(a: Arch, W: Widths):
        sh = int(np.log2(2 * 16))                     # D2 = 32
        return f"""
template<class P> static void layernorm(const act_t x[D2], act_t y[D2]) {{
#pragma HLS INLINE
    ap_int<{W.act + sh}> s = 0;
    for (int i = 0; i < D2; i++) {{
{a.unroll()}        s += x[i];
    }}
    act_t mu = rsh<{sh}>(s);
    ap_uint<{2 * W.act + sh}> v = 0;
    for (int i = 0; i < D2; i++) {{
{a.unroll()}        act_t c = x[i] - mu; v += c * c;
    }}
    var_t var = rsh<{sh + F}>(v);
    inv_t inv = irsqrt<{W.var}, 3 * FRAC / 2, {W.inv_ln}>(var);
    for (int i = 0; i < D2; i++) {{
{a.unroll()}        act_t c = x[i] - mu;
        y[i] = rsh<FRAC>(rsh<FRAC>(c * inv) * (typename P::w_t)P::W[i]) + P::B[i];
    }}
}}
"""




# ------------------------------------------------------------------ etages (processus) --------------------------------
# Chaque etage est une boucle aplatie sur (sous-porteuse, token) ou (sous-porteuse, groupe d'antennes) a II = 1, reliee
# aux autres par des hls::stream : un element par token (vec_t), par groupe d'antennes (mgrp_t, hgrp_t, wgrp_t) ou par
# sous-porteuse (ri_sc_t, pdb_sc_t, gsc_t, qa_sc_<i>, qb_sc_<i>, c_sc_t : emis au dernier token / groupe, lus au premier).

class NormalizeGen:
    """Normalisation par utilisateur : n2 = sum |mantisse|^2 (ANT antennes par cycle), puis a la fin de la sous-porteuse
    irsqrt (Newton) et puissances relatives en dB ; second processus : h = mantisse * ri, quantification q_h, deux copies
    (pour G et pour W)."""

    def __init__(self, qh: QuantGen, W: Widths):
        self.qh, self.W = qh, W

    def source(self, a: Arch):
        W = self.W
        return f"""
static void normalize_acc(const mant_t mr[NSC][K][M], const mant_t mi[NSC][K][M], const ex_t ex[NSC][K],
                          hls::stream<mgrp_t>& m_out, hls::stream<ri_sc_t>& ri_out, hls::stream<pdb_sc_t>& pdb_out) {{
{a.stage()}    n2_t n2[K];
#pragma HLS ARRAY_PARTITION variable=n2 complete
    for (int it = 0; it < NSC * NG; it++) {{
{a.pipe()}        const int sc = it / NG, g = it % NG;
        mgrp_t grp;
        for (int k = 0; k < K; k++) {{
{a.unroll()}            ap_uint<{W.n2}> s = 0;
            for (int q = 0; q < ANT; q++) {{
{a.unroll()}                grp.r[k][q] = mr[sc][k][g * ANT + q]; grp.i[k][q] = mi[sc][k][g * ANT + q];
                s += grp.r[k][q] * grp.r[k][q] + grp.i[k][q] * grp.i[k][q];
            }}
            n2[k] = (g == 0 ? (n2_t)0 : n2[k]) + s;
        }}
        m_out.write(grp);
        if (g == NG - 1) {{                                                    // fin de la sous-porteuse
            ri_sc_t ri; pdb_sc_t pdb; n2a_t n2a[K]; ap_uint<{W.n2a + 2}> sa = 0; int exk[K];
            for (int k = 0; k < K; k++) {{
{a.unroll()}                exk[k] = ex[sc][k].to_int();                        // une lecture par exposant
            }}
            int emin = exk[0];
            for (int k = 1; k < K; k++) {{
{a.unroll()}                if (exk[k] < emin) emin = exk[k];
            }}
            for (int k = 0; k < K; k++) {{
{a.unroll()}                ri.v[k] = irsqrt<{W.n2}, FRAC + RCP, {W.ri_norm}>(n2[k]);
                n2a[k] = (n2a_t)n2[k] << (2 * (exk[k] - emin)); sa += n2a[k];
            }}
            ap_int<24> ls = ilog2_fix<{W.n2a + 2}>(sa) - 2 * ONE;                   // log2 de la moyenne (K = 4)
            for (int k = 0; k < K; k++) {{
{a.unroll()}                pdb.v[k] = rsh<FRAC>((ilog2_fix<{W.n2a}>(n2a[k]) - ls) * (ap_int<13>)LOG10_2);
            }}
            ri_out.write(ri); pdb_out.write(pdb);
        }}
    }}
}}

static void normalize_scale(hls::stream<mgrp_t>& m_in, hls::stream<ri_sc_t>& ri_in, hls::stream<hgrp_t>& h_g,
                            hls::stream<hgrp_t>& h_w) {{
{a.stage()}    ri_sc_t ri;
    for (int it = 0; it < NSC * NG; it++) {{
{a.pipe()}        if (it % NG == 0) ri = ri_in.read();
        mgrp_t grp = m_in.read();
        hgrp_t h;
        for (int k = 0; k < K; k++) {{
{a.unroll()}            for (int q = 0; q < ANT; q++) {{
{a.unroll()}                h.r[k][q] = {self.qh.call("rsh<RCP>(grp.r[k][q] * ri.v[k])")};
                h.i[k][q] = {self.qh.call("rsh<RCP>(grp.i[k][q] * ri.v[k])")};
            }}
        }}
        h_g.write(h); h_w.write(h);
    }}
}}
"""


class GramGen:
    """G = h h^H : accumulation sur les 10 paires j <= k (ANT antennes par cycle), puis les 16 entrees par token
    (G[k][j] = conj G[j][k], arrondi fait sur la valeur signee comme dans la reference), |G| par racine entiere.
    Une copie d'ef par consommateur (init et chaque couche)."""

    def __init__(self, ip: IntPairPrecoder, W: Widths):
        self.ip, self.W = ip, W
        K = ip.K
        self.pairs = [(j, k) for j in range(K) for k in range(j, K)]
        self.idx = np.zeros((K, K), dtype=np.int64)
        for p, (j, k) in enumerate(self.pairs):
            self.idx[j, k] = self.idx[k, j] = p

    def constants(self):
        g = self.ip.g_out
        return (f"#define N_PAIRS {len(self.pairs)}\nstatic const int GM = {int(g.m)}, GS = {int(g.e) - F};\n"
                + carr("PJ", [j for j, _ in self.pairs]) + carr("PK", [k for _, k in self.pairs]) + carr("PIDX", self.idx))

    def source(self, a: Arch, ncopies):
        W = self.W
        args = ", ".join(f"hls::stream<ef_t>& ef{c}" for c in range(ncopies))
        writes = " ".join(f"ef{c}.write(e);" for c in range(ncopies))
        return f"""
static void gram_acc(hls::stream<hgrp_t>& h_in, hls::stream<gsc_t>& g_out) {{
{a.stage()}    gacc_t gr[N_PAIRS], gi[N_PAIRS];
#pragma HLS ARRAY_PARTITION variable=gr complete
#pragma HLS ARRAY_PARTITION variable=gi complete
    for (int it = 0; it < NSC * NG; it++) {{
{a.pipe()}        const int g = it % NG;
        hgrp_t h = h_in.read();
        for (int p = 0; p < N_PAIRS; p++) {{
{a.unroll()}            const int j = PJ[p], k = PK[p];
            gacc_t sr = 0, si = 0;
            for (int q = 0; q < ANT; q++) {{
{a.unroll()}                sr += h.r[j][q] * h.r[k][q] + h.i[j][q] * h.i[k][q];
                si += h.i[j][q] * h.r[k][q] - h.r[j][q] * h.i[k][q];
            }}
            gr[p] = (g == 0 ? (gacc_t)0 : gr[p]) + sr; gi[p] = (g == 0 ? (gacc_t)0 : gi[p]) + si;
        }}
        if (g == NG - 1) {{
            gsc_t G;
            for (int p = 0; p < N_PAIRS; p++) {{
{a.unroll()}                G.r[p] = gr[p]; G.i[p] = gi[p];
            }}
            g_out.write(G);
        }}
    }}
}}

static void gram_edges(hls::stream<gsc_t>& g_in, {args}) {{
{a.stage()}    gsc_t G;
    for (int it = 0; it < NSC * NT; it++) {{
{a.pipe()}        const int t = it % NT, j = t / K, k = t % K, p = PIDX[j][k];
        if (t == 0) G = g_in.read();
        gacc_t gim = j <= k ? G.i[p] : (gacc_t)(-G.i[p]);
        ap_int<{W.g}> g_re = rnd_shift(G.r[p] * (ap_int<17>)GM, GS), g_im = rnd_shift(gim * (ap_int<17>)GM, GS);
        ap_uint<{W.g2}> g2 = g_re * g_re + g_im * g_im;
        ef_t e;
        e.v[0] = g_re; e.v[1] = g_im; e.v[2] = isqrt<{W.g2}>(g2);
        {writes}
    }}
}}
"""


class InitGen:
    """Couche d'entree sur [Re G, Im G, |G|, 1(j = k), SNR / 10, p_j, p_k] pour les K*K tokens."""

    def __init__(self, dense: DenseGen):
        self.dense = dense

    def source(self, a: Arch):
        return f"""
static void init_layer(hls::stream<ef_t>& ef_in, const act_t snr10[NSC], hls::stream<pdb_sc_t>& pdb_in,
                       hls::stream<tok_t>& e_out) {{
{a.stage()}    pdb_sc_t pdb;
    for (int it = 0; it < NSC * NT; it++) {{
{a.pipe()}        const int sc = it / NT, t = it % NT, j = t / K, k = t % K;
        if (t == 0) pdb = pdb_in.read();
        ef_t ef = ef_in.read();
        act_t x[7];
#pragma HLS ARRAY_PARTITION variable=x complete
        x[0] = ef.v[0]; x[1] = ef.v[1]; x[2] = ef.v[2]; x[3] = j == k ? (act_t)ONE : (act_t)0; x[4] = snr10[sc];
        x[5] = pdb.v[j]; x[6] = pdb.v[k];
        tok_t e;
        {self.dense.call(a, "x", "e.v")}
        e_out.write(e);
    }}
}}
"""


class ProjGen:
    """Etage 1 d'une couche triangulaire : x = [LayerNorm(e), ef] ; a, b = projections ; quantification q_a, q_b.
    Les 16 tokens quantifies de la sous-porteuse sont emis en quatre blocs re/im (qa_sc_<i>, qb_sc_<i>, <= 4096 bits) au dernier token ; e et x sont relayes."""

    def __init__(self, i, ln: LayerNormGen, da: DenseGen, db: DenseGen, qa: QuantGen, qb: QuantGen):
        self.i, self.ln, self.da, self.db, self.qa, self.qb = i, ln, da, db, qa, qb

    def source(self, a: Arch):
        i, qa, qb = self.i, self.qa.ctype, self.qb.ctype
        return f"""
// Les 16 tokens quantifies d'une sous-porteuse, en quatre blocs re/im (un element de stream <= 4096 bits).
struct qa_sc_{i} {{ {qa} v[NT][D]; }};
struct qb_sc_{i} {{ {qb} v[NT][D]; }};

static void proj_{i}(hls::stream<tok_t>& e_in, hls::stream<ef_t>& ef_in, hls::stream<tok_t>& e_out, hls::stream<x_t>& x_out,
                   hls::stream<qa_sc_{i}>& ar_out, hls::stream<qa_sc_{i}>& ai_out, hls::stream<qb_sc_{i}>& br_out,
                   hls::stream<qb_sc_{i}>& bi_out) {{
{a.stage()}    {qa} ar[NT][D], ai[NT][D]; {qb} br[NT][D], bi[NT][D];
{a.partition("ar", 0)}{a.partition("ai", 0)}{a.partition("br", 0)}{a.partition("bi", 0)}    for (int it = 0; it < NSC * NT; it++) {{
{a.pipe()}        const int t = it % NT;
        tok_t e = e_in.read(); ef_t ef = ef_in.read();
        x_t x; act_t pa[D2], pb[D2];
#pragma HLS ARRAY_PARTITION variable=pa complete
#pragma HLS ARRAY_PARTITION variable=pb complete
        {self.ln.call("e.v", "x.v")}
        for (int c = 0; c < 3; c++) {{
{a.unroll()}            x.v[D2 + c] = ef.v[c];
        }}
        {self.da.call(a, "x.v", "pa")}
        {self.db.call(a, "x.v", "pb")}
        for (int c = 0; c < D; c++) {{
{a.unroll()}            ar[t][c] = {self.qa.call("pa[c]")}; ai[t][c] = {self.qa.call("pa[D + c]")};
            br[t][c] = {self.qb.call("pb[c]")}; bi[t][c] = {self.qb.call("pb[D + c]")};
        }}
        e_out.write(e); x_out.write(x);
        if (t == NT - 1) {{
            qa_sc_{i} sar, sai; qb_sc_{i} sbr, sbi;
            for (int u = 0; u < NT; u++) {{
{a.unroll()}                for (int c = 0; c < D; c++) {{
{a.unroll()}                    sar.v[u][c] = ar[u][c]; sai.v[u][c] = ai[u][c]; sbr.v[u][c] = br[u][c]; sbi.v[u][c] = bi[u][c];
                }}
            }}
            ar_out.write(sar); ai_out.write(sai); br_out.write(sbr); bi_out.write(sbi);
        }}
    }}
}}
"""


class FfnGen:
    """Etage 2 : produit triangulaire t(j, k) = sum_l a(j, l) b(l, k) par canal (les 16 tokens quantifies sont lus d'un
    bloc au premier token), FFN f1 -> GELU -> f2, residuel."""

    def __init__(self, i, blk, f1: DenseGen, f2: DenseGen, qa: QuantGen, qb: QuantGen):
        self.i, self.f1, self.f2, self.qa, self.qb = i, f1, f2, qa, qb
        self.t_out = blk["t_out"]
        self.tracc = bits_s(4 * 2 * qa.q.lim * qb.q.lim)       # K = 4 termes complexes

    def constants(self):
        return f"static const int TM{self.i} = {int(self.t_out.m)}, TS{self.i} = {int(self.t_out.e) - F};\n"

    def source(self, a: Arch):
        i, hid, qa, qb = self.i, self.f1.nout, self.qa.ctype, self.qb.ctype
        return f"""
static void ffn_{i}(hls::stream<tok_t>& e_in, hls::stream<x_t>& x_in, hls::stream<qa_sc_{i}>& ar_in, hls::stream<qa_sc_{i}>& ai_in,
                  hls::stream<qb_sc_{i}>& br_in, hls::stream<qb_sc_{i}>& bi_in, hls::stream<tok_t>& e_out) {{
{a.stage()}    {qa} ar[NT][D], ai[NT][D]; {qb} br[NT][D], bi[NT][D];
{a.partition("ar", 0)}{a.partition("ai", 0)}{a.partition("br", 0)}{a.partition("bi", 0)}    for (int it = 0; it < NSC * NT; it++) {{
{a.pipe()}        const int t = it % NT, j = t / K, k = t % K;
        if (t == 0) {{
            qa_sc_{i} sar = ar_in.read(), sai = ai_in.read(); qb_sc_{i} sbr = br_in.read(), sbi = bi_in.read();
            for (int u = 0; u < NT; u++) {{
{a.unroll()}                for (int c = 0; c < D; c++) {{
{a.unroll()}                    ar[u][c] = sar.v[u][c]; ai[u][c] = sai.v[u][c]; br[u][c] = sbr.v[u][c]; bi[u][c] = sbi.v[u][c];
                }}
            }}
        }}
        tok_t e = e_in.read(); x_t x = x_in.read();
        act_t in1[NIN + D2], h1[{hid}], h2[D2];
#pragma HLS ARRAY_PARTITION variable=in1 complete
#pragma HLS ARRAY_PARTITION variable=h1 complete
#pragma HLS ARRAY_PARTITION variable=h2 complete
        for (int n = 0; n < NIN; n++) {{
{a.unroll()}            in1[n] = x.v[n];
        }}
        for (int c = 0; c < D; c++) {{
{a.unroll()}            ap_int<{self.tracc}> tr = 0, ti = 0;
            for (int l = 0; l < K; l++) {{
{a.unroll()}                tr += ar[j * K + l][c] * br[l * K + k][c] - ai[j * K + l][c] * bi[l * K + k][c];
                ti += ar[j * K + l][c] * bi[l * K + k][c] + ai[j * K + l][c] * br[l * K + k][c];
            }}
            in1[NIN + c] = rsh<TS{i}>(tr * (ap_int<17>)TM{i}); in1[NIN + D + c] = rsh<TS{i}>(ti * (ap_int<17>)TM{i});
        }}
        {self.f1.call(a, "in1", "h1")}
        for (int n = 0; n < {hid}; n++) {{
{a.unroll()}            h1[n] = gelu(h1[n]);
        }}
        {self.f2.call(a, "h1", "h2")}
        tok_t eo;
        for (int n = 0; n < D2; n++) {{
{a.unroll()}            eo.v[n] = e.v[n] + h2[n];
        }}
        e_out.write(eo);
    }}
}}
"""


class ReadGen:
    """Couche de lecture : e -> (Re C, Im C) quantifies par q_c, emis d'un bloc par sous-porteuse."""

    def __init__(self, dense: DenseGen, qc: QuantGen):
        self.dense, self.qc = dense, qc

    def source(self, a: Arch):
        return f"""
static void read_layer(hls::stream<tok_t>& e_in, hls::stream<c_sc_t>& c_out) {{
{a.stage()}    c_sc_t C;
    for (int it = 0; it < NSC * NT; it++) {{
{a.pipe()}        const int t = it % NT;
        tok_t e = e_in.read();
        act_t c[2];
#pragma HLS ARRAY_PARTITION variable=c complete
        {self.dense.call(a, "e.v", "c")}
        C.r[t / K][t % K] = {self.qc.call("c[0]")}; C.i[t / K][t % K] = {self.qc.call("c[1]")};
        if (t == NT - 1) c_out.write(C);
    }}
}}
"""


class CombineGen:
    """W = h^H C (entiers, ANT antennes par cycle) et puissance totale ; a la fin de la sous-porteuse irsqrt (Newton) ;
    second processus : normalisation et ecriture de W."""

    def __init__(self, W: Widths):
        self.W = W

    def source(self, a: Arch):
        W = self.W
        return f"""
static void combine_acc(hls::stream<hgrp_t>& h_in, hls::stream<c_sc_t>& c_in, hls::stream<wgrp_t>& w_out,
                        hls::stream<rip_t>& ri_out) {{
{a.stage()}    c_sc_t C; pw_t p = 0;
    for (int it = 0; it < NSC * NG; it++) {{
{a.pipe()}        const int g = it % NG;
        if (g == 0) C = c_in.read();
        hgrp_t h = h_in.read();
        wgrp_t w; pw_t ps = 0;
        for (int q = 0; q < ANT; q++) {{
{a.unroll()}            for (int k = 0; k < K; k++) {{
{a.unroll()}                wacc_t r = 0, im = 0;
                for (int j = 0; j < K; j++) {{
{a.unroll()}                    r += h.r[j][q] * C.r[j][k] + h.i[j][q] * C.i[j][k];
                    im += h.r[j][q] * C.i[j][k] - h.i[j][q] * C.r[j][k];
                }}
                w.r[q][k] = r; w.i[q][k] = im; ps += r * r + im * im;
            }}
        }}
        p = (g == 0 ? (pw_t)0 : p) + ps;
        w_out.write(w);
        if (g == NG - 1) ri_out.write(irsqrt<{W.p}, FRAC + RCP, {W.ri_pow}>(p));
    }}
}}

static void combine_scale(hls::stream<wgrp_t>& w_in, hls::stream<rip_t>& ri_in, act_t w_re[NSC][M][K], act_t w_im[NSC][M][K]) {{
{a.stage()}    rip_t ri;
    for (int it = 0; it < NSC * NG; it++) {{
{a.pipe()}        const int sc = it / NG, g = it % NG;
        if (g == 0) ri = ri_in.read();
        wgrp_t w = w_in.read();
        for (int q = 0; q < ANT; q++) {{
{a.unroll()}            for (int k = 0; k < K; k++) {{
{a.unroll()}                w_re[sc][g * ANT + q][k] = rsh<RCP>(w.r[q][k] * ri); w_im[sc][g * ANT + q][k] = rsh<RCP>(w.i[q][k] * ri);
            }}
        }}
    }}
}}
"""


# ------------------------------------------------------------------ projet --------------------------------------------
class Project:
    """Assemble les blocs selon l'architecture et ecrit types.h, weights.h, precoder.h, precoder.cpp, le banc de test
    (vectors.h, tb.cpp) et les scripts Tcl."""

    def __init__(self, ip: IntPairPrecoder, arch: Arch, peak):
        self.ip, self.a = ip, arch
        self.W = Widths(ip, peak)
        self.math = MathGen(ip)
        self.qh = QuantGen(ip.q_h, "qh", "hq_t")
        self.qc = QuantGen(ip.q_c, "qc", "cq_t")
        self.init = DenseGen(ip.init, "init")
        self.read = DenseGen(ip.read, "read")
        self.blocks = []
        for i, b in enumerate(ip.blocks):
            qa = QuantGen(b["q_a"], f"qa{i}", f"ap_int<{bits_s(b['q_a'].lim)}>")
            qb = QuantGen(b["q_b"], f"qb{i}", f"ap_int<{bits_s(b['q_b'].lim)}>")
            ln = LayerNormGen(b, i, self.W)
            da, db = DenseGen(b["a"], f"a{i}"), DenseGen(b["b"], f"b{i}")
            f1, f2 = DenseGen(b["f1"], f"f1{i}"), DenseGen(b["f2"], f"f2{i}")
            self.blocks.append((ProjGen(i, ln, da, db, qa, qb), FfnGen(i, b, f1, f2, qa, qb)))
        self.gram = GramGen(ip, self.W)
        self.stages = [NormalizeGen(self.qh, self.W), self.gram, InitGen(self.init)]
        for p, f in self.blocks:
            self.stages += [p, f]
        self.stages += [ReadGen(self.read, self.qc), CombineGen(self.W)]

    # fichiers ----------------------------------------------------------------------------------------------------
    def types_h(self):
        ip, hid, a = self.ip, self.blocks[0][1].f1.nout, self.a
        return ("// Genere par hls/codegen.py -- dimensions et types (inclus par le banc de test).\n#pragma once\n#include <ap_int.h>\n"
                f"#define K_USERS {ip.K}\n#define M_ANT {ip.M}\n#define D_STATE {ip.d}\n#define N_LAYERS {len(ip.blocks)}\n"
                f"#define HIDDEN {hid}\n#define FRAC {F}\n#define RCP {RCP}\n#define N_SC {a.nsc}\n#define ANT_PER_CYCLE {a.ant}\n"
                + self.W.types())

    def weights_h(self):
        s = "// Genere par hls/codegen.py -- ne pas modifier a la main.\n#pragma once\n#include \"types.h\"\n"
        s += self.math.constants() + self.gram.constants()
        s += self.init.constants() + self.read.constants() + self.qh.constants() + self.qc.constants()
        for p, f in self.blocks:
            s += p.ln.constants() + p.da.constants() + p.db.constants() + p.qa.constants() + p.qb.constants()
            s += f.f1.constants() + f.f2.constants() + f.constants()
        return s

    def precoder_h(self):
        return ("// Genere par hls/codegen.py -- precodeur par paires quantifie (N_SC sous-porteuses par appel).\n#pragma once\n"
                "#include \"types.h\"\n"
                "// Entrees : canal en virgule flottante par bloc (h_k = mantisse_k * 2^ex_k), SNR / 10 (unites 2^-FRAC).\n"
                "// Sortie : W (M x K) normalise en puissance, unites 2^-FRAC. Tout en entiers.\n"
                "void pair_precoder(const mant_t mr[N_SC][K_USERS][M_ANT], const mant_t mi[N_SC][K_USERS][M_ANT],\n"
                "                   const ex_t ex[N_SC][K_USERS], const act_t snr10[N_SC],\n"
                "                   act_t w_re[N_SC][M_ANT][K_USERS], act_t w_im[N_SC][M_ANT][K_USERS]);\n")

    def preamble(self):
        return f"""// Genere par hls/codegen.py -- meme ordre d'operations que hls/intref.py (reference entiere).
// Architecture : {self.a.name} ({self.a.nsc} sous-porteuse(s) par appel, {self.a.ant} antenne(s) par cycle).
#include "precoder.h"
#include "weights.h"
#include <hls_stream.h>

static const int K = K_USERS, M = M_ANT, D = D_STATE, D2 = 2 * D_STATE, NSC = N_SC, ANT = ANT_PER_CYCLE;
static const int NG = M / ANT, NT = K * K;        // groupes d'antennes et tokens (paires) par sous-porteuse
static const int NIN = D2 + 3;                    // [LayerNorm(e), Re G, Im G, |G|]
#define ONE (1 << FRAC)

template<int N> struct vec_t {{ act_t v[N]; }};
typedef vec_t<D2> tok_t; typedef vec_t<NIN> x_t; typedef vec_t<3> ef_t;
struct mgrp_t {{ mant_t r[K][ANT], i[K][ANT]; }};  // mantisses d'un groupe d'antennes
struct hgrp_t {{ hq_t r[K][ANT], i[K][ANT]; }};    // canal normalise quantifie d'un groupe d'antennes
struct wgrp_t {{ wacc_t r[ANT][K], i[ANT][K]; }};  // W non normalise d'un groupe d'antennes
struct ri_sc_t {{ rin_t v[K]; }};
struct pdb_sc_t {{ act_t v[K]; }};
struct gsc_t {{ gacc_t r[N_PAIRS], i[N_PAIRS]; }};
struct c_sc_t {{ cq_t r[K][K], i[K][K]; }};
"""

    def top(self):
        a, nl = self.a, len(self.blocks)
        s = ("void pair_precoder(const mant_t mr[NSC][K][M], const mant_t mi[NSC][K][M], const ex_t ex[NSC][K],\n"
             "                   const act_t snr10[NSC], act_t w_re[NSC][M][K], act_t w_im[NSC][M][K]) {\n")
        if a.dataflow:
            s += "#pragma HLS DATAFLOW\n"
        for v in ("mr", "mi"):
            s += a.partition(v, 2) + a.partition(v, 3, a.ant)
        s += a.partition("ex", 2)
        for v in ("w_re", "w_im"):
            s += a.partition(v, 3) + a.partition(v, 2, a.ant)
        streams = [("mgrp_t", "m_s", 64), ("ri_sc_t", "ri_s", 4), ("pdb_sc_t", "pdb_s", 4), ("hgrp_t", "h_g", 64),
                   ("hgrp_t", "h_w", 1024), ("gsc_t", "g_s", 4), ("c_sc_t", "c_s", 4), ("wgrp_t", "w_s", 64),
                   ("rip_t", "rip_s", 4)]
        streams += [("ef_t", f"ef{c}", 128 * (2 * c + 1)) for c in range(nl + 1)]
        streams += [("tok_t", f"e{c}", 64) for c in range(2 * nl + 1)]
        for i in range(nl):
            streams += [("x_t", f"x{i}", 64), (f"qa_sc_{i}", f"ar{i}", 2), (f"qa_sc_{i}", f"ai{i}", 2),
                        (f"qb_sc_{i}", f"br{i}", 2), (f"qb_sc_{i}", f"bi{i}", 2)]
        for typ, name, depth in streams:
            s += f"    hls::stream<{typ}> {name};\n#pragma HLS STREAM variable={name} depth={a.depth(depth)}\n"
        efs = ", ".join(f"ef{c}" for c in range(nl + 1))
        s += ("    normalize_acc(mr, mi, ex, m_s, ri_s, pdb_s);\n    normalize_scale(m_s, ri_s, h_g, h_w);\n"
              f"    gram_acc(h_g, g_s);\n    gram_edges(g_s, {efs});\n    init_layer(ef0, snr10, pdb_s, e0);\n")
        for i in range(nl):
            s += (f"    proj_{i}(e{2 * i}, ef{i + 1}, e{2 * i + 1}, x{i}, ar{i}, ai{i}, br{i}, bi{i});\n"
                  f"    ffn_{i}(e{2 * i + 1}, x{i}, ar{i}, ai{i}, br{i}, bi{i}, e{2 * i + 2});\n")
        s += f"    read_layer(e{2 * nl}, c_s);\n    combine_acc(h_w, c_s, w_s, rip_s);\n    combine_scale(w_s, rip_s, w_re, w_im);\n}}\n"
        return s

    def precoder_cpp(self):
        a = self.a
        s = self.preamble() + self.math.source(a) + DenseGen.templates(a) + LayerNormGen.template(a, self.W)
        for st in self.stages:
            s += st.source(a, len(self.blocks) + 1) if isinstance(st, GramGen) else st.source(a)
        return s + "\n" + self.top()

    def tcl(self, synth):
        return ("open_project -reset pair_precoder_prj\nset_top pair_precoder\n"
                "add_files precoder.cpp\nadd_files -tb tb.cpp\nopen_solution -reset solution1\n"
                "set_part xcu280-fsvh2892-2L-e\ncreate_clock -period 4 -name default\n"
                "csim_design -O\n" + ("csynth_design\n" if synth else "") + "exit\n")

    def write(self, out, vectors):
        os.makedirs(out, exist_ok=True)
        files = {"types.h": self.types_h(), "weights.h": self.weights_h(), "precoder.h": self.precoder_h(),
                 "precoder.cpp": self.precoder_cpp(), "vectors.h": vectors, "tb.cpp": TESTBENCH,
                 "run_csim.tcl": self.tcl(False), "run_synth_grm07.tcl": self.tcl(True)}
        for name, text in files.items():
            open(os.path.join(out, name), "w").write(text)
        open(os.path.join(out, "widths.txt"), "w").write(
            "\n".join(f"{k} = {v}" for k, v in vars(self.W).items()) + "\n")


class Vectors:
    """Canaux St-Catherine -> entrees (virgule flottante par bloc) et sorties attendues de la reference entiere.
    Le meme passage (sur n_stats canaux) fournit les maxima pour les largeurs. n est arrondi a un multiple de nsc."""

    def __init__(self, ip: IntPairPrecoder, n=32, n_stats=1000, nsc=1, seed=2026):
        from torch_port.stecath import draw, ours_input, pool
        n = -(-n // nsc) * nsc
        H = draw(pool(), max(n, n_stats), seed=seed)
        h, snr = ours_input(H)
        mr, mi, ex = block_float(h.numpy().astype(np.complex128))
        snr = snr.numpy().astype(np.float64)
        wr, wi = ip.forward(mr, mi, ex, snr)                                  # releve les maxima (ip.peak)
        self.n = n
        self.mr, self.mi, self.ex, self.snr, self.wr, self.wi = mr[:n], mi[:n], ex[:n], snr[:n], wr[:n], wi[:n]

    def header(self):
        s = "// Genere par hls/codegen.py : entrees et sorties de la reference entiere (hls/intref.py).\n#pragma once\n"
        s += f"#define N_VEC {self.n}\n"
        s += carr("MR", self.mr) + carr("MI", self.mi) + carr("EX", self.ex) + carr("SNR", to_fix(self.snr / 10.0))
        return s + carr("W_RE", self.wr) + carr("W_IM", self.wi)


TESTBENCH = r"""// Genere par hls/codegen.py : compare la sortie C++ a la reference entiere (hls/intref.py), bit a bit.
#include "precoder.h"
#include "vectors.h"
#include <cmath>
#include <cstdio>

static mant_t mr[N_SC][K_USERS][M_ANT], mi[N_SC][K_USERS][M_ANT];
static ex_t ex[N_SC][K_USERS];
static act_t snr[N_SC], wr[N_SC][M_ANT][K_USERS], wi[N_SC][M_ANT][K_USERS];

int main() {
    double err = 0, cos_min = 1;
    for (int b = 0; b < N_VEC / N_SC; b++) {                      // un appel = N_SC sous-porteuses
        for (int s = 0; s < N_SC; s++) {
            const int n = b * N_SC + s;
            snr[s] = SNR[n];
            for (int k = 0; k < K_USERS; k++) {
                ex[s][k] = EX[n][k];
                for (int m = 0; m < M_ANT; m++) { mr[s][k][m] = MR[n][k][m]; mi[s][k][m] = MI[n][k][m]; }
            }
        }
        pair_precoder(mr, mi, ex, snr, wr, wi);
        for (int s = 0; s < N_SC; s++) {
            const int n = b * N_SC + s;
            for (int k = 0; k < K_USERS; k++) {
                double dot = 0, na = 0, nb = 0;
                for (int m = 0; m < M_ANT; m++) {
                    double r = wr[s][m][k].to_double(), i = wi[s][m][k].to_double();
                    err = fmax(err, fmax(fabs(r - W_RE[n][m][k]), fabs(i - W_IM[n][m][k])));
                    dot += r * W_RE[n][m][k] + i * W_IM[n][m][k];
                    na += r * r + i * i;
                    nb += (double)W_RE[n][m][k] * W_RE[n][m][k] + (double)W_IM[n][m][k] * W_IM[n][m][k];
                }
                cos_min = fmin(cos_min, dot / sqrt(na * nb + 1e-30));
            }
        }
    }
    printf("ecart max |W| = %.3e, cosinus minimal = %.6f (%d sous-porteuses)\n", err, cos_min, N_VEC);
    bool ok = err == 0;
    printf("%s\n", ok ? "PASS (identique a la reference entiere)" : "FAIL");
    return ok ? 0 : 1;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/users/sala/Documents/test_projet/hls_work/pair_precoder")
    ap.add_argument("--arch", default="stream", choices=sorted(ARCHS))
    ap.add_argument("--nsc", type=int, help="sous-porteuses par appel (defaut : celui de l'architecture)")
    ap.add_argument("--ant", type=int, help="antennes par cycle (defaut : celui de l'architecture)")
    ap.add_argument("--n", type=int, default=64, help="vecteurs de test (arrondi a un multiple de nsc)")
    ap.add_argument("--stats-n", type=int, default=1000, help="canaux pour relever les maxima (largeurs)")
    ap.add_argument("--model", default="art_pair_h16_L2_lam5_qprod_norm")
    a = ap.parse_args()
    arch = ARCHS[a.arch]
    if a.nsc or a.ant:
        arch = Arch(arch.name, arch.dataflow, arch.unrolled, a.ant or arch.ant, a.nsc or arch.nsc)
    ip = IntPairPrecoder(QuantPairPrecoder.load(a.model)[0])
    vec = Vectors(ip, n=a.n, n_stats=a.stats_n, nsc=arch.nsc)
    prj = Project(ip, arch, ip.peak)
    prj.write(a.out, vec.header())
    print("maxima :", ", ".join(f"{k} {v}" for k, v in ip.peak.items()))
    print("largeurs :", ", ".join(f"{k} {v}" for k, v in vars(prj.W).items()))
    print("->", a.out)


if __name__ == "__main__":
    main()
