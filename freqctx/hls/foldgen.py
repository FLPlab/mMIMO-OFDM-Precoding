"""Architecture repliee « fold » : chaque etage traite un token en S cycles (facteur de parallelisme 1/S).

Le budget 5G NR (100 MHz a 30 kHz : 3 276 sous-porteuses par creneau de 0,5 ms, 1 092 avec l'interpolation 1 SC sur 3)
laisse ~137 cycles par sous-porteuse a 300 MHz la ou `stream` en prend 17 : on replie donc les couches denses d'un
facteur S (cycles par token, 16 S cycles par sous-porteuse) pour diviser les multiplieurs par S.

Principe (identique bit a bit a hls/intref.py, seul l'ordonnancement change) :
  - chaque processus est une boucle aplatie (token, s) a II = 1 ; le token d'entree est lu a s = 0, le token de
    sortie emis a s = S - 1 ; a chaque cycle une tranche 1/S du travail (NOUT/S sorties d'une couche dense,
    D2/S elements d'une LayerNorm, D/S canaux du produit triangulaire) ;
  - couche dense : les poids d'une tranche changent a chaque cycle -> petites ROM de S entrees par position de
    MAC (poids de 1 a 2 bits avec l'ART a cout FPGA : produits en LUT, BIND_OP fabric) ; la quantification de
    l'entree est faite sur les S cycles du token precedent (double registre xq_cur / xq_nxt), avec NIN/S
    multiplieurs par la meme constante ; rescale par canal : NOUT/S multiplieurs (constantes en ROM) ;
  - LayerNorm scindee : ln1 (moyenne par arbre, variance par tranches, une inverse de racine) puis ln2 (affine
    par tranches, quantifications faites par les couches a et b consommatrices) ;
  - ef = (Re G, Im G, |G|) est relaye dans le token (plus de FIFO profonde par couche) ;
  - normalize_fin : les K inverses de racine et log2 de fin de sous-porteuse en boucle sur k (une instance) ;
  - etages par antenne (normalisation, G, W) : ANT = max(1, 4 / S) antennes par cycle.

    python -m hls.codegen --arch fold --fold 8 --model art_pair_h16_L2_lam1_costfpga_qprod_norm --out .../l2_fold8
"""
import numpy as np

from hls.codegen import (Arch, CombineGen, DenseGen, GramGen, LayerNormGen, MathGen, NormalizeGen, Project, QuantGen,
                         Widths, bits_s, carr)
from hls.intref import F


class FoldArch(Arch):
    fold: int = 8

    @staticmethod
    def make(fold, nsc=64, cmul="dsp"):
        assert fold >= 2 and fold & (fold - 1) == 0, "S doit etre une puissance de 2 >= 2"
        a = FoldArch("fold", dataflow=True, unrolled=True, ant=max(1, 4 // fold), nsc=nsc, cmul=cmul)
        a.fold = fold
        return a


def ceil_div(a, b):
    return -(-a // b)


def rom(name, a, ctype="int"):
    """Tableau constant local (ROM) partitionne sur la premiere dimension : une petite ROM de S entrees par position."""
    return "    " + carr(name, a, ctype, qual="static const") + f"#pragma HLS ARRAY_PARTITION variable={name} complete dim=1\n"


# ------------------------------------------------------------------ couche dense repliee ------------------------------
class FoldDense:
    """Couche dense repliee : NIN/S quantifications et NOUT/S sorties par cycle, poids en ROM [position][s].
    pre : d'ou vient le vecteur d'entree ; post : ce qu'on fait des sorties (voir source)."""

    def __init__(self, d: DenseGen, S, pre, post, tag=None):
        self.d, self.S, self.pre, self.post = d, S, pre, post
        self.tag = tag or d.tag
        L = d.L
        self.nin, self.nout = d.nin, d.nout
        self.OS, self.QN = ceil_div(self.nout, S), ceil_div(self.nin, S)
        # poids reorganises : WF[oo * NIN + i][s] = W[s * OS + oo][i] (0 au-dela de NOUT)
        W = np.zeros((self.OS * self.nin, S), dtype=np.int64)
        M, SH, B = np.ones((self.OS, S), dtype=np.int64), np.ones((self.OS, S), dtype=np.int64), np.zeros((self.OS, S), dtype=np.int64)
        bias = L.bias if L.bias is not None else np.zeros(self.nout, dtype=np.int64)
        for s in range(S):
            for oo in range(self.OS):
                o = s * self.OS + oo
                if o < self.nout:
                    W[oo * self.nin:(oo + 1) * self.nin, s] = L.w[o]
                    M[oo, s], SH[oo, s], B[oo, s] = L.out.m[o], L.out.e[o] - F, bias[o]
        self.WF, self.MF, self.SF, self.BF = W, M, SH, B
        self.w_bits = bits_s(np.abs(L.w).max())
        self.in_bits = bits_s(L.lim)

    def source(self, a: FoldArch, io):
        """io : in_sig, in_read (code lisant le token d'entree dans xr[0..NIN-1]), extra_sig, post_decl, pre_mac,
        puis soit post_val (expression de la sortie o a partir de y, rangee dans le registre a decalage yr ; emit lit
        yr[n]), soit post_out (code libre par sortie, variables o et y) ; emit : code a s = S - 1 pour le token tok - 1.
        Les tranches sont prises a indices CONSTANTS dans des registres a decalage (xr, xq_nxt, yr) : avec un indice
        variable, Vitis duplique l'operation (quantification) sur chaque element du tableau."""
        S, s = self.S, self.d.struct
        NP, OP = self.QN * S, self.OS * S                     # tailles rembourrees (multiples de S)
        post_val = io.get('post_val')
        yt = io.get('yr_type', 'act_t')
        yr_decl = f"    {yt} yr[{OP}];\n#pragma HLS ARRAY_PARTITION variable=yr complete\n" if post_val else ""
        if post_val:
            post = f"""                {yt} val = 0;
                if (o < {self.nout}) {{
{post_val}                }}
                yv[oo] = val;
"""
            shift_y = f"""            for (int n = 0; n < {OP - self.OS}; n++) {{
#pragma HLS UNROLL
                yr[n] = yr[n + {self.OS}];
            }}
            for (int oo = 0; oo < {self.OS}; oo++) {{
#pragma HLS UNROLL
                yr[{OP - self.OS} + oo] = yv[oo];
            }}
"""
        else:
            post, shift_y = f"""                if (o < {self.nout}) {{
{io['post_out']}                }}
""", ""
        return f"""
static void dense_{self.tag}({io['in_sig']}{io.get('extra_sig', '')}{io['out_sig']}) {{
#pragma HLS INLINE off
{rom('WF', self.WF)}{rom('MF', self.MF)}{rom('SF', self.SF)}{rom('BF', self.BF)}    {s}::in_t xq_cur[{self.nin}], xq_nxt[{NP}]; act_t xr[{NP}];
#pragma HLS ARRAY_PARTITION variable=xq_cur complete
#pragma HLS ARRAY_PARTITION variable=xq_nxt complete
#pragma HLS ARRAY_PARTITION variable=xr complete
{yr_decl}{io.get('post_decl', '')}    for (int it = 0; it < (NTOK + 1) * {S}; it++) {{
#pragma HLS PIPELINE II=1
        const int tok = it / {S}, s = it % {S};
        if (s == 0 && tok < NTOK) {{
{io['in_read']}        }}
        {s}::in_t q[{self.QN}];
#pragma HLS ARRAY_PARTITION variable=q complete
        for (int u = 0; u < {self.QN}; u++) {{                                // quantification du token tok : tranche en tete de xr
#pragma HLS UNROLL
            q[u] = quant<{s}::QM, {s}::QS, {s}::LIM, {s}::in_t>(xr[u]);
        }}
        for (int n = 0; n < {NP - self.QN}; n++) {{                              // registres a decalage (indices constants)
#pragma HLS UNROLL
            xr[n] = xr[n + {self.QN}]; xq_nxt[n] = xq_nxt[n + {self.QN}];
        }}
        for (int u = 0; u < {self.QN}; u++) {{
#pragma HLS UNROLL
            xq_nxt[{NP - self.QN} + u] = q[u];
        }}
{io.get('pre_mac', '')}        if (tok >= 1) {{                                                 // sorties du token tok - 1 (tranche s)
            {yt} yv[{self.OS}];
#pragma HLS ARRAY_PARTITION variable=yv complete
            for (int oo = 0; oo < {self.OS}; oo++) {{
#pragma HLS UNROLL
                const int o = s * {self.OS} + oo;
                {s}::acc_t acc = 0;
                for (int i = 0; i < {self.nin}; i++) {{
#pragma HLS UNROLL
                    ap_int<{self.w_bits}> w = WF[oo * {self.nin} + i][s];
                    ap_int<{self.in_bits + self.w_bits}> p = xq_cur[i] * w;
#pragma HLS BIND_OP variable=p op=mul impl=fabric
                    acc += p;
                }}
                ap_int<{s}::ACC + 17> am = acc * (ap_int<17>)MF[oo][s];
{a.bind_fabric('am')}                act_t y = rnd_shift(am, SF[oo][s]) + BF[oo][s];
{post}            }}
{shift_y}        }}
        if (s == {S} - 1) {{
            for (int i = 0; i < {self.nin}; i++) {{
#pragma HLS UNROLL
                xq_cur[i] = xq_nxt[i];
            }}
            if (tok >= 1) {{
{io['emit']}            }}
        }}
    }}
}}
"""


# ------------------------------------------------------------------ processus ------------------------------------------
class FoldStages:
    """Ecrit tous les processus de l'architecture repliee a partir des blocs du projet."""

    def __init__(self, prj: "FoldProject"):
        self.p, self.a, self.S = prj, prj.a, prj.a.fold
        self.W = prj.W

    # -- normalisation : accumulation (ANT antennes / cycle) + fin de sous-porteuse en boucle sur k -------------------
    def normalize(self):
        W, a = self.W, self.a
        return f"""
struct n2_sc_t {{ n2_t v[K]; ex_t ex[K]; }};

static void normalize_acc(const mant_t mr[NSC][K][M], const mant_t mi[NSC][K][M], const ex_t ex[NSC][K],
                          hls::stream<mgrp_t>& m_out, hls::stream<n2_sc_t>& n2_out) {{
#pragma HLS INLINE off
    n2_t n2[K];
#pragma HLS ARRAY_PARTITION variable=n2 complete
    for (int it = 0; it < NSC * NG; it++) {{
#pragma HLS PIPELINE II=1
        const int sc = it / NG, g = it % NG;
        mgrp_t grp;
        for (int k = 0; k < K; k++) {{
#pragma HLS UNROLL
            ap_uint<{W.n2}> s = 0;
            for (int q = 0; q < ANT; q++) {{
#pragma HLS UNROLL
                grp.r[k][q] = mr[sc][k][g * ANT + q]; grp.i[k][q] = mi[sc][k][g * ANT + q];
                s += grp.r[k][q] * grp.r[k][q] + grp.i[k][q] * grp.i[k][q];
            }}
            n2[k] = (g == 0 ? (n2_t)0 : n2[k]) + s;
        }}
        m_out.write(grp);
        if (g == NG - 1) {{
            n2_sc_t n;
            for (int k = 0; k < K; k++) {{
#pragma HLS UNROLL
                n.v[k] = n2[k]; n.ex[k] = ex[sc][k];
            }}
            n2_out.write(n);
        }}
    }}
}}

// Fin de sous-porteuse en K iterations : une inverse de racine et un log2 par cycle (au lieu de K en parallele).
static void normalize_fin(hls::stream<n2_sc_t>& n2_in, hls::stream<ri_sc_t>& ri_out, hls::stream<pdb_sc_t>& pdb_out) {{
#pragma HLS INLINE off
    n2_sc_t n; ri_sc_t ri; pdb_sc_t pdb; int emin = 0; ap_uint<{W.n2a + 2}> sa = 0; ap_int<24> lg[K];
#pragma HLS ARRAY_PARTITION variable=lg complete
    for (int it = 0; it < NSC * K; it++) {{
#pragma HLS PIPELINE II=1
        const int k = it % K;
        if (k == 0) {{
            n = n2_in.read();
            emin = n.ex[0].to_int();
            for (int kk = 1; kk < K; kk++) {{
#pragma HLS UNROLL
                if (n.ex[kk].to_int() < emin) emin = n.ex[kk].to_int();
            }}
            sa = 0;
        }}
        ri.v[k] = irsqrt<{W.n2}, FRAC + RCP, {W.ri_norm}>(n.v[k]);
        n2a_t n2a = (n2a_t)n.v[k] << (2 * (n.ex[k].to_int() - emin));
        sa += n2a;
        lg[k] = ilog2_fix<{W.n2a}>(n2a);
        if (k == K - 1) {{
            ap_int<24> ls = ilog2_fix<{W.n2a + 2}>(sa) - 2 * ONE;                   // log2 de la moyenne (K = 4)
            for (int kk = 0; kk < K; kk++) {{
#pragma HLS UNROLL
                pdb.v[kk] = rsh<FRAC>((lg[kk] - ls) * (ap_int<13>)LOG10_2);
            }}
            ri_out.write(ri); pdb_out.write(pdb);
        }}
    }}
}}
""" + self.p.norm.source_scale(a)

    # -- couche d'entree -------------------------------------------------------------------------------------------
    def init(self):
        fd = FoldDense(self.p.init, self.S, "init", "tok")
        io = dict(
            in_sig="hls::stream<ef_t>& ef_in, const act_t snr10[NSC], hls::stream<pdb_sc_t>& pdb_in, ",
            out_sig="hls::stream<tok_t>& e_out",
            post_decl="    pdb_sc_t pdb; ef_t efh[2];\n#pragma HLS ARRAY_PARTITION variable=efh complete\n",
            in_read="""            const int sc = tok / NT, t = tok % NT, j = t / K, k = t % K;
            if (t == 0) pdb = pdb_in.read();
            ef_t ef = ef_in.read(); efh[tok & 1] = ef;                       // garde pour le relais (token tok - 1 a l'emission)
            xr[0] = ef.v[0]; xr[1] = ef.v[1]; xr[2] = ef.v[2]; xr[3] = j == k ? (act_t)ONE : (act_t)0;
            xr[4] = snr10[sc]; xr[5] = pdb.v[j]; xr[6] = pdb.v[k];
""",
            post_val="                    val = y;\n",
            emit="""                tok_t e;
                for (int n = 0; n < D2; n++) {
#pragma HLS UNROLL
                    e.v[n] = yr[n];
                }
                e.ef[0] = efh[(tok - 1) & 1].v[0]; e.ef[1] = efh[(tok - 1) & 1].v[1]; e.ef[2] = efh[(tok - 1) & 1].v[2];
                e_out.write(e);
""")
        return fd.source(self.a, io)

    # -- LayerNorm en deux processus -------------------------------------------------------------------------------
    def layernorm(self, i, ln: LayerNormGen):
        S, W = self.S, self.W
        LS = ceil_div(32, S)
        sh = 5                                                         # D2 = 32
        WL = np.zeros((LS, S), dtype=np.int64); BL = np.zeros((LS, S), dtype=np.int64)
        for s in range(S):
            for u in range(LS):
                n = s * LS + u
                if n < 32:
                    WL[u, s], BL[u, s] = ln.w[n], ln.b[n]
        return f"""
static void ln1_{i}(hls::stream<tok_t>& e_in, hls::stream<ln_t>& out) {{
#pragma HLS INLINE off
    ln_t cur; ap_uint<{2 * W.act + sh}> v = 0;
    for (int it = 0; it < NTOK * {S}; it++) {{
#pragma HLS PIPELINE II=1
        const int s = it % {S};
        if (s == 0) {{
            tok_t e = e_in.read();
            ap_int<{W.act + sh}> sum = 0;
            for (int n = 0; n < D2; n++) {{
#pragma HLS UNROLL
                sum += e.v[n]; cur.v[n] = e.v[n];
            }}
            cur.ef[0] = e.ef[0]; cur.ef[1] = e.ef[1]; cur.ef[2] = e.ef[2];
            cur.mu = rsh<{sh}>(sum);
            v = 0;
        }}
        for (int u = 0; u < {LS}; u++) {{
#pragma HLS UNROLL
            const int n = s * {LS} + u;
            if (n < D2) {{ act_t c = cur.v[n] - cur.mu; v += c * c; }}
        }}
        if (s == {S} - 1) {{
            var_t var = rsh<{sh + F}>(v);
            cur.inv = irsqrt<{W.var}, 3 * FRAC / 2, {W.inv_ln}>(var);
            out.write(cur);
        }}
    }}
}}

static void ln2_{i}(hls::stream<ln_t>& in, hls::stream<x_t>& xa, hls::stream<x_t>& xb, hls::stream<x_t>& xt,
                  hls::stream<tok_t>& e_res) {{
#pragma HLS INLINE off
{rom('WL', WL)}{rom('BL', BL)}    ln_t cur; x_t x;
    for (int it = 0; it < NTOK * {S}; it++) {{
#pragma HLS PIPELINE II=1
        const int s = it % {S};
        if (s == 0) cur = in.read();
        for (int u = 0; u < {LS}; u++) {{
#pragma HLS UNROLL
            const int n = s * {LS} + u;
            if (n < D2) {{
                act_t c = cur.v[n] - cur.mu;
                x.v[n] = rsh<FRAC>(rsh<FRAC>(c * cur.inv) * (LN_{i}::w_t)WL[u][s]) + BL[u][s];
            }}
        }}
        if (s == {S} - 1) {{
            x.v[D2] = cur.ef[0]; x.v[D2 + 1] = cur.ef[1]; x.v[D2 + 2] = cur.ef[2];
            xa.write(x); xb.write(x); xt.write(x);
            tok_t e;
            for (int n = 0; n < D2; n++) {{
#pragma HLS UNROLL
                e.v[n] = cur.v[n];
            }}
            e.ef[0] = cur.ef[0]; e.ef[1] = cur.ef[1]; e.ef[2] = cur.ef[2];
            e_res.write(e);
        }}
    }}
}}
"""

    # -- projections a et b : quantification de sortie et bloc des 16 tokens par sous-porteuse (emis au dernier token) --
    #    (essais ecartes : tokens par token + registres a decalage dans tri = +9 k LUT par couche ; memoire LUTRAM double
    #    banque avec extraction de canal par decalage variable = +30 k LUT par couche, estimation Vitis)
    def proj(self, i, dg: DenseGen, q: QuantGen, which):
        fd = FoldDense(dg, self.S, "x", "block")
        typ = f"q{which}_sc_{i}"
        io = dict(
            in_sig="hls::stream<x_t>& x_in, ",
            out_sig=f"hls::stream<{typ}>& r_out, hls::stream<{typ}>& i_out",
            post_decl=f"    {q.ctype} br[NT][D], bi[NT][D];\n#pragma HLS ARRAY_PARTITION variable=br complete dim=0\n"
                      f"#pragma HLS ARRAY_PARTITION variable=bi complete dim=0\n",
            in_read="""            x_t x = x_in.read();
            for (int n = 0; n < NIN; n++) {
#pragma HLS UNROLL
                xr[n] = x.v[n];
            }
""",
            post_out=f"""                    const int t = (tok - 1) % NT;
                    {q.ctype} yq = {q.call('y')};
                    if (o < D) br[t][o] = yq; else bi[t][o - D] = yq;
""",
            emit=f"""                if ((tok - 1) % NT == NT - 1) {{
                    {typ} sr, si;
                    for (int u = 0; u < NT; u++) {{
#pragma HLS UNROLL
                        for (int c = 0; c < D; c++) {{
#pragma HLS UNROLL
                            sr.v[u][c] = br[u][c]; si.v[u][c] = bi[u][c];
                        }}
                    }}
                    r_out.write(sr); i_out.write(si);
                }}
""")
        return fd.source(self.a, io)

    # -- produit triangulaire : les 4 blocs (re/im de a et b) sont lus au premier token de la sous-porteuse, D/S canaux
    #    par cycle, concatenation [x, t] pour f1 -------------------------------------------------------------------
    def tri(self, i, ffn):
        S = self.S
        CS = ceil_div(16, S)
        qa, qb = ffn.qa.ctype, ffn.qb.ctype
        return f"""
static void tri_{i}(hls::stream<x_t>& x_in, hls::stream<qa_sc_{i}>& ar_in, hls::stream<qa_sc_{i}>& ai_in,
                  hls::stream<qb_sc_{i}>& br_in, hls::stream<qb_sc_{i}>& bi_in, hls::stream<f_t>& out) {{
#pragma HLS INLINE off
    {qa} ar[NT][D], ai[NT][D]; {qb} br[NT][D], bi[NT][D];
#pragma HLS ARRAY_PARTITION variable=ar complete dim=0
#pragma HLS ARRAY_PARTITION variable=ai complete dim=0
#pragma HLS ARRAY_PARTITION variable=br complete dim=0
#pragma HLS ARRAY_PARTITION variable=bi complete dim=0
    f_t f;
    for (int it = 0; it < NTOK * {S}; it++) {{
#pragma HLS PIPELINE II=1
        const int tok = it / {S}, s = it % {S}, t = tok % NT, j = t / K, k = t % K;
        if (s == 0) {{
            if (t == 0) {{
                qa_sc_{i} sar = ar_in.read(), sai = ai_in.read(); qb_sc_{i} sbr = br_in.read(), sbi = bi_in.read();
                for (int u = 0; u < NT; u++) {{
#pragma HLS UNROLL
                    for (int c = 0; c < D; c++) {{
#pragma HLS UNROLL
                        ar[u][c] = sar.v[u][c]; ai[u][c] = sai.v[u][c]; br[u][c] = sbr.v[u][c]; bi[u][c] = sbi.v[u][c];
                    }}
                }}
            }}
            x_t x = x_in.read();
            for (int n = 0; n < NIN; n++) {{
#pragma HLS UNROLL
                f.v[n] = x.v[n];
            }}
        }}
        for (int u = 0; u < {CS}; u++) {{
#pragma HLS UNROLL
            const int c = s * {CS} + u;
            if (c < D) {{
                ap_int<{ffn.tracc}> tr = 0, ti = 0;
                for (int l = 0; l < K; l++) {{
#pragma HLS UNROLL
                    tr += ar[j * K + l][c] * br[l * K + k][c] - ai[j * K + l][c] * bi[l * K + k][c];
                    ti += ar[j * K + l][c] * bi[l * K + k][c] + ai[j * K + l][c] * br[l * K + k][c];
                }}
                ap_int<{ffn.tracc + 17}> tmr = tr * (ap_int<17>)TM{i}, tmi = ti * (ap_int<17>)TM{i};
{self.a.bind_fabric('tmr')}{self.a.bind_fabric('tmi')}                f.v[NIN + c] = rsh<TS{i}>(tmr); f.v[NIN + D + c] = rsh<TS{i}>(tmi);
            }}
        }}
        if (s == {S} - 1) out.write(f);
    }}
}}
"""

    # -- FFN : f1 (GELU en sortie) puis f2 (residuel) -------------------------------------------------------------------
    def ffn(self, i, ffn):
        hid = ffn.f1.nout
        f1 = FoldDense(ffn.f1, self.S, "f", "gelu")
        io1 = dict(
            in_sig="hls::stream<f_t>& f_in, ", out_sig="hls::stream<h_t>& h_out",
            in_read=f"""            f_t f = f_in.read();
            for (int n = 0; n < NIN + D2; n++) {{
#pragma HLS UNROLL
                xr[n] = f.v[n];
            }}
""",
            post_val="                    val = gelu(y);\n",
            emit=f"""                h_t h;
                for (int n = 0; n < {hid}; n++) {{
#pragma HLS UNROLL
                    h.v[n] = yr[n];
                }}
                h_out.write(h);
""")
        f2 = FoldDense(ffn.f2, self.S, "h", "res")
        io2 = dict(
            in_sig="hls::stream<h_t>& h_in, hls::stream<tok_t>& e_res, ", out_sig="hls::stream<tok_t>& e_out",
            post_decl="    tok_t er;\n",
            in_read=f"""            h_t h = h_in.read();
            for (int n = 0; n < {hid}; n++) {{
#pragma HLS UNROLL
                xr[n] = h.v[n];
            }}
""",
            pre_mac="        if (s == 0 && tok >= 1) er = e_res.read();                       // token tok - 1 (residuel)\n",
            post_val="                    val = er.v[o] + y;\n",
            emit="""                tok_t eo;
                for (int n = 0; n < D2; n++) {
#pragma HLS UNROLL
                    eo.v[n] = yr[n];
                }
                eo.ef[0] = er.ef[0]; eo.ef[1] = er.ef[1]; eo.ef[2] = er.ef[2];
                e_out.write(eo);
""")
        return f1.source(self.a, io1) + f2.source(self.a, io2)

    # -- lecture : C quantifie, bloc par sous-porteuse -------------------------------------------------------------------
    def read(self):
        qc = self.p.qc
        fd = FoldDense(self.p.read, self.S, "tok", "cblock")
        io = dict(
            in_sig="hls::stream<tok_t>& e_in, ", out_sig="hls::stream<c_sc_t>& c_out",
            post_decl="    c_sc_t C;\n",
            in_read="""            tok_t e = e_in.read();
            for (int n = 0; n < D2; n++) {
#pragma HLS UNROLL
                xr[n] = e.v[n];
            }
""",
            post_out=f"""                    const int t = (tok - 1) % NT;
                    if (o == 0) C.r[t / K][t % K] = {qc.call('y')}; else C.i[t / K][t % K] = {qc.call('y')};
""",
            emit="                if ((tok - 1) % NT == NT - 1) c_out.write(C);\n")
        return fd.source(self.a, io)


# ------------------------------------------------------------------ projet repli ---------------------------------------
class FoldProject(Project):
    """Projet HLS de l'architecture repliee : memes blocs/constantes que Project, processus de FoldStages."""

    def __init__(self, ip, arch: FoldArch, peak):
        super().__init__(ip, arch, peak)
        self.norm = self.stages[0]
        assert isinstance(self.norm, NormalizeGen)
        self.fs = FoldStages(self)

    def preamble(self):
        a = self.a
        return f"""// Genere par hls/foldgen.py -- meme ordre d'operations que hls/intref.py (reference entiere).
// Architecture : fold (S = {a.fold} cycles par token, {a.nsc} sous-porteuses par appel, {a.ant} antenne(s) par cycle).
#include "precoder.h"
#include "weights.h"
#include <hls_stream.h>

static const int K = K_USERS, M = M_ANT, D = D_STATE, D2 = 2 * D_STATE, NSC = N_SC, ANT = ANT_PER_CYCLE;
static const int NG = M / ANT, NT = K * K;        // groupes d'antennes et tokens (paires) par sous-porteuse
static const int NIN = D2 + 3;                    // [LayerNorm(e), Re G, Im G, |G|]
static const int NTOK = NSC * NT;                 // tokens par appel
#define ONE (1 << FRAC)

template<int N> struct vec_t {{ act_t v[N]; }};
struct tok_t {{ act_t v[D2]; act_t ef[3]; }};      // etat e d'un token + (Re G, Im G, |G|) relaye
struct ln_t {{ act_t v[D2]; act_t ef[3]; act_t mu; inv_t inv; }};
typedef vec_t<NIN> x_t; typedef vec_t<3> ef_t; typedef vec_t<NIN + D2> f_t; typedef vec_t<HIDDEN> h_t;
struct mgrp_t {{ mant_t r[K][ANT], i[K][ANT]; }};  // mantisses d'un groupe d'antennes
struct hgrp_t {{ hq_t r[K][ANT], i[K][ANT]; }};    // canal normalise quantifie d'un groupe d'antennes
struct wgrp_t {{ wacc_t r[ANT][K], i[ANT][K]; }};  // W non normalise d'un groupe d'antennes
struct ri_sc_t {{ rin_t v[K]; }};
struct pdb_sc_t {{ act_t v[K]; }};
struct gsc_t {{ gacc_t r[N_PAIRS], i[N_PAIRS]; }};
struct c_sc_t {{ cq_t r[K][K], i[K][K]; }};
"""

    def precoder_cpp(self):
        a, fs = self.a, self.fs
        s = self.preamble() + self.math.source(a) + DenseGen.templates(a)
        s += fs.normalize() + self.gram.source(a, 1) + fs.init()
        for i, (p, f) in enumerate(self.blocks):
            s += f"\n// Les 16 tokens quantifies d'une sous-porteuse (un element de stream <= 4096 bits).\n"
            s += f"struct qa_sc_{i} {{ {p.qa.ctype} v[NT][D]; }};\nstruct qb_sc_{i} {{ {p.qb.ctype} v[NT][D]; }};\n"
            s += fs.layernorm(i, p.ln) + fs.proj(i, p.da, p.qa, "a") + fs.proj(i, p.db, p.qb, "b") + fs.tri(i, f) + fs.ffn(i, f)
        s += fs.read() + CombineGen(self.W).source(a)
        return s + "\n" + self.top()

    def top(self):
        a, nl = self.a, len(self.blocks)
        s = ("void pair_precoder(const mant_t mr[NSC][K][M], const mant_t mi[NSC][K][M], const ex_t ex[NSC][K],\n"
             "                   const act_t snr10[NSC], act_t w_re[NSC][M][K], act_t w_im[NSC][M][K]) {\n#pragma HLS DATAFLOW\n")
        for v in ("mr", "mi"):
            s += a.partition(v, 2) + a.partition(v, 3, a.ant)
        s += a.partition("ex", 2)
        for v in ("w_re", "w_im"):
            s += a.partition(v, 3) + a.partition(v, 2, a.ant)
        ng = 64 // a.ant
        streams = [("mgrp_t", "m_s", 64), ("n2_sc_t", "n2_s", 2), ("ri_sc_t", "ri_s", 4), ("pdb_sc_t", "pdb_s", 4),
                   ("hgrp_t", "h_g", 64), ("hgrp_t", "h_w", ng * a.nsc), ("gsc_t", "g_s", 2), ("ef_t", "ef_s", 32),
                   ("c_sc_t", "c_s", 1), ("wgrp_t", "w_s", 64), ("rip_t", "rip_s", 2)]
        # tokens larges (600-1300 bits) : profondeur 2 (FIFO en registres) ; blocs par sous-porteuse (2 304 bits) :
        # profondeur 1 (le consommateur attend le bloc) ; relais x -> tri et e -> f2 : >= 16 tokens (attente du bloc) -> 32
        # tokens larges (600-1300 bits) : profondeur 1 (registre + poignee de main, pas de multiplexeur ; un token
        # toutes les S cycles, lu avant l'ecriture suivante) ; relais x -> tri et e -> f2 : >= NT tokens d'attente -> 32
        streams += [("tok_t", f"e{c}", 1) for c in range(nl + 1)]
        for i in range(nl):
            streams += [("ln_t", f"l{i}", 1), ("x_t", f"xa{i}", 1), ("x_t", f"xb{i}", 1), ("x_t", f"xt{i}", 32),
                        ("tok_t", f"er{i}", 32), (f"qa_sc_{i}", f"ar{i}", 1), (f"qa_sc_{i}", f"ai{i}", 1),
                        (f"qb_sc_{i}", f"br{i}", 1), (f"qb_sc_{i}", f"bi{i}", 1), ("f_t", f"f{i}", 1), ("h_t", f"h{i}", 1)]
        for typ, name, depth in streams:
            s += f"    hls::stream<{typ}> {name};\n#pragma HLS STREAM variable={name} depth={depth}\n"
        s += ("    normalize_acc(mr, mi, ex, m_s, n2_s);\n    normalize_fin(n2_s, ri_s, pdb_s);\n"
              "    normalize_scale(m_s, ri_s, h_g, h_w);\n    gram_acc(h_g, g_s);\n    gram_edges(g_s, ef_s);\n"
              "    dense_init(ef_s, snr10, pdb_s, e0);\n")
        for i in range(nl):
            s += (f"    ln1_{i}(e{i}, l{i});\n    ln2_{i}(l{i}, xa{i}, xb{i}, xt{i}, er{i});\n"
                  f"    dense_a{i}(xa{i}, ar{i}, ai{i});\n    dense_b{i}(xb{i}, br{i}, bi{i});\n"
                  f"    tri_{i}(xt{i}, ar{i}, ai{i}, br{i}, bi{i}, f{i});\n"
                  f"    dense_f1{i}(f{i}, h{i});\n    dense_f2{i}(h{i}, er{i}, e{i + 1});\n")
        s += f"    dense_read(e{nl}, c_s);\n    combine_acc(h_w, c_s, w_s, rip_s);\n    combine_scale(w_s, rip_s, w_re, w_im);\n}}\n"
        return s
