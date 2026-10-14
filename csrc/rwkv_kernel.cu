/*
 * rwkv_kernel.cu — CUDA-accelerated RWKV training and inference.
 *
 * Canonical GPU path for the RWKV arch (tinylm/models_linear.py).
 * The legacy MLP path lives in accel_cuda.py / csrc/train_kernel_cuda.cu.
 *
 * Compatible with GTX 970+ (compute capability 5.2+, Maxwell architecture).
 * Double precision throughout; global atomics use CAS (no native sm_52
 * double atomicAdd). One device thread per training window (grid-stride),
 * per-window scratch in global memory (no shared-memory size limits, no
 * large per-thread stack arrays).
 *
 * Flat param layout (matches RWKVModel.params order):
 *   C          : vocab * emb (embeddings)
 *   in_proj    : hidden * emb + hidden bias (only when emb != hidden)
 *   blocks[i]  : log_w[H], log_k[H], W_out[H*H], b_out[H],
 *                ln1_w[H], ln1_b[H], ffn_W1[(4H)*H], ffn_b1[4H],
 *                ffn_W2[H*(4H)], ffn_b2[H], ln2_w[H], ln2_b[H], lora[H]
 *                (lora is unused in forward; its grads stay zero)
 *   out_proj   : vocab * hidden (output projection)
 *   out_bias   : vocab
 *
 * Forward/backward mirror the Value-autograd math in models_linear.py,
 * including the no-sqrt layer norm y=(x-mean)*w/(var+eps)+b and the
 * simplified GELU h*(0.5*(1+tanh(0.7978845608028654*h))). Sampled loss
 * projects the final hidden state through candidate rows only (target
 * first, then negatives). WKV max/exp-scale terms are constants (no
 * gradient), exactly like the Python graph.
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_52 -shared -Xcompiler -fPIC \
 *        -o libtinylm_rwkv.so rwkv_kernel.cu -lcublas -lcurand
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define RWKV_GELU_C 0.7978845608028654
#define RWKV_LN_EPS 1e-5
#define RWKV_WKV_EPS 1e-8

/* atomicAdd for double (not natively supported on compute capability < 6.0) */
__device__ double rwkv_atomic_add(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* xorshift64 RNG for device (uniform negatives fallback, mirrors legacy kernel) */
__device__ uint64_t rwkv_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* ---- flat-layout helpers ------------------------------------------------ */

__host__ __device__ static size_t rwkv_block_size(int hidden) {
    size_t H = (size_t)hidden;
    return 9u * H * H + 13u * H;
}

__host__ __device__ static size_t rwkv_n_params(int vocab, int emb, int hidden,
                                               int n_layers, int has_in) {
    size_t V = (size_t)vocab, E = (size_t)emb, H = (size_t)hidden;
    size_t n = V * E;
    if (has_in) n += H * E + H;
    n += (size_t)n_layers * rwkv_block_size(hidden);
    n += V * H + V;
    return n;
}

/* Offsets of block tensors relative to the block base. */
struct RWKVBlockOff {
    size_t log_w, log_k, W_out, b_out, ln1_w, ln1_b;
    size_t ffn_W1, ffn_b1, ffn_W2, ffn_b2, ln2_w, ln2_b;
};

__host__ __device__ static struct RWKVBlockOff rwkv_block_off(int hidden) {
    struct RWKVBlockOff o;
    size_t H = (size_t)hidden, p = 0;
    o.log_w = p; p += H;
    o.log_k = p; p += H;
    o.W_out = p; p += H * H;
    o.b_out = p; p += H;
    o.ln1_w = p; p += H;
    o.ln1_b = p; p += H;
    o.ffn_W1 = p; p += 4u * H * H;
    o.ffn_b1 = p; p += 4u * H;
    o.ffn_W2 = p; p += 4u * H * H;
    o.ffn_b2 = p; p += H;
    o.ln2_w = p; p += H;
    o.ln2_b = p; p += H;
    return o;
}

/* In-place no-sqrt layer-norm backward. g holds dL/dy on entry and dL/dx
 * on exit; dw/db grads are atomically accumulated. Mirrors the Value graph
 * in RWKVBlock.layer_norm (mean/var are graph nodes, eps is a constant). */
__device__ void rwkv_ln_backward(double *g, const double *x, const double *w,
                                 double D, int H, double *dw_acc,
                                 double *db_acc) {
    double invD = 1.0 / D;
    double mean = 0.0;
    for (int i = 0; i < H; i++) mean += x[i];
    mean /= (double)H;
    double s1 = 0.0;
    for (int i = 0; i < H; i++) {
        double d = x[i] - mean;
        s1 += g[i] * d * w[i];
        rwkv_atomic_add(dw_acc + i, g[i] * invD * d);
        rwkv_atomic_add(db_acc + i, g[i]);
    }
    double ds = -s1 / (D * D) * (2.0 / (double)H);
    double S = 0.0;
    for (int i = 0; i < H; i++) {
        double d = x[i] - mean;
        double dd = g[i] * w[i] * invD + ds * d;
        g[i] = dd;
        S += dd;
    }
    double adj = -S / (double)H;
    for (int i = 0; i < H; i++) g[i] += adj;
}

/* Doubles of scratch per layer / per window (see rwkv_window_grads). */
__host__ __device__ static size_t rwkv_per_layer_d(int T, int H) {
    size_t F = 4u * (size_t)H;
    return 12u * (size_t)T * H + 2u * (size_t)T * F + 2u * (size_t)T;
}

__host__ __device__ static size_t rwkv_per_window_d(int T, int H, int L, int NC) {
    size_t F = 4u * (size_t)H;
    return (size_t)T * H + (size_t)L * rwkv_per_layer_d(T, H) + (size_t)NC
         + 6u * (size_t)H + 2u * F + 2u * (size_t)L * H;
}

/* ---- per-window forward + backward ---------------------------------------
 * Returns 0 on success, negative when a token id is out of range (caller
 * skips the window: no loss added, no grads written).
 *
 * Scratch layout per window (doubles), H=hidden, T=ctx, L=n_layers, F=4H:
 *   x_emb   : T*H
 *   layers  : L * (12 H-arrays of T*H: ln1,k,num,den,mx,e1,e2,aout,x1,ln2,
 *             fout,xout; then hpre,hact of T*F; then D1,D2 of T scalars)
 *   logit   : NC
 *   tmps    : gcur[H], dhact[F], dhpre[F], dln[H], dwkv[H], ddx[H]
 *   carry   : gdnum[H], gdden[H] per layer (2*H*L)
 * Ints: cand (NC).
 */
__device__ int rwkv_window_grads(const double *params, const int *win,
                                 int T, int V, int E, int H, int L, int has_in,
                                 int n_neg, uint64_t *rng, const int *negs,
                                 double *s, int *cand,
                                 double *dC, double *dIP, double *dIB,
                                 const double *blk, double *dblk, size_t blk_sz,
                                 const double *OP, const double *OB,
                                 double *dOP, double *dOB,
                                 double *loss_out) {
    const int NC = n_neg + 1;
    const size_t F = 4u * (size_t)H;
    const size_t per_layer = rwkv_per_layer_d(T, H);
    struct RWKVBlockOff bo = rwkv_block_off(H);

    const int target = win[T];
    if (target < 0 || target >= V) return -1;
    for (int t = 0; t < T; t++)
        if (win[t] < 0 || win[t] >= V) return -1;

    /* carve scratch */
    double *x_emb = s; s += (size_t)T * H;
    double *layers = s; s += (size_t)L * per_layer;
    double *logit = s; s += NC;
    double *gcur = s; s += H;
    double *dhact = s; s += F;
    double *dhpre = s; s += F;
    double *dln = s; s += H;
    double *dwkv = s; s += H;
    double *ddx = s; s += H;
    double *carry = s; /* 2*H*L: gdnum then gdden per layer */

    /* candidates: target first, then provided negatives or uniform samples */
    cand[0] = target;
    for (int j = 1; j < NC; j++) {
        int c = negs ? negs[j - 1] : (int)(rwkv_rand(rng) % (uint64_t)V);
        if (c < 0 || c >= V) return -1;
        cand[j] = c;
    }

    const double *C = params;
    const double *IP = has_in ? C + (size_t)V * E : NULL;
    const double *IB = has_in ? IP + (size_t)H * E : NULL;

    /* ---- forward ---- */
    for (int p = 0; p < T; p++) {
        const double *erow = C + (size_t)win[p] * E;
        double *xe = x_emb + (size_t)p * H;
        if (has_in) {
            for (int i = 0; i < H; i++) {
                const double *row = IP + (size_t)i * E;
                double acc = IB[i];
                for (int j = 0; j < E; j++) acc += row[j] * erow[j];
                xe[i] = acc;
            }
        } else {
            for (int i = 0; i < H; i++) xe[i] = erow[i];
        }
        for (int l = 0; l < L; l++) {
            double *lb = layers + (size_t)l * per_layer;
            double *Ln1 = lb + (size_t)p * H;
            double *KK = lb + (size_t)T * H + (size_t)p * H;
            double *NN = lb + (size_t)2 * T * H + (size_t)p * H;
            double *DN = lb + (size_t)3 * T * H + (size_t)p * H;
            double *MX = lb + (size_t)4 * T * H + (size_t)p * H;
            double *E1 = lb + (size_t)5 * T * H + (size_t)p * H;
            double *E2 = lb + (size_t)6 * T * H + (size_t)p * H;
            double *AO = lb + (size_t)7 * T * H + (size_t)p * H;
            double *X1 = lb + (size_t)8 * T * H + (size_t)p * H;
            double *L2 = lb + (size_t)9 * T * H + (size_t)p * H;
            double *FO = lb + (size_t)10 * T * H + (size_t)p * H;
            double *XO = lb + (size_t)11 * T * H + (size_t)p * H;
            double *HP = lb + (size_t)12 * T * H + (size_t)p * F;
            double *HA = HP + (size_t)T * F;
            double *DD = lb + (size_t)12 * T * H + 2u * (size_t)T * F;

            const double *xin = (l == 0)
                ? xe
                : (layers + (size_t)(l - 1) * per_layer + (size_t)11 * T * H + (size_t)p * H);

            const double *bp = blk + (size_t)l * blk_sz;

            /* layer norm 1 (no sqrt) */
            {
                double m = 0.0;
                for (int i = 0; i < H; i++) m += xin[i];
                m /= (double)H;
                double v = 0.0;
                for (int i = 0; i < H; i++) { double d = xin[i] - m; v += d * d; }
                v /= (double)H;
                double D = v + RWKV_LN_EPS;
                DD[p] = D;
                const double *w = bp + bo.ln1_w, *b = bp + bo.ln1_b;
                for (int i = 0; i < H; i++)
                    Ln1[i] = (xin[i] - m) * w[i] / D + b[i];
            }
            /* WKV recurrence + attention output projection */
            {
                const double *lw = bp + bo.log_w, *lk = bp + bo.log_k;
                const double *Wo = bp + bo.W_out, *bob = bp + bo.b_out;
                for (int i = 0; i < H; i++) {
                    double pn, pd, pm;
                    if (p == 0) { pn = 0.0; pd = 0.0; pm = -1e30; }
                    else { pn = NN[i - H]; pd = DN[i - H]; pm = MX[i - H]; }
                    double a = exp(lw[i]);
                    double bb = exp(-a);
                    double ew = exp(-bb);
                    double kval = exp(lk[i]) * Ln1[i];
                    double nmax = pm > kval ? pm : kval;
                    double q1 = exp(pm - nmax);
                    double q2 = exp(kval - nmax);
                    double nn = pn * ew * q1 + Ln1[i] * kval * q2;
                    double dd = pd * ew * q1 + kval * q2;
                    KK[i] = kval; NN[i] = nn; DN[i] = dd; MX[i] = nmax;
                    E1[i] = q1; E2[i] = q2;
                }
                for (int i = 0; i < H; i++) {
                    const double *orow = Wo + (size_t)i * H;
                    double acc = bob[i];
                    for (int j = 0; j < H; j++)
                        acc += orow[j] * (NN[j] / (DN[j] + RWKV_WKV_EPS));
                    AO[i] = acc;
                }
            }
            /* residual 1 + layer norm 2 */
            {
                for (int i = 0; i < H; i++) X1[i] = xin[i] + AO[i];
                double m = 0.0;
                for (int i = 0; i < H; i++) m += X1[i];
                m /= (double)H;
                double v = 0.0;
                for (int i = 0; i < H; i++) { double d = X1[i] - m; v += d * d; }
                v /= (double)H;
                double D = v + RWKV_LN_EPS;
                DD[T + p] = D;
                const double *w = bp + bo.ln2_w, *b = bp + bo.ln2_b;
                for (int i = 0; i < H; i++)
                    L2[i] = (X1[i] - m) * w[i] / D + b[i];
            }
            /* FFN + residual 2 */
            {
                const double *W1 = bp + bo.ffn_W1, *b1 = bp + bo.ffn_b1;
                const double *W2 = bp + bo.ffn_W2, *b2 = bp + bo.ffn_b2;
                for (int j = 0; j < (int)F; j++) {
                    const double *row = W1 + (size_t)j * H;
                    double acc = b1[j];
                    for (int i = 0; i < H; i++) acc += row[i] * L2[i];
                    HP[j] = acc;
                    HA[j] = acc * 0.5 * (1.0 + tanh(acc * RWKV_GELU_C));
                }
                for (int i = 0; i < H; i++) {
                    const double *row = W2 + (size_t)i * F;
                    double acc = b2[i];
                    for (int j = 0; j < (int)F; j++) acc += row[j] * HA[j];
                    FO[i] = acc;
                    XO[i] = X1[i] + acc;
                }
            }
        }
    }

    /* ---- sampled head (final hidden state through candidate rows) ---- */
    {
        double *XO = layers + (size_t)(L - 1) * per_layer
                   + (size_t)11 * T * H + (size_t)(T - 1) * H;
        double m = -1e308;
        for (int j = 0; j < NC; j++) {
            const double *row = OP + (size_t)cand[j] * H;
            double acc = OB[cand[j]];
            for (int i = 0; i < H; i++) acc += row[i] * XO[i];
            logit[j] = acc;
            if (acc > m) m = acc;
        }
        double ssum = 0.0;
        for (int j = 0; j < NC; j++) ssum += exp(logit[j] - m);
        *loss_out = -(logit[0] - m - log(ssum));
        for (int j = 0; j < NC; j++) logit[j] = exp(logit[j] - m) / ssum;
        logit[0] -= 1.0;
    }

    /* ---- backward ---- */
    for (int l = 0; l < L; l++)
        for (int i = 0; i < H; i++) {
            carry[(size_t)l * H + i] = 0.0;
            carry[(size_t)L * H + (size_t)l * H + i] = 0.0;
        }

    /* head grads -> final hidden (gcur) + candidate out-proj rows */
    {
        double *XO = layers + (size_t)(L - 1) * per_layer
                   + (size_t)11 * T * H + (size_t)(T - 1) * H;
        for (int i = 0; i < H; i++) gcur[i] = 0.0;
        for (int j = 0; j < NC; j++) {
            double g = logit[j];
            const double *row = OP + (size_t)cand[j] * H;
            double *grow = dOP + (size_t)cand[j] * H;
            for (int i = 0; i < H; i++) {
                rwkv_atomic_add(grow + i, g * XO[i]);
                gcur[i] += g * row[i];
            }
            rwkv_atomic_add(dOB + cand[j], g);
        }
    }

    for (int p = T - 1; p >= 0; p--) {
        /* gcur = dL/d(top-layer output at p): only the final position feeds
           the sampled head, so clear stale grads for every earlier one. */
        if (p != T - 1)
            for (int i = 0; i < H; i++) gcur[i] = 0.0;
        for (int l = L - 1; l >= 0; l--) {
            double *lb = layers + (size_t)l * per_layer;
            double *L1 = lb + (size_t)p * H;
            double *KK = lb + (size_t)T * H + (size_t)p * H;
            double *NN = lb + (size_t)2 * T * H + (size_t)p * H;
            double *DN = lb + (size_t)3 * T * H + (size_t)p * H;
            double *E1 = lb + (size_t)5 * T * H + (size_t)p * H;
            double *E2 = lb + (size_t)6 * T * H + (size_t)p * H;
            double *X1 = lb + (size_t)8 * T * H + (size_t)p * H;
            double *L2 = lb + (size_t)9 * T * H + (size_t)p * H;
            double *HP = lb + (size_t)12 * T * H + (size_t)p * F;
            double *HA = HP + (size_t)T * F;
            double *DD = lb + (size_t)12 * T * H + 2u * (size_t)T * F;
            double D1 = DD[p], D2 = DD[T + p];

            const double *xin = (l == 0)
                ? (x_emb + (size_t)p * H)
                : (layers + (size_t)(l - 1) * per_layer + (size_t)11 * T * H + (size_t)p * H);

            const double *bp = blk + (size_t)l * blk_sz;
            double *db = dblk + (size_t)l * blk_sz;
            double *gcn = carry + (size_t)l * H;
            double *gcd = carry + (size_t)L * H + (size_t)l * H;

            /* xout = x1 + fout: dL/dx1 gets gcur; dL/dfout = gcur.
               FFN: dhact = W2^T gcur; dW2/db2; GELU -> dhpre;
               dln2 = W1^T dhpre; dW1/db1. */
            {
                const double *W2 = bp + bo.ffn_W2;
                for (int j = 0; j < (int)F; j++) {
                    double acc = 0.0;
                    for (int i = 0; i < H; i++)
                        acc += W2[(size_t)i * F + j] * gcur[i];
                    dhact[j] = acc;
                }
                for (int i = 0; i < H; i++) {
                    double g = gcur[i];
                    double *grow = db + bo.ffn_W2 + (size_t)i * F;
                    for (int j = 0; j < (int)F; j++)
                        rwkv_atomic_add(grow + j, g * HA[j]);
                    rwkv_atomic_add(db + bo.ffn_b2 + i, g);
                }
                for (int j = 0; j < (int)F; j++) {
                    double h = HP[j];
                    double t = tanh(h * RWKV_GELU_C);
                    double dq = dhact[j];
                    dhpre[j] = dq * (0.5 * (1.0 + t)
                                     + h * (0.5 * (1.0 - t * t) * RWKV_GELU_C));
                }
                const double *W1 = bp + bo.ffn_W1;
                for (int i = 0; i < H; i++) {
                    double acc = 0.0;
                    for (int j = 0; j < (int)F; j++)
                        acc += W1[(size_t)j * H + i] * dhpre[j];
                    dln[i] = acc;
                }
                for (int j = 0; j < (int)F; j++) {
                    double g = dhpre[j];
                    double *grow = db + bo.ffn_W1 + (size_t)j * H;
                    for (int i = 0; i < H; i++)
                        rwkv_atomic_add(grow + i, g * L2[i]);
                    rwkv_atomic_add(db + bo.ffn_b1 + j, g);
                }
            }
            /* ln2 backward (in place: dln = dL/dx1 via ln2), then add the
               direct residual grad: dx1 = gcur + that. */
            rwkv_ln_backward(dln, X1, bp + bo.ln2_w, D2, H,
                             db + bo.ln2_w, db + bo.ln2_b);
            for (int i = 0; i < H; i++) ddx[i] = gcur[i] + dln[i];
            /* stash the x1 total for the xin residual path below (dhpre is
               free: the FFN loops above are done with it) */
            for (int i = 0; i < H; i++) dhpre[i] = ddx[i];

            /* attn out-proj: dwkv = Wout^T ddx; dWout/dbout. */
            {
                const double *Wo = bp + bo.W_out;
                for (int j = 0; j < H; j++) {
                    double acc = 0.0;
                    for (int i = 0; i < H; i++)
                        acc += Wo[(size_t)i * H + j] * ddx[i];
                    dwkv[j] = acc;
                }
                for (int i = 0; i < H; i++) {
                    double g = ddx[i];
                    double *grow = db + bo.W_out + (size_t)i * H;
                    for (int j = 0; j < H; j++) {
                        double wkv = NN[j] / (DN[j] + RWKV_WKV_EPS);
                        rwkv_atomic_add(grow + j, g * wkv);
                    }
                    rwkv_atomic_add(db + bo.b_out + i, g);
                }
            }
            /* WKV step backward (reverse time inside the p loop via carry). */
            {
                const double *lw = bp + bo.log_w, *lk = bp + bo.log_k;
                double *dlw = db + bo.log_w, *dlk = db + bo.log_k;
                for (int i = 0; i < H; i++) ddx[i] = 0.0;
                for (int i = 0; i < H; i++) {
                    double g = dwkv[i];
                    double nn = NN[i], dd = DN[i];
                    double q1 = E1[i], q2 = E2[i], kval = KK[i], x = L1[i];
                    double pn = (p > 0) ? NN[i - H] : 0.0;
                    double pd = (p > 0) ? DN[i - H] : 0.0;
                    double invD = 1.0 / (dd + RWKV_WKV_EPS);
                    double dnp = g * invD + gcn[i];
                    double ddp = -g * nn * invD * invD + gcd[i];
                    double a = exp(lw[i]);
                    double bb = exp(-a);
                    double ew = exp(-bb);
                    double dew = q1 * (dnp * pn + ddp * pd);
                    double dlwi = dew * ew * bb * a;
                    rwkv_atomic_add(dlw + i, dlwi);
                    double dk = q2 * (dnp * x + ddp);
                    rwkv_atomic_add(dlk + i, dk * kval);
                    double elk = exp(lk[i]);
                    ddx[i] = dk * elk + dnp * kval * q2;
                    gcn[i] = dnp * ew * q1;
                    gcd[i] = ddp * ew * q1;
                }
            }
            /* ln1 backward (in place: ddx = dL/dxin via ln1). The block
               input also feeds the x1 = xin + aout residual directly, so
               add the saved dx1 total (stashed in dhpre, free by now). */
            rwkv_ln_backward(ddx, xin, bp + bo.ln1_w, D1, H,
                             db + bo.ln1_w, db + bo.ln1_b);
            for (int i = 0; i < H; i++) gcur[i] = ddx[i] + dhpre[i];
        }
        /* embedding of token p */
        {
            int tok = win[p];
            if (has_in) {
                for (int i = 0; i < H; i++) {
                    double g = gcur[i];
                    double *grow = dIP + (size_t)i * E;
                    const double *erow = C + (size_t)tok * E;
                    const double *prow = IP + (size_t)i * E;
                    for (int j = 0; j < E; j++) {
                        rwkv_atomic_add(grow + j, g * erow[j]);
                        rwkv_atomic_add(dC + (size_t)tok * E + j, g * prow[j]);
                    }
                    rwkv_atomic_add(dIB + i, g);
                }
            } else {
                for (int i = 0; i < H; i++)
                    rwkv_atomic_add(dC + (size_t)tok * E + i, gcur[i]);
            }
        }
    }
    return 0;
}

/* ---- batch kernels ------------------------------------------------------- */

__global__ void rwkv_batch_kernel(const double *params, const int *windows,
                                  const int *negatives,
                                  int batch, int T, int V, int E, int H, int L,
                                  int has_in, int n_neg, uint64_t seed,
                                  double *out_loss, double *out_grads,
                                  double *scratch, int *iscratch,
                                  size_t per_win) {
    size_t n_params = rwkv_n_params(V, E, H, L, has_in);
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_params; i += blockDim.x * gridDim.x) {
        out_grads[i] = 0.0;
    }
    __syncthreads();

    struct RWKVBlockOff bo = rwkv_block_off(H);
    (void)bo;
    size_t VEs = (size_t)V * E;
    size_t in_sz = has_in ? (size_t)H * E + H : 0;
    const double *blk = params + VEs + in_sz;
    double *dblk = out_grads + VEs + in_sz;
    size_t blk_sz = rwkv_block_size(H);
    const double *OP = blk + (size_t)L * blk_sz;
    const double *OB = OP + (size_t)V * H;
    double *dOP = dblk + (size_t)L * blk_sz;
    double *dOB = dOP + (size_t)V * H;
    double *dC = out_grads;
    double *dIP = has_in ? out_grads + VEs : NULL;
    double *dIB = has_in ? dIP + (size_t)H * E : NULL;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nthreads = blockDim.x * gridDim.x;
    for (int w = (int)tid; w < batch; w += (int)nthreads) {
        const int *win = windows + (size_t)w * (T + 1);
        const int *negs = negatives ? negatives + (size_t)w * n_neg : NULL;
        double wl = 0.0;
        uint64_t rng = (seed ^ 0x9E3779B97F4A7C15ULL) + (uint64_t)w * 0x9E3779B97F4A7C15ULL;
        if (rng == 0) rng = 0x9E3779B97F4A7C15ULL;
        int rc = rwkv_window_grads(params, win, T, V, E, H, L, has_in, n_neg,
                                   &rng, negs, scratch + (size_t)w * per_win,
                                   iscratch + (size_t)w * (n_neg + 1),
                                   dC, dIP, dIB, blk, dblk, blk_sz,
                                   OP, OB, dOP, dOB, &wl);
        if (rc == 0) rwkv_atomic_add(out_loss, wl);
    }
}

__global__ void rwkv_divide_loss(double *loss, int batch) {
    *loss /= (double)batch;
}

/* sum of squared batch-AVERAGED grads into *total (for global-norm clip).
 * Averages inside the square to mirror clip_grads() summation order. */
__global__ void rwkv_norm_kernel(const double *grads, size_t n, int batch,
                                 double *total) {
    __shared__ double sh[256];
    double b = (double)batch;
    double acc = 0.0;
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n; i += blockDim.x * gridDim.x) {
        double g = grads[i] / b;
        acc += g * g;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) rwkv_atomic_add(total, sh[0]);
}

/* average grads, apply clip scale, SGD+momentum update (matches train() MCU) */
__global__ void rwkv_update_kernel(double *params, double *grads, double *vel,
                                   size_t n, int batch, double lr,
                                   double momentum, double clip_scale) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n; i += blockDim.x * gridDim.x) {
        double g = grads[i] / (double)batch * clip_scale;
        if (momentum > 0.0) {
            double v = momentum * vel[i] + g;
            vel[i] = v;
            params[i] -= lr * v;
        } else {
            params[i] -= lr * g;
        }
    }
}

/* ---- persistent velocity -------------------------------------------------- */

static double *rwkv_vel = NULL;
static size_t rwkv_vel_len = 0;

extern "C" {

int rwkv_init_buffers(int vocab, int emb, int hidden, int n_layers,
                      int has_in, int ctx, int batch_size) {
    (void)ctx; (void)batch_size;
    size_t n = rwkv_n_params(vocab, emb, hidden, n_layers, has_in);
    if (rwkv_vel && rwkv_vel_len == n) return 0;
    if (rwkv_vel) { cudaFree(rwkv_vel); rwkv_vel = NULL; rwkv_vel_len = 0; }
    if (cudaMalloc(&rwkv_vel, n * sizeof(double)) != cudaSuccess) return -3;
    if (cudaMemset(rwkv_vel, 0, n * sizeof(double)) != cudaSuccess) return -3;
    rwkv_vel_len = n;
    return 0;
}

void rwkv_reset_state() {
    if (rwkv_vel) cudaMemset(rwkv_vel, 0, rwkv_vel_len * sizeof(double));
}

int rwkv_batch_loss_grads(const double *params, const int *windows,
                          const int *negatives,
                          int batch, int ctx, int vocab, int emb, int hidden,
                          int n_layers, int has_in, int n_negatives,
                          unsigned long long seed,
                          double *out_loss, double *out_grads,
                          double max_norm) {
    (void)max_norm; /* informational; clipping happens on the host */
    if (!params || !windows || !out_loss || !out_grads) return -1;
    if (batch < 1 || ctx < 1 || vocab < 1 || emb < 1 || hidden < 1) return -1;
    if (n_layers < 1 || n_negatives < 0) return -1;
    size_t n_params = rwkv_n_params(vocab, emb, hidden, n_layers, has_in);
    size_t per_win = rwkv_per_window_d(ctx, hidden, n_layers, n_negatives + 1);

    int *d_windows = NULL;
    if (cudaMalloc(&d_windows, (size_t)batch * (ctx + 1) * sizeof(int)) != cudaSuccess)
        return -3;
    if (cudaMemcpy(d_windows, windows, (size_t)batch * (ctx + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_windows);
        return -3;
    }
    int *d_negatives = NULL;
    if (negatives && n_negatives > 0) {
        if (cudaMalloc(&d_negatives, (size_t)batch * n_negatives * sizeof(int)) != cudaSuccess) {
            cudaFree(d_windows);
            return -3;
        }
        if (cudaMemcpy(d_negatives, negatives, (size_t)batch * n_negatives * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_windows);
            cudaFree(d_negatives);
            return -3;
        }
    }
    double *d_params = NULL, *d_grads = NULL, *d_loss = NULL, *d_scratch = NULL;
    int *d_cand = NULL;
    if (cudaMalloc(&d_params, n_params * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_grads, n_params * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_loss, sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_scratch, (size_t)batch * per_win * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_cand, (size_t)batch * (n_negatives + 1) * sizeof(int)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_scratch); cudaFree(d_cand);
        return -3;
    }
    if (cudaMemcpy(d_params, params, n_params * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemset(d_loss, 0, sizeof(double)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_scratch); cudaFree(d_cand);
        return -3;
    }
    int nthreads = batch < 256 ? batch : 256;
    int nblocks = (batch + nthreads - 1) / nthreads;
    if (nblocks > 16) nblocks = 16;
    rwkv_batch_kernel<<<nblocks, nthreads>>>(
        d_params, d_windows, d_negatives, batch, ctx, vocab, emb, hidden,
        n_layers, has_in, n_negatives, seed, d_loss, d_grads,
        d_scratch, d_cand, per_win);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RWKV batch grads launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_scratch); cudaFree(d_cand);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "RWKV batch grads execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_scratch); cudaFree(d_cand);
        return -1;
    }
    if (cudaMemcpy(out_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(out_grads, d_grads, n_params * sizeof(double),
                          cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_scratch); cudaFree(d_cand);
        return -3;
    }
    cudaFree(d_windows);
    if (d_negatives) cudaFree(d_negatives);
    cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
    cudaFree(d_scratch); cudaFree(d_cand);
    return 0;
}

int rwkv_train_step(double *params, const int *windows, const int *negatives,
                    int batch, int ctx, int vocab, int emb, int hidden,
                    int n_layers, int has_in, int n_negatives,
                    double lr, double momentum, unsigned long long seed,
                    double *out_loss, double max_norm) {
    if (!params || !windows || !out_loss) return -1;
    if (batch < 1 || ctx < 1 || vocab < 1 || emb < 1 || hidden < 1) return -1;
    if (n_layers < 1 || n_negatives < 0) return -1;
    size_t n_params = rwkv_n_params(vocab, emb, hidden, n_layers, has_in);
    size_t per_win = rwkv_per_window_d(ctx, hidden, n_layers, n_negatives + 1);

    if (rwkv_init_buffers(vocab, emb, hidden, n_layers, has_in, ctx, batch) != 0)
        return -3;

    int *d_windows = NULL;
    if (cudaMalloc(&d_windows, (size_t)batch * (ctx + 1) * sizeof(int)) != cudaSuccess)
        return -3;
    if (cudaMemcpy(d_windows, windows, (size_t)batch * (ctx + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_windows);
        return -3;
    }
    int *d_negatives = NULL;
    if (negatives && n_negatives > 0) {
        if (cudaMalloc(&d_negatives, (size_t)batch * n_negatives * sizeof(int)) != cudaSuccess) {
            cudaFree(d_windows);
            return -3;
        }
        if (cudaMemcpy(d_negatives, negatives, (size_t)batch * n_negatives * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_windows);
            cudaFree(d_negatives);
            return -3;
        }
    }
    double *d_params = NULL, *d_grads = NULL, *d_loss = NULL, *d_scratch = NULL;
    double *d_norm = NULL;
    int *d_cand = NULL;
    if (cudaMalloc(&d_params, n_params * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_grads, n_params * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_loss, sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_norm, sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_scratch, (size_t)batch * per_win * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_cand, (size_t)batch * (n_negatives + 1) * sizeof(int)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
        return -3;
    }
    if (cudaMemcpy(d_params, params, n_params * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemset(d_loss, 0, sizeof(double)) != cudaSuccess
            || cudaMemset(d_norm, 0, sizeof(double)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
        return -3;
    }

    int nthreads = batch < 256 ? batch : 256;
    int nblocks = (batch + nthreads - 1) / nthreads;
    if (nblocks > 16) nblocks = 16;
    rwkv_batch_kernel<<<nblocks, nthreads>>>(
        d_params, d_windows, d_negatives, batch, ctx, vocab, emb, hidden,
        n_layers, has_in, n_negatives, seed, d_loss, d_grads,
        d_scratch, d_cand, per_win);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "RWKV train grads failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
        return -1;
    }
    rwkv_divide_loss<<<1, 1>>>(d_loss, batch);

    /* global-norm clip after averaging (matches train()/C semantics) */
    double clip_scale = 1.0;
    if (max_norm > 0.0) {
        int nt = 256, nb = 16;
        rwkv_norm_kernel<<<nb, nt>>>(d_grads, n_params, batch, d_norm);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(d_windows);
            if (d_negatives) cudaFree(d_negatives);
            cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
            cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
            return -1;
        }
        double sumsq = 0.0;
        if (cudaMemcpy(&sumsq, d_norm, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(d_windows);
            if (d_negatives) cudaFree(d_negatives);
            cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
            cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
            return -3;
        }
        /* grads are averaged on-device inside the norm (mirrors
           clip_grads); the +1e-6 matches its clip coefficient exactly. */
        double norm = sqrt(sumsq);
        if (norm > max_norm && norm > 0.0) clip_scale = max_norm / (norm + 1e-6);
    }
    rwkv_update_kernel<<<16, 256>>>(d_params, d_grads, rwkv_vel, n_params,
                                    batch, lr, momentum, clip_scale);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "RWKV train update failed\n");
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
        return -1;
    }
    if (cudaMemcpy(params, d_params, n_params * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(out_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
        cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
        return -3;
    }
    cudaFree(d_windows);
    if (d_negatives) cudaFree(d_negatives);
    cudaFree(d_params); cudaFree(d_grads); cudaFree(d_loss);
    cudaFree(d_norm); cudaFree(d_scratch); cudaFree(d_cand);
    return 0;
}

/* Forward-only logits for one sequence (mirrors fast_logits, CPU-cheap
 * shapes run on a single block with hidden-strided threads). */
int rwkv_generate_step(const double *params, const int *input_ids, int seq_len,
                       int vocab, int emb, int hidden, int n_layers,
                       int has_in, int ctx,
                       unsigned long long seed, double *out_logits,
                       double *states, int max_seq_len) {
    (void)ctx; (void)seed; (void)states; (void)max_seq_len;
    if (!params || !input_ids || !out_logits) return -1;
    if (seq_len < 1 || vocab < 1 || emb < 1 || hidden < 1 || n_layers < 1) return -1;
    /* Host-side forward (exact float order, small shapes): the GPU win for
       single-sequence generation is negative vs CPU once transfers count,
       so run the same math on the host and keep one code path. Device
       offload for batched generation is future work. */
    size_t VEs = (size_t)vocab * emb;
    size_t in_sz = has_in ? (size_t)hidden * emb + hidden : 0;
    struct RWKVBlockOff bo = rwkv_block_off(hidden);
    size_t blk_sz = rwkv_block_size(hidden);
    const double *C = params;
    const double *IP = has_in ? C + VEs : NULL;
    const double *IB = has_in ? IP + (size_t)hidden * emb : NULL;
    const double *blk = params + VEs + in_sz;
    const double *OP = blk + (size_t)n_layers * blk_sz;
    const double *OB = OP + (size_t)vocab * hidden;

    double *x = (double *)malloc((size_t)hidden * sizeof(double));
    double *tmp = (double *)malloc((size_t)hidden * sizeof(double));
    double *h4 = (double *)malloc(4u * (size_t)hidden * sizeof(double));
    double *st = (double *)malloc(3u * (size_t)n_layers * hidden * sizeof(double));
    if (!x || !tmp || !h4 || !st) {
        free(x); free(tmp); free(h4); free(st);
        return -3;
    }
    for (size_t i = 0; i < 3u * (size_t)n_layers * hidden; i++) st[i] = 0.0;
    for (int i = 0; i < n_layers; i++)
        for (int j = 0; j < hidden; j++) st[(3u * i + 2u) * hidden + j] = -1e30;

    for (int s = 0; s < seq_len; s++) {
        int tok = input_ids[s];
        if (tok < 0 || tok >= vocab) tok = 0;
        const double *erow = C + (size_t)tok * emb;
        if (has_in) {
            for (int i = 0; i < hidden; i++) {
                const double *row = IP + (size_t)i * emb;
                double acc = IB[i];
                for (int j = 0; j < emb; j++) acc += row[j] * erow[j];
                x[i] = acc;
            }
        } else {
            for (int i = 0; i < hidden; i++) x[i] = erow[i];
        }
        for (int l = 0; l < n_layers; l++) {
            const double *bp = blk + (size_t)l * blk_sz;
            double *num = st + (3u * l + 0u) * hidden;
            double *den = st + (3u * l + 1u) * hidden;
            double *amx = st + (3u * l + 2u) * hidden;
            /* ln1 */
            double m = 0.0;
            for (int i = 0; i < hidden; i++) m += x[i];
            m /= (double)hidden;
            double v = 0.0;
            for (int i = 0; i < hidden; i++) { double d = x[i] - m; v += d * d; }
            v /= (double)hidden;
            double D = v + RWKV_LN_EPS;
            const double *w1 = bp + bo.ln1_w, *b1 = bp + bo.ln1_b;
            for (int i = 0; i < hidden; i++)
                tmp[i] = (x[i] - m) * w1[i] / D + b1[i];
            /* wkv */
            const double *lw = bp + bo.log_w, *lk = bp + bo.log_k;
            for (int i = 0; i < hidden; i++) {
                double a = exp(lw[i]);
                double ew = exp(-exp(-a));
                double kval = exp(lk[i]) * tmp[i];
                double nmax = amx[i] > kval ? amx[i] : kval;
                double q1 = exp(amx[i] - nmax);
                double q2 = exp(kval - nmax);
                double nn = num[i] * ew * q1 + tmp[i] * kval * q2;
                double dd = den[i] * ew * q1 + kval * q2;
                num[i] = nn; den[i] = dd; amx[i] = nmax;
                tmp[i] = nn / (dd + RWKV_WKV_EPS);
            }
            /* attn out + residual */
            const double *Wo = bp + bo.W_out, *bob = bp + bo.b_out;
            for (int i = 0; i < hidden; i++) {
                const double *orow = Wo + (size_t)i * hidden;
                double acc = bob[i];
                for (int j = 0; j < hidden; j++) acc += orow[j] * tmp[j];
                x[i] = x[i] + acc;
            }
            /* ln2 */
            m = 0.0;
            for (int i = 0; i < hidden; i++) m += x[i];
            m /= (double)hidden;
            v = 0.0;
            for (int i = 0; i < hidden; i++) { double d = x[i] - m; v += d * d; }
            v /= (double)hidden;
            D = v + RWKV_LN_EPS;
            const double *w2 = bp + bo.ln2_w, *b2 = bp + bo.ln2_b;
            for (int i = 0; i < hidden; i++)
                tmp[i] = (x[i] - m) * w2[i] / D + b2[i];
            /* ffn */
            const double *W1 = bp + bo.ffn_W1, *B1 = bp + bo.ffn_b1;
            const double *W2 = bp + bo.ffn_W2, *B2 = bp + bo.ffn_b2;
            for (int j = 0; j < 4 * hidden; j++) {
                const double *row = W1 + (size_t)j * hidden;
                double acc = B1[j];
                for (int i = 0; i < hidden; i++) acc += row[i] * tmp[i];
                h4[j] = acc * 0.5 * (1.0 + tanh(acc * RWKV_GELU_C));
            }
            for (int i = 0; i < hidden; i++) {
                const double *row = W2 + (size_t)i * 4u * hidden;
                double acc = B2[i];
                for (int j = 0; j < 4 * hidden; j++) acc += row[j] * h4[j];
                x[i] = x[i] + acc;
            }
        }
    }
    for (int i = 0; i < vocab; i++) {
        const double *row = OP + (size_t)i * hidden;
        double acc = OB[i];
        for (int j = 0; j < hidden; j++) acc += row[j] * x[j];
        out_logits[i] = acc;
    }
    free(x); free(tmp); free(h4); free(st);
    return 0;
}

} // extern "C"
