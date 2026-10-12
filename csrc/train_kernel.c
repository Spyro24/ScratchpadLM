/* train_kernel.c — fused sampled-softmax training kernels for TinyLM.
 *
 * Numerically equivalent to ai.py's TinyLM.sampled_loss + backward() +
 * in-place SGD update, but computed on flat double buffers in C.
 *
 * Flat param layout (matches ai.py TinyLM.params order):
 *   C  : vocab * emb
 *   W1 : hidden * (ctx * emb)
 *   b1 : hidden
 *   W2 : vocab * hidden
 *   b2 : vocab
 *
 * Windows: batch_size groups of (ctx context ids + 1 target id).
 * Negatives: if non-NULL, batch_size * n_negatives explicit ids (used by
 * parity tests); otherwise sampled internally per candidate with an
 * xorshift64 RNG seeded per call (same distribution as random.randrange:
 * with replacement, target may repeat).
 *
 * Build: cc -O2 -shared -fPIC -o libtinylm_accel.so train_kernel.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t tl_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Accumulate loss and gradients for one window (matches sampled_loss math). */
static int window_grads(const double *params, const int *win, int ctx, int vocab,
                        int emb, int hidden, int n_negatives, uint64_t *rng,
                        const int *negs,
                        double *x, double *dx, double *h, double *dh,
                        double *logit, int *cand,
                        double *dC, double *dW1, double *db1,
                        double *dW2, double *db2, double *loss_out)
{
    const double *C = params;
    const double *W1 = C + (size_t)vocab * emb;
    const double *b1 = W1 + (size_t)hidden * ((size_t)ctx * emb);
    const double *W2 = b1 + hidden;
    const double *b2 = W2 + (size_t)vocab * hidden;
    const size_t xlen = (size_t)ctx * emb;
    const int n_cand = n_negatives + 1;

    const int target = win[ctx];
    if (target < 0 || target >= vocab) return -2;

    /* forward: gather embeddings x = concat(C[tok]) */
    for (int t = 0; t < ctx; t++) {
        int tok = win[t];
        if (tok < 0 || tok >= vocab) return -2;
        memcpy(x + (size_t)t * emb, C + (size_t)tok * emb, emb * sizeof(double));
    }

    /* h = tanh(W1 x + b1) */
    for (int i = 0; i < hidden; i++) {
        const double *row = W1 + (size_t)i * xlen;
        double s = b1[i];
        for (size_t k = 0; k < xlen; k++) s += row[k] * x[k];
        h[i] = tanh(s);
    }

    /* candidates: target first, then negatives (may repeat, may hit target) */
    cand[0] = target;
    for (int j = 1; j < n_cand; j++) {
        int c = negs ? negs[j - 1] : (int)(tl_rand(rng) % (uint64_t)vocab);
        if (c < 0 || c >= vocab) return -2;
        cand[j] = c;
    }

    /* logits over candidates */
    double mx = -1e308;
    for (int j = 0; j < n_cand; j++) {
        const double *row = W2 + (size_t)cand[j] * hidden;
        double s = b2[cand[j]];
        for (int i = 0; i < hidden; i++) s += row[i] * h[i];
        logit[j] = s;
        if (s > mx) mx = s;
    }
    double sum = 0.0;
    for (int j = 0; j < n_cand; j++) sum += exp(logit[j] - mx);
    /* loss = -(l0 - m - log s)  (same as softmax_cross_entropy at index 0) */
    *loss_out = -(logit[0] - mx - log(sum));

    /* dlogit_j = p_j - delta(j, 0) */
    for (int j = 0; j < n_cand; j++) logit[j] = exp(logit[j] - mx) / sum;
    logit[0] -= 1.0;

    /* backward through output layer: dW2, db2, dh */
    memset(dh, 0, (size_t)hidden * sizeof(double));
    for (int j = 0; j < n_cand; j++) {
        const double g = logit[j];
        const double *row = W2 + (size_t)cand[j] * hidden;
        double *grow = dW2 + (size_t)cand[j] * hidden;
        for (int i = 0; i < hidden; i++) {
            grow[i] += g * h[i];
            dh[i] += g * row[i];
        }
        db2[cand[j]] += g;
    }

    /* backward through tanh + hidden layer: dW1, db1, dx */
    memset(dx, 0, xlen * sizeof(double));
    for (int i = 0; i < hidden; i++) {
        const double dhpre = dh[i] * (1.0 - h[i] * h[i]);
        const double *row = W1 + (size_t)i * xlen;
        double *grow = dW1 + (size_t)i * xlen;
        for (size_t k = 0; k < xlen; k++) {
            grow[k] += dhpre * x[k];
            dx[k] += dhpre * row[k];
        }
        db1[i] += dhpre;
    }

    /* scatter-add dx into embedding grads */
    for (int t = 0; t < ctx; t++) {
        double *drow = dC + (size_t)win[t] * emb;
        const double *src = dx + (size_t)t * emb;
        for (int e = 0; e < emb; e++) drow[e] += src[e];
    }
    return 0;
}

static size_t count_params(int vocab, int ctx, int emb, int hidden) {
    return (size_t)vocab * emb + (size_t)hidden * ((size_t)ctx * emb) + (size_t)hidden
         + (size_t)vocab * hidden + (size_t)vocab;
}

/* Helper: compute global gradient norm */
static double compute_grad_norm(const double *grads, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += grads[i] * grads[i];
    return sqrt(sum);
}

/* Core: summed loss + summed grads over the batch (unscaled). */
static int core(const double *params, const int *windows, const int *negatives,
                int batch, int ctx, int vocab, int emb, int hidden,
                int n_negatives, uint64_t *rng, double *loss_sum, double *grads)
{
    if (!params || !windows || !loss_sum || !grads) return -1;
    if (batch < 1 || ctx < 1 || vocab < 1 || emb < 1 || hidden < 1) return -1;
    if (n_negatives < 0) return -1;

    const size_t xlen = (size_t)ctx * emb;
    const int n_cand = n_negatives + 1;

    memset(grads, 0, count_params(vocab, ctx, emb, hidden) * sizeof(double));
    double *dC = grads;
    double *dW1 = dC + (size_t)vocab * emb;
    double *db1 = dW1 + (size_t)hidden * xlen;
    double *dW2 = db1 + hidden;
    double *db2 = dW2 + (size_t)vocab * hidden;

    double *x = malloc(xlen * sizeof(double));
    double *dx = malloc(xlen * sizeof(double));
    double *h = malloc((size_t)hidden * sizeof(double));
    double *dh = malloc((size_t)hidden * sizeof(double));
    double *logit = malloc((size_t)n_cand * sizeof(double));
    int *cand = malloc((size_t)n_cand * sizeof(int));
    if (!x || !dx || !h || !dh || !logit || !cand) {
        free(x); free(dx); free(h); free(dh); free(logit); free(cand);
        return -3;
    }

    double total = 0.0;
    int rc = 0;
    for (int b = 0; b < batch; b++) {
        const int *win = windows + (size_t)b * (ctx + 1);
        const int *negs = negatives ? negatives + (size_t)b * n_negatives : NULL;
        double wl = 0.0;
        rc = window_grads(params, win, ctx, vocab, emb, hidden, n_negatives,
                          rng, negs, x, dx, h, dh, logit, cand,
                          dC, dW1, db1, dW2, db2, &wl);
        if (rc != 0) break;
        total += wl;
    }

    free(x); free(dx); free(h); free(dh); free(logit); free(cand);
    if (rc == 0) *loss_sum = total;
    return rc;
}

/* Exported: summed loss and summed grads over the batch with optional gradient clipping
 * Returns summed loss and summed grads (no scaling, no parameter update).
 * Used by train_parallel workers and parity tests. `negatives` may be NULL to sample
 * internally with `seed`. If max_norm > 0, gradients are clipped by global norm.
 * Returns 0 on success, -1 invalid arg, -2 token id out of range, -3 OOM. */
int tl_batch_loss_grads(const double *params, const int *windows,
                        const int *negatives,
                        int batch, int ctx, int vocab, int emb, int hidden,
                        int n_negatives, unsigned long long seed,
                        double *out_loss, double *out_grads,
                        double max_norm)
{
    if (!out_loss || !out_grads) return -1;
    uint64_t rng = (uint64_t)seed ^ 0x9E3779B97F4A7C15ULL;
    if (rng == 0) rng = 0x9E3779B97F4A7C15ULL;
    int rc = core(params, windows, negatives, batch, ctx, vocab, emb, hidden,
                  n_negatives, &rng, out_loss, out_grads);
    if (rc == 0 && max_norm > 0.0) {
        const size_t n_params = count_params(vocab, ctx, emb, hidden);
        double norm = compute_grad_norm(out_grads, n_params);
        if (norm > max_norm && norm > 0.0) {
            double scale = max_norm / norm;
            for (size_t i = 0; i < n_params; i++) out_grads[i] *= scale;
        }
    }
    return rc;
}

/* Exported: one fused training step with optional gradient clipping
 * params[i] -= lr * grad[i] / batch (with clipping if max_norm > 0)
 * Negatives are sampled internally. Returns 0 on success, see codes above. */
int tl_train_step(double *params, const int *windows,
                  int batch, int ctx, int vocab, int emb, int hidden,
                  int n_negatives, double lr, unsigned long long seed,
                  double *out_loss, double max_norm)
{
    if (!out_loss) return -1;
    const size_t n_params = count_params(vocab, ctx, emb, hidden);
    double *grads = malloc(n_params * sizeof(double));
    if (!grads) return -3;
    uint64_t rng = (uint64_t)seed ^ 0x9E3779B97F4A7C15ULL;
    if (rng == 0) rng = 0x9E3779B97F4A7C15ULL;
    int rc = core(params, windows, NULL, batch, ctx, vocab, emb, hidden,
                  n_negatives, &rng, out_loss, grads);
    if (rc == 0) {
        *out_loss /= batch;
        if (max_norm > 0.0) {
            double norm = compute_grad_norm(grads, n_params);
            if (norm > max_norm && norm > 0.0) {
                double scale = max_norm / norm;
                for (size_t i = 0; i < n_params; i++) grads[i] *= scale;
            }
        }
        for (size_t i = 0; i < n_params; i++) params[i] -= lr * grads[i] / batch;
    }
    free(grads);
    return rc;
}