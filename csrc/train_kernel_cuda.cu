/*
 * train_kernel_cuda.cu — CUDA-accelerated sampled-softmax training for TinyLM.
 *
 * LEGACY path for the old MLP arch (tinylm/models_tiny.py). Canonical going
 * forward is the RWKV arch (tinylm/models_linear.py + csrc/rwkv_kernel.cu).
 * Kept as fallback until the old arch is removed.
 * 
 * Compatible with GTX 970 (compute capability 5.2, Maxwell architecture).
 * 
 * Build: nvcc -O2 -std=c++11 -arch=sm_52 -shared -Xcompiler -fPIC -o libtinylm_cuda.so train_kernel_cuda.cu -lcublas
 * 
 * Flat param layout (matches TinyLM.params order):
 *   C  : vocab * emb
 *   W1 : hidden * (ctx * emb)
 *   b1 : hidden
 *   W2 : vocab * hidden
 *   b2 : vocab
 * 
 * Windows: batch_size groups of (ctx context ids + 1 target id).
 * Negatives: if non-NULL, batch_size * n_negatives explicit ids.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* atomicAdd for double (not natively supported on compute capability < 6.0) */
__device__ double atomicAdd_double(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* Thread-local pre-allocated buffers to avoid malloc/free per batch */
typedef struct {
    double *d_params;
    double *d_grads;
    double *d_x;
    double *d_dx;
    double *d_h;
    double *d_dh;
    double *d_logit;
    int *d_cand;
    double *d_vel;      /* momentum velocity */
    size_t xlen;
    size_t n_cand;
    size_t hidden;
    size_t vocab;
    size_t emb;
    size_t ctx;
    size_t vel_len;
    int initialized;
    curandState *d_rng_states;
    int batch_size;
    int n_negatives;
} cuda_Buffers;

static cuda_Buffers cuda_buffers = {0};

static __host__ __device__ size_t count_params(int vocab, int ctx, int emb, int hidden) {
    return (size_t)vocab * emb + (size_t)hidden * ((size_t)ctx * emb) + (size_t)hidden
         + (size_t)vocab * hidden + (size_t)vocab;
}

/* xorshift64 RNG for device */
__device__ uint64_t dev_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Accumulate loss and gradients for one window (matches sampled_loss math) */
__device__ void window_grads_device(const double *params, const int *win, int ctx, int vocab,
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
    if (target < 0 || target >= vocab) { *loss_out = -1; return; }

    /* forward: gather embeddings x = concat(C[tok]) */
    for (int t = 0; t < ctx; t++) {
        int tok = win[t];
        if (tok < 0 || tok >= vocab) { *loss_out = -1; return; }
        for (int e = 0; e < emb; e++) {
            x[(size_t)t * emb + e] = C[(size_t)tok * emb + e];
        }
    }

    /* h = tanh(W1 x + b1) */
    for (int i = 0; i < hidden; i++) {
        const double *row = W1 + (size_t)i * xlen;
        double s = b1[i];
        for (size_t k = 0; k < xlen; k++) s += row[k] * x[k];
        h[i] = tanh(s);
    }

    /* candidates: target first, then negatives */
    cand[0] = target;
    for (int j = 1; j < n_cand; j++) {
        int c = negs ? negs[j - 1] : (int)(dev_rand(rng) % (uint64_t)vocab);
        if (c < 0 || c >= vocab) { *loss_out = -1; return; }
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
    *loss_out = -(logit[0] - mx - log(sum));

    /* dlogit_j = p_j - delta(j, 0) */
    for (int j = 0; j < n_cand; j++) logit[j] = exp(logit[j] - mx) / sum;
    logit[0] -= 1.0;

    /* backward through output layer: dW2, db2, dh */
    for (int i = 0; i < hidden; i++) dh[i] = 0.0;
    for (int j = 0; j < n_cand; j++) {
        const double g = logit[j];
        const double *row = W2 + (size_t)cand[j] * hidden;
        double *grow = dW2 + (size_t)cand[j] * hidden;
        for (int i = 0; i < hidden; i++) {
            atomicAdd_double(grow + i, g * h[i]);
            atomicAdd_double(dh + i, g * row[i]);
        }
        atomicAdd_double(db2 + cand[j], g);
    }

    /* backward through tanh + hidden layer: dW1, db1, dx */
    for (size_t k = 0; k < xlen; k++) dx[k] = 0.0;
    for (int i = 0; i < hidden; i++) {
        const double dhpre = dh[i] * (1.0 - h[i] * h[i]);
        const double *row = W1 + (size_t)i * xlen;
        double *grow = dW1 + (size_t)i * xlen;
        for (size_t k = 0; k < xlen; k++) {
            atomicAdd_double(grow + k, dhpre * x[k]);
            atomicAdd_double(dx + k, dhpre * row[k]);
        }
        atomicAdd_double(db1 + i, dhpre);
    }

    /* scatter-add dx into embedding grads */
    for (int t = 0; t < ctx; t++) {
        double *drow = dC + (size_t)win[t] * emb;
        for (int e = 0; e < emb; e++) atomicAdd_double(drow + e, dx[(size_t)t * emb + e]);
    }
}

/* Batch loss and gradients kernel - one thread per window (grid-stride loop).
 * Per-thread scratch lives in global memory (no shared-memory size limits):
 *   dbl_scratch[tid] = [x, dx (xlen each), h, dh (hidden each), logit (n_cand)]
 *   int_scratch[tid] = [cand (n_cand)]
 * with xlen = ctx*emb, n_cand = n_negatives + 1 passed in (host-sized).
 * Gradient accumulation uses global atomics (see window_grads_device), so any
 * grid shape is correct; loss is summed with a double atomic. */
__global__ void batch_loss_grads_kernel(const double *params, const int *windows,
                                        const int *negatives,
                                        int batch, int ctx, int vocab, int emb, int hidden,
                                        int n_negatives, uint64_t seed,
                                        double *out_loss, double *out_grads,
                                        double *dbl_scratch, int *int_scratch,
                                        size_t xlen, int n_cand,
                                        double max_norm)
{
    size_t n_params = count_params(vocab, ctx, emb, hidden);

    /* Initialize gradients to zero (parallel over threads) */
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n_params; i += blockDim.x * gridDim.x) {
        out_grads[i] = 0.0;
    }
    __syncthreads();

    size_t dbl_per_thread = 2 * xlen + 2 * (size_t)hidden + (size_t)n_cand;
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nthreads = blockDim.x * gridDim.x;
    for (int w = (int)tid; w < batch; w += (int)nthreads) {
        const int *win = windows + (size_t)w * (ctx + 1);
        const int *negs = negatives ? negatives + (size_t)w * n_negatives : NULL;
        double wl = 0.0;

        uint64_t rng = (seed ^ 0x9E3779B97F4A7C15ULL) + (uint64_t)w * 0x9E3779B97F4A7C15ULL;
        if (rng == 0) rng = 0x9E3779B97F4A7C15ULL;

        double *base = dbl_scratch + (size_t)w * dbl_per_thread;
        double *x = base;
        double *dx = x + xlen;
        double *h = dx + xlen;
        double *dh = h + hidden;
        double *logit = dh + hidden;
        int *cand = int_scratch + (size_t)w * n_cand;

        window_grads_device(params, win, ctx, vocab, emb, hidden, n_negatives,
                            &rng, negs, x, dx, h, dh, logit, cand,
                            out_grads,
                            out_grads + (size_t)vocab * emb,
                            out_grads + (size_t)vocab * emb + (size_t)hidden * xlen,
                            out_grads + (size_t)vocab * emb + (size_t)hidden * xlen + hidden,
                            out_grads + (size_t)vocab * emb + (size_t)hidden * xlen + hidden + (size_t)vocab * hidden,
                            &wl);

        if (wl >= 0) {
            atomicAdd_double(out_loss, wl);
        }
    }
}

/* Divide loss by batch size */
__global__ void divide_loss_kernel(double *loss, int batch) {
    *loss /= batch;
}

/* Momentum update kernel */
__global__ void momentum_update_kernel(double *params, double *grads, double *vel,
                                       size_t n_params, int batch, double lr, double momentum)
{
    /* Average gradients */
    for (size_t i = threadIdx.x; i < n_params; i += blockDim.x) {
        grads[i] /= batch;
    }
    __syncthreads();
    
    /* Momentum update: v = momentum * v + g; params -= lr * v */
    if (momentum > 0.0) {
        for (size_t i = threadIdx.x; i < n_params; i += blockDim.x) {
            double v = momentum * vel[i] + grads[i];
            vel[i] = v;
            params[i] -= lr * v;
        }
    } else {
        for (size_t i = threadIdx.x; i < n_params; i += blockDim.x) {
            params[i] -= lr * grads[i];
        }
    }
}

/* RNG initialization kernel */
__global__ void init_rng_states_kernel(curandState *states, int n, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/* Host helper functions */
extern "C" {

int cuda_init_buffers(int ctx, int vocab, int emb, int hidden, int n_negatives, int batch_size) {
    cuda_Buffers *b = &cuda_buffers;
    size_t xlen = (size_t)ctx * emb;
    size_t n_cand = (size_t)n_negatives + 1;
    size_t n_params = count_params(vocab, ctx, emb, hidden);
    
    if (b->initialized && b->xlen == xlen && b->n_cand == n_cand && 
        b->hidden == (size_t)hidden && b->vocab == (size_t)vocab && 
        b->emb == (size_t)emb && b->ctx == (size_t)ctx) {
        return 0;
    }
    
    if (b->initialized) {
        cudaFree(b->d_params);
        cudaFree(b->d_grads);
        cudaFree(b->d_x);
        cudaFree(b->d_dx);
        cudaFree(b->d_h);
        cudaFree(b->d_dh);
        cudaFree(b->d_logit);
        cudaFree(b->d_cand);
        cudaFree(b->d_vel);
        cudaFree(b->d_rng_states);
    }
    
    b->xlen = xlen;
    b->n_cand = n_cand;
    b->hidden = hidden;
    b->vocab = vocab;
    b->emb = emb;
    b->ctx = ctx;
    b->vel_len = n_params;
    b->batch_size = batch_size;
    b->n_negatives = n_negatives;
    
    cudaMalloc(&b->d_params, n_params * sizeof(double));
    cudaMalloc(&b->d_grads, n_params * sizeof(double));
    cudaMalloc(&b->d_x, xlen * sizeof(double));
    cudaMalloc(&b->d_dx, xlen * sizeof(double));
    cudaMalloc(&b->d_h, hidden * sizeof(double));
    cudaMalloc(&b->d_dh, hidden * sizeof(double));
    cudaMalloc(&b->d_logit, n_cand * sizeof(double));
    cudaMalloc(&b->d_cand, n_cand * sizeof(int));
    cudaMalloc(&b->d_vel, n_params * sizeof(double));
    cudaMalloc(&b->d_rng_states, batch_size * sizeof(curandState));
    
    /* Initialize RNG states on device */
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    init_rng_states_kernel<<<blocks, threads>>>(b->d_rng_states, batch_size, 12345);
    cudaDeviceSynchronize();
    
    /* Zero velocity */
    cudaMemset(b->d_vel, 0, n_params * sizeof(double));
    
    b->initialized = 1;
    return 0;
}

void cuda_reset_velocity() {
    cuda_Buffers *b = &cuda_buffers;
    if (b->initialized && b->d_vel) {
        cudaMemset(b->d_vel, 0, b->vel_len * sizeof(double));
    }
}

int cuda_train_step(double *params, const int *windows,
                    const int *negatives,
                    int batch, int ctx, int vocab, int emb, int hidden,
                    int n_negatives, double lr, double momentum,
                    unsigned long long seed,
                    double *out_loss, double max_norm)
{
    cuda_Buffers *b = &cuda_buffers;
    size_t n_params = count_params(vocab, ctx, emb, hidden);
    
    /* Copy params to device */
    cudaMemcpy(b->d_params, params, n_params * sizeof(double), cudaMemcpyHostToDevice);
    
    /* Copy windows to device */
    int *d_windows;
    cudaMalloc(&d_windows, (size_t)batch * (ctx + 1) * sizeof(int));
    cudaMemcpy(d_windows, windows, (size_t)batch * (ctx + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_negatives = NULL;
    if (negatives) {
        cudaMalloc(&d_negatives, (size_t)batch * n_negatives * sizeof(int));
        cudaMemcpy(d_negatives, negatives, (size_t)batch * n_negatives * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    double *d_out_loss;
    cudaMalloc(&d_out_loss, sizeof(double));
    cudaMemset(d_out_loss, 0, sizeof(double));
    
    /* Per-window scratch in global memory (sized from this call's dims). */
    size_t xlen = (size_t)ctx * emb;
    int n_cand = n_negatives + 1;
    size_t dbl_per_thread = 2 * xlen + 2 * (size_t)hidden + (size_t)n_cand;
    double *d_scratch_d = NULL;
    int *d_scratch_i = NULL;
    int nthreads = batch < 256 ? batch : 256;
    int nblocks = (batch + nthreads - 1) / nthreads;
    if (nblocks > 16) nblocks = 16;
    if (cudaMalloc(&d_scratch_d, (size_t)batch * dbl_per_thread * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_scratch_i, (size_t)batch * n_cand * sizeof(int)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_out_loss);
        cudaFree(d_scratch_d);
        cudaFree(d_scratch_i);
        return -3;
    }

    /* Step 1: Batch loss and gradients */
    batch_loss_grads_kernel<<<nblocks, nthreads>>>(b->d_params, d_windows, d_negatives,
                                                   batch, ctx, vocab, emb, hidden,
                                                   n_negatives, seed,
                                                   d_out_loss, b->d_grads,
                                                   d_scratch_d, d_scratch_i,
                                                   xlen, n_cand, max_norm);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA main kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA main kernel execution failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    /* Divide loss by batch size (match C kernel behavior) */
    if (batch > 1) {
        divide_loss_kernel<<<1, 1>>>(d_out_loss, batch);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA divide loss kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA divide loss kernel execution failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }
    
    /* Step 2: Momentum update */
    momentum_update_kernel<<<1, 256>>>(b->d_params, b->d_grads, b->d_vel,
                                       n_params, batch, lr, momentum);
    
    /* Copy results back */
    cudaMemcpy(params, b->d_params, n_params * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_loss, d_out_loss, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_windows);
    if (d_negatives) cudaFree(d_negatives);
    cudaFree(d_out_loss);
    cudaFree(d_scratch_d);
    cudaFree(d_scratch_i);

    return 0;
}

/* Multi-step training on GPU - runs multiple steps on device, only syncs for checkpoint/validation */
int cuda_train_steps(double *params, const int *all_ids, int ids_len,
                     int steps, int batch_size, int n_negatives,
                     double lr, double momentum, unsigned long long seed,
                     int ctx, int vocab, int emb, int hidden,
                     int checkpoint_every, int eval_every,
                     double *out_final_loss, int *out_steps_done,
                     int val_split, const int *val_ids, int val_len)
{
    cuda_Buffers *b = &cuda_buffers;
    size_t n_params = count_params(vocab, ctx, emb, hidden);
    int train_ids_len = ids_len;
    
    if (val_split > 0 && val_ids == NULL) {
        train_ids_len = (int)(ids_len * (1.0 - val_split));
    }
    
    if (train_ids_len <= ctx) {
        return -1;
    }
    
    /* Copy all training IDs to device */
    int *d_all_ids;
    cudaMalloc(&d_all_ids, (size_t)ids_len * sizeof(int));
    cudaMemcpy(d_all_ids, all_ids, (size_t)ids_len * sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_val_ids = NULL;
    int val_ids_len = 0;
    if (val_ids && val_len > 0) {
        val_ids_len = val_len;
        cudaMalloc(&d_val_ids, (size_t)val_len * sizeof(int));
        cudaMemcpy(d_val_ids, val_ids, (size_t)val_len * sizeof(int), cudaMemcpyHostToDevice);
    } else if (val_split > 0) {
        val_ids_len = ids_len - train_ids_len;
    }
    
    /* Copy params to device */
    cudaMemcpy(b->d_params, params, n_params * sizeof(double), cudaMemcpyHostToDevice);
    
    int starts_count = train_ids_len - ctx;
    int *d_windows;
    cudaMalloc(&d_windows, (size_t)batch_size * (ctx + 1) * sizeof(int));
    
    double *d_out_loss;
    cudaMalloc(&d_out_loss, sizeof(double));
    
    size_t xlen = (size_t)ctx * emb;
    int n_cand = n_negatives + 1;
    size_t dbl_per_thread = 2 * xlen + 2 * (size_t)hidden + (size_t)n_cand;
    double *d_scratch_d = NULL;
    int *d_scratch_i = NULL;
    if (cudaMalloc(&d_scratch_d, (size_t)batch_size * dbl_per_thread * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_scratch_i, (size_t)batch_size * n_cand * sizeof(int)) != cudaSuccess) {
        cudaFree(d_all_ids);
        if (d_val_ids) cudaFree(d_val_ids);
        cudaFree(d_windows);
        cudaFree(d_out_loss);
        cudaFree(d_scratch_d);
        cudaFree(d_scratch_i);
        return -3;
    }
    int nthreads = batch_size < 256 ? batch_size : 256;
    int nblocks = (batch_size + nthreads - 1) / nthreads;
    if (nblocks > 16) nblocks = 16;
    
    double total_loss = 0.0;
    int steps_done = 0;
    double last_loss = 0.0;
    
    for (int step = 0; step < steps; step++) {
        /* Generate random windows on GPU */
        // We'll generate on host for simplicity, could be moved to device
        int *h_windows = (int*)malloc((size_t)batch_size * (ctx + 1) * sizeof(int));
        unsigned long long rng_state = seed + step;
        for (int i = 0; i < batch_size; i++) {
            int start = (int)(rng_state % (unsigned long long)starts_count);
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            rng_state = rng_state * 0x2545F4914F6CDD1DULL;
            for (int t = 0; t < ctx; t++) {
                h_windows[i * (ctx + 1) + t] = all_ids[start + t];
            }
            h_windows[i * (ctx + 1) + ctx] = all_ids[start + ctx];
        }
        cudaMemcpy(d_windows, h_windows, (size_t)batch_size * (ctx + 1) * sizeof(int), cudaMemcpyHostToDevice);
        free(h_windows);
        
        cudaMemset(d_out_loss, 0, sizeof(double));
        
        /* Step 1: Batch loss and gradients */
        batch_loss_grads_kernel<<<nblocks, nthreads>>>(b->d_params, d_windows, NULL,
                                                        batch_size, ctx, vocab, emb, hidden,
                                                        n_negatives, seed + step,
                                                        d_out_loss, b->d_grads,
                                                        d_scratch_d, d_scratch_i,
                                                        xlen, n_cand, 0.0);
        
        /* Step 2: Momentum update */
        momentum_update_kernel<<<1, 256>>>(b->d_params, b->d_grads, b->d_vel,
                                           n_params, batch_size, lr, momentum);
        
        /* Get loss */
        double step_loss = 0.0;
        cudaMemcpy(&step_loss, d_out_loss, sizeof(double), cudaMemcpyDeviceToHost);
        total_loss += step_loss;
        last_loss = step_loss;
        steps_done = step + 1;
        
        /* Checkpoint/validation would require host sync - for now just track */
        if (checkpoint_every > 0 && (step + 1) % checkpoint_every == 0) {
            /* Would sync params and save - placeholder */
        }
    }
    
    cudaMemcpy(params, b->d_params, n_params * sizeof(double), cudaMemcpyDeviceToHost);
    *out_final_loss = last_loss;
    *out_steps_done = steps_done;
    
    cudaFree(d_all_ids);
    if (d_val_ids) cudaFree(d_val_ids);
    cudaFree(d_windows);
    cudaFree(d_out_loss);
    cudaFree(d_scratch_d);
    cudaFree(d_scratch_i);

    return 0;
}


/* Exported: summed loss + summed grads for one batch (no update).
 * Mirrors accel.tl_batch_loss_grads semantics: grads are SUMMED (caller
 * divides), loss is the batch SUM (caller divides). Pre-sampled `negatives`
 * (batch * n_negatives) or internal xorshift sampling with `seed`.
 * Parallel across windows (grid-stride loop, global atomics); per-thread
 * scratch is sized from the call's own dims (no fixed limits).
 * Returns 0 on success, see codes above. */
int cuda_batch_loss_grads(const double *params, const int *windows,
                          const int *negatives,
                          int batch, int ctx, int vocab, int emb, int hidden,
                          int n_negatives, unsigned long long seed,
                          double *out_loss, double *out_grads,
                          double max_norm)
{
    if (!params || !windows || !out_loss || !out_grads) return -1;
    if (batch < 1 || ctx < 1 || vocab < 1 || emb < 1 || hidden < 1) return -1;
    if (n_negatives < 0) return -1;
    const size_t n_params = count_params(vocab, ctx, emb, hidden);
    const size_t xlen = (size_t)ctx * (size_t)emb;
    const int n_cand = n_negatives + 1;

    int *d_windows;
    if (cudaMalloc(&d_windows, (size_t)batch * (ctx + 1) * sizeof(int)) != cudaSuccess)
        return -3;
    if (cudaMemcpy(d_windows, windows, (size_t)batch * (ctx + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_windows);
        return -3;
    }

    int *d_negatives = NULL;
    if (negatives) {
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

    double *d_params, *d_grads, *d_out_loss;
    if (cudaMalloc(&d_params, n_params * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_grads, n_params * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_out_loss, sizeof(double)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params);
        cudaFree(d_grads);
        cudaFree(d_out_loss);
        return -3;
    }
    if (cudaMemcpy(d_params, params, n_params * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemset(d_out_loss, 0, sizeof(double)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params);
        cudaFree(d_grads);
        cudaFree(d_out_loss);
        return -3;
    }

    /* Per-thread scratch in global memory (doubles + ints, byte-exact). */
    size_t dbl_per_thread = 2 * xlen + 2 * (size_t)hidden + (size_t)n_cand;
    double *d_scratch_d = NULL;
    int *d_scratch_i = NULL;
    if (cudaMalloc(&d_scratch_d, (size_t)batch * dbl_per_thread * sizeof(double)) != cudaSuccess
            || cudaMalloc(&d_scratch_i, (size_t)batch * n_cand * sizeof(int)) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params);
        cudaFree(d_grads);
        cudaFree(d_out_loss);
        cudaFree(d_scratch_d);
        cudaFree(d_scratch_i);
        return -3;
    }
    int nthreads = batch < 256 ? batch : 256;
    int nblocks = (batch + nthreads - 1) / nthreads;
    if (nblocks > 16) nblocks = 16;
    batch_loss_grads_kernel<<<nblocks, nthreads>>>(d_params, d_windows, d_negatives,
                                                   batch, ctx, vocab, emb, hidden,
                                                   n_negatives, seed,
                                                   d_out_loss, d_grads,
                                                   d_scratch_d, d_scratch_i,
                                                   xlen, n_cand, max_norm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA batch grads launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params);
        cudaFree(d_grads);
        cudaFree(d_out_loss);
        cudaFree(d_scratch_d);
        cudaFree(d_scratch_i);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA batch grads execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params);
        cudaFree(d_grads);
        cudaFree(d_out_loss);
        cudaFree(d_scratch_d);
        cudaFree(d_scratch_i);
        return -1;
    }

    if (cudaMemcpy(out_loss, d_out_loss, sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(out_grads, d_grads, n_params * sizeof(double),
                          cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_windows);
        if (d_negatives) cudaFree(d_negatives);
        cudaFree(d_params);
        cudaFree(d_grads);
        cudaFree(d_out_loss);
        cudaFree(d_scratch_d);
        cudaFree(d_scratch_i);
        return -3;
    }

    cudaFree(d_windows);
    if (d_negatives) cudaFree(d_negatives);
    cudaFree(d_params);
    cudaFree(d_grads);
    cudaFree(d_out_loss);
    cudaFree(d_scratch_d);
    cudaFree(d_scratch_i);
    return 0;
}

} // extern "C"