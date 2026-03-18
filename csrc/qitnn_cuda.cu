#if defined(_WIN32) && !defined(QITNN_BUILD)
#define QITNN_BUILD 1
#endif

#include "qitnn/qitnn.h"
#include "qitnn/qitnn_device.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct DeviceSlot {
    void* dev = nullptr;
    size_t bytes = 0;
};

std::unordered_map<void*, DeviceSlot> g_pin_map;
std::mutex g_pin_lock;

bool g_cuda_ready = false;
bool g_cuda_checked = false;
cublasHandle_t g_cublas = nullptr;
bool g_cublas_ready = false;

float* g_scores = nullptr;
size_t g_scores_cap = 0;
float* g_loss = nullptr;
size_t g_loss_cap = 0;

void qitnn_report(cudaError_t err, const char* op, int line) {
    if (err == cudaSuccess) {
        return;
    }
    const char* name = cudaGetErrorName(err);
    const char* msg = cudaGetErrorString(err);
    std::fprintf(
        stderr,
        "[libQITNN] %s failed at line %d: err=%d (%s) %s\n",
        op,
        line,
        (int)err,
        name ? name : "?",
        msg ? msg : "?"
    );
}

const char* qitnn_cublas_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

bool qitnn_report_cublas(cublasStatus_t status, const char* op, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(
        stderr,
        "[libQITNN] %s failed at line %d: status=%d (%s)\n",
        op,
        line,
        (int)status,
        qitnn_cublas_name(status)
    );
    return false;
}

#define QITNN_CU(call) qitnn_report((call), #call, __LINE__)
#define QITNN_CUBLAS(call) qitnn_report_cublas((call), #call, __LINE__)

bool qitnn_init_cuda() {
    if (g_cuda_checked) {
        return g_cuda_ready;
    }
    g_cuda_checked = true;
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count <= 0) {
        qitnn_report(err, "cudaGetDeviceCount", __LINE__);
        g_cuda_ready = false;
        return false;
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        qitnn_report(err, "cudaSetDevice", __LINE__);
        g_cuda_ready = false;
        return false;
    }
    cudaGetLastError();
    if (g_cublas == nullptr) {
        if (!QITNN_CUBLAS(cublasCreate(&g_cublas))) {
            g_cublas_ready = false;
            g_cuda_ready = false;
            return false;
        }
        g_cublas_ready = true;
    }
    g_cuda_ready = true;
    return true;
}

bool qitnn_is_pinned(const void* host) {
    std::lock_guard<std::mutex> guard(g_pin_lock);
    return g_pin_map.find(const_cast<void*>(host)) != g_pin_map.end();
}

bool qitnn_all_pinned(std::initializer_list<const void*> ptrs) {
    for (const void* ptr : ptrs) {
        if (!qitnn_is_pinned(ptr)) {
            return false;
        }
    }
    return true;
}

DeviceSlot qitnn_lookup_slot(const void* host) {
    std::lock_guard<std::mutex> guard(g_pin_lock);
    auto it = g_pin_map.find(const_cast<void*>(host));
    if (it == g_pin_map.end()) {
        return {};
    }
    return it->second;
}

void qitnn_store_slot(void* host, void* dev, size_t bytes) {
    std::lock_guard<std::mutex> guard(g_pin_lock);
    g_pin_map[host] = DeviceSlot{dev, bytes};
}

void qitnn_erase_slot(void* host) {
    std::lock_guard<std::mutex> guard(g_pin_lock);
    g_pin_map.erase(host);
}

void* qitnn_dev(float* host, size_t bytes, bool upload) {
    DeviceSlot slot = qitnn_lookup_slot(host);
    if (slot.dev != nullptr) {
        return slot.dev;
    }
    void* dev = nullptr;
    QITNN_CU(cudaMalloc(&dev, bytes));
    if (upload && bytes > 0) {
        QITNN_CU(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    }
    return dev;
}

void qitnn_undev(float* host, void* dev, size_t bytes, bool download) {
    if (qitnn_is_pinned(host)) {
        return;
    }
    if (download && bytes > 0) {
        QITNN_CU(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
    }
    QITNN_CU(cudaFree(dev));
}

float* qitnn_ensure_buffer(float*& ptr, size_t& cap, size_t bytes) {
    if (cap >= bytes) {
        return ptr;
    }
    if (ptr != nullptr) {
        QITNN_CU(cudaFree(ptr));
    }
    QITNN_CU(cudaMalloc(&ptr, bytes));
    cap = bytes;
    return ptr;
}

__global__ void qitnn_matmul_k(const float* a, const float* b, float* c, int rows, int cols, int inner) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < inner; ++k) {
        acc += a[row * inner + k] * b[k * cols + col];
    }
    c[row * cols + col] = acc;
}

bool qitnn_gemm_row_major(const float* a, const float* b, float* c, int rows, int cols, int inner) {
    if (rows <= 0 || cols <= 0 || inner <= 0) {
        return true;
    }

    if (g_cublas_ready && g_cublas != nullptr) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if (QITNN_CUBLAS(cublasSgemm(
            g_cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            cols,
            rows,
            inner,
            &alpha,
            b,
            cols,
            a,
            inner,
            &beta,
            c,
            cols
        ))) {
            return true;
        }
    }

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    qitnn_matmul_k<<<grid, block>>>(a, b, c, rows, cols, inner);
    return false;
}

__global__ void qitnn_sgd_k(float* w, const float* g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] -= lr * g[i];
    }
}

__global__ void qitnn_prior_k(float* an, float* az, float* ap, float step, float entropy_floor, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float prob_eps = 1e-6f;
    const float inv_ln2 = 1.4426950408889634f;

    float a = an[i];
    float b = az[i];
    float c = ap[i];
    float a2 = a * a;
    float b2 = b * b;
    float c2 = c * c;
    float z = a2 + b2 + c2;

    if (z < 1e-12f) {
        an[i] = 0.577350269f;
        az[i] = 0.577350269f;
        ap[i] = 0.577350269f;
        return;
    }

    float inv_z = 1.0f / z;
    float pn = a2 * inv_z;
    float pz = b2 * inv_z;
    float pp = c2 * inv_z;

    float pn_safe = fmaxf(pn, prob_eps);
    float pz_safe = fmaxf(pz, prob_eps);
    float pp_safe = fmaxf(pp, prob_eps);

    float h = 0.0f;
    h -= pn_safe * log2f(pn_safe);
    h -= pz_safe * log2f(pz_safe);
    h -= pp_safe * log2f(pp_safe);

    float gap = entropy_floor - h;
    if (gap <= 0.0f) {
        return;
    }

    float gpn = gap * (logf(pn_safe) + 1.0f) * inv_ln2;
    float gpz = gap * (logf(pz_safe) + 1.0f) * inv_ln2;
    float gpp = gap * (logf(pp_safe) + 1.0f) * inv_ln2;

    float common = 2.0f * inv_z;
    float ga = common * a * (gpn * (1.0f - pn) - gpz * pz - gpp * pp);
    float gb = common * b * (gpz * (1.0f - pz) - gpn * pn - gpp * pp);
    float gc = common * c * (gpp * (1.0f - pp) - gpn * pn - gpz * pz);

    ga = fminf(fmaxf(ga, -4.0f), 4.0f);
    gb = fminf(fmaxf(gb, -4.0f), 4.0f);
    gc = fminf(fmaxf(gc, -4.0f), 4.0f);

    a -= step * ga;
    b -= step * gb;
    c -= step * gc;

    float norm = sqrtf(a * a + b * b + c * c);
    if (norm > 1e-8f) {
        float inv = 1.0f / norm;
        an[i] = a * inv;
        az[i] = b * inv;
        ap[i] = c * inv;
    } else {
        an[i] = 0.577350269f;
        az[i] = 0.577350269f;
        ap[i] = 0.577350269f;
    }
}

__global__ void qitnn_normalize3_k(
    const float* cn,
    const float* cz,
    const float* cp,
    float* u,
    float* v,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) {
        return;
    }
    float a = cn[i];
    float b = cz[i];
    float c = cp[i];
    float a2 = a * a;
    float b2 = b * b;
    float c2 = c * c;
    float z = a2 + b2 + c2;
    float inv_z = (z > 1e-12f) ? (1.0f / z) : 0.0f;
    u[i] = (c2 - a2) * inv_z;
    v[i] = b2 * inv_z;
}

__global__ void qitnn_backnorm3_k(
    const float* du_in,
    const float* dv_in,
    const float* cn,
    const float* cz,
    const float* cp,
    float* dcn,
    float* dcz,
    float* dcp,
    float ent_lambda,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) {
        return;
    }

    float d_u = du_in[i];
    float d_v = dv_in[i];
    float a = cn[i];
    float b = cz[i];
    float c = cp[i];
    float a2 = a * a;
    float b2 = b * b;
    float c2 = c * c;
    float z = a2 + b2 + c2;
    float inv_z = (z > 1e-12f) ? (1.0f / z) : 0.0f;
    float inv_z2 = (z > 1e-12f) ? (1.0f / (z * z)) : 0.0f;

    dcn[i] = -2.0f * a * inv_z2 * (d_u * (b2 + 2.0f * c2) + d_v * b2);
    dcz[i] = 2.0f * b * inv_z2 * (d_u * (a2 - c2) + d_v * (a2 + c2));
    dcp[i] = 2.0f * c * inv_z2 * (d_u * (2.0f * a2 + b2) - d_v * b2);

    if (ent_lambda > 0.0f) {
        float pn = a2 * inv_z;
        float p0 = b2 * inv_z;
        float pp = c2 * inv_z;
        float pne = pn + 1e-7f;
        float p0e = p0 + 1e-7f;
        float ppe = pp + 1e-7f;
        float ln = logf(pne);
        float lz = logf(p0e);
        float lp = logf(ppe);
        dcn[i] -= ent_lambda * 2.0f * a * (b2 * (lz - ln) + c2 * (lp - ln)) * inv_z2;
        dcz[i] -= ent_lambda * 2.0f * b * (a2 * (ln - lz) + c2 * (lp - lz)) * inv_z2;
        dcp[i] -= ent_lambda * 2.0f * c * (a2 * (ln - lp) + b2 * (lz - lp)) * inv_z2;
    }
}

__global__ void qitnn_centered_simplex_k(
    const float* u,
    const float* v,
    float* out_x,
    float* out_y,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) {
        return;
    }

    out_x[i] = u[i];
    out_y[i] = (1.7320508075688772f * v[i]) - 0.5773502691896258f;
}

__global__ void qitnn_seq_embed_k(const float* table, const float* tokens, float* out, int seq_len, int dim, int vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;
    if (idx >= total) {
        return;
    }
    int t = idx / dim;
    int d = idx % dim;
    int tok = (int)tokens[t];
    if (tok < 0) {
        tok = 0;
    }
    if (tok >= vocab) {
        tok = vocab - 1;
    }
    out[t * dim + d] = table[tok * dim + d];
}

__global__ void qitnn_embed_backward_k(
    float* table,
    const float* tokens,
    const float* grad,
    int seq_len,
    int dim,
    int vocab,
    float lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = vocab * dim;
    if (idx >= total) {
        return;
    }

    int tok = idx / dim;
    int d = idx % dim;
    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        int tt = (int)tokens[t];
        if (tt == tok) {
            acc += grad[t * dim + d];
        }
    }
    table[idx] -= lr * acc;
}

__global__ void qitnn_pos_add_k(float* x, const float* pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += pos[i];
    }
}

__global__ void qitnn_layernorm_batch_k(float* x, const float* gamma, const float* beta, int seq_len, int dim) {
    int t = blockIdx.x;
    if (t >= seq_len) {
        return;
    }
    float* row = x + t * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        sum += row[d];
    }
    float mean = sum / (float)dim;
    float var = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float v = row[d] - mean;
        var += v * v;
    }
    float rstd = rsqrtf(var / (float)dim + 1e-5f);
    for (int d = 0; d < dim; ++d) {
        row[d] = gamma[d] * (row[d] - mean) * rstd + beta[d];
    }
}

__global__ void qitnn_attn_scores2_k(
    const float* qx,
    const float* qy,
    const float* kx,
    const float* ky,
    float* scores,
    int seq_len,
    int dim,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * seq_len) {
        return;
    }
    int i = idx / seq_len;
    int j = idx % seq_len;
    if (j > i) {
        scores[idx] = -1e9f;
        return;
    }
    float s = 0.0f;
    for (int d = 0; d < dim; ++d) {
        s += qx[i * dim + d] * kx[j * dim + d];
        s += qy[i * dim + d] * ky[j * dim + d];
    }
    scores[idx] = s * scale;
}

__global__ void qitnn_attn_softmax_rows_k(float* scores, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) {
        return;
    }
    float* row = scores + i * seq_len;
    float mx = row[0];
    for (int j = 1; j < seq_len; ++j) {
        if (row[j] > mx) {
            mx = row[j];
        }
    }
    float sm = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        row[j] = expf(row[j] - mx);
        sm += row[j];
    }
    float inv = (sm > 1e-20f) ? (1.0f / sm) : 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        row[j] *= inv;
    }
}

__global__ void qitnn_attn_apply2_k(
    const float* scores,
    const float* vx,
    const float* vy,
    float* ox,
    float* oy,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) {
        return;
    }
    int i = idx / dim;
    int d = idx % dim;
    float sx = 0.0f;
    float sy = 0.0f;
    for (int k = 0; k < seq_len; ++k) {
        float a = scores[i * seq_len + k];
        sx += a * vx[k * dim + d];
        sy += a * vy[k * dim + d];
    }
    ox[idx] = sx;
    oy[idx] = sy;
}

__global__ void qitnn_attn_dv2_k(
    const float* attn,
    const float* dox,
    const float* doy,
    float* dvx,
    float* dvy,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) {
        return;
    }
    int i = idx / dim;
    int d = idx % dim;
    float gx = 0.0f;
    float gy = 0.0f;
    for (int k = 0; k < seq_len; ++k) {
        float a = attn[k * seq_len + i];
        gx += a * dox[k * dim + d];
        gy += a * doy[k * dim + d];
    }
    dvx[idx] = gx;
    dvy[idx] = gy;
}

__global__ void qitnn_attn_dattn2_k(
    const float* dox,
    const float* doy,
    const float* vx,
    const float* vy,
    float* d_attn,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * seq_len) {
        return;
    }
    int i = idx / seq_len;
    int j = idx % seq_len;
    float s = 0.0f;
    for (int d = 0; d < dim; ++d) {
        s += dox[i * dim + d] * vx[j * dim + d];
        s += doy[i * dim + d] * vy[j * dim + d];
    }
    d_attn[idx] = s;
}

__global__ void qitnn_attn_dscores2_k(
    const float* attn,
    const float* d_attn,
    float* d_scores,
    int seq_len,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) {
        return;
    }
    float dot = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        dot += attn[i * seq_len + j] * d_attn[i * seq_len + j];
    }
    for (int j = 0; j < seq_len; ++j) {
        d_scores[i * seq_len + j] = (j <= i) ? (attn[i * seq_len + j] * (d_attn[i * seq_len + j] - dot) * scale) : 0.0f;
    }
}

__global__ void qitnn_attn_dq2_k(
    const float* d_scores,
    const float* kx,
    const float* ky,
    float* dqx,
    float* dqy,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) {
        return;
    }
    int i = idx / dim;
    int d = idx % dim;
    float gx = 0.0f;
    float gy = 0.0f;
    for (int j = 0; j <= i; ++j) {
        float g = d_scores[i * seq_len + j];
        gx += g * kx[j * dim + d];
        gy += g * ky[j * dim + d];
    }
    dqx[idx] = gx;
    dqy[idx] = gy;
}

__global__ void qitnn_attn_dk2_k(
    const float* d_scores,
    const float* qx,
    const float* qy,
    float* dkx,
    float* dky,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) {
        return;
    }
    int i = idx / dim;
    int d = idx % dim;
    float gx = 0.0f;
    float gy = 0.0f;
    for (int j = i; j < seq_len; ++j) {
        float g = d_scores[j * seq_len + i];
        gx += g * qx[j * dim + d];
        gy += g * qy[j * dim + d];
    }
    dkx[idx] = gx;
    dky[idx] = gy;
}

__global__ void qitnn_celoss_rows_k(float* logits, const float* targets, float* losses, int seq_len, int vocab) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq_len) {
        return;
    }
    float* row = logits + t * vocab;
    int tgt = (int)targets[t];
    if (tgt < 0) {
        tgt = 0;
    }
    if (tgt >= vocab) {
        tgt = vocab - 1;
    }
    float mx = row[0];
    for (int v = 1; v < vocab; ++v) {
        if (row[v] > mx) {
            mx = row[v];
        }
    }
    float sm = 0.0f;
    for (int v = 0; v < vocab; ++v) {
        row[v] = expf(row[v] - mx);
        sm += row[v];
    }
    float inv = (sm > 1e-20f) ? (1.0f / sm) : 0.0f;
    for (int v = 0; v < vocab; ++v) {
        row[v] *= inv;
    }
    float p = row[tgt];
    if (p < 1e-12f) {
        p = 1e-12f;
    }
    losses[t] = -logf(p);
    row[tgt] -= 1.0f;
}

}  /* namespace */

extern "C" QITNN_API const char* Qitnn_Version(void) {
    return "0.1.0";
}

extern "C" QITNN_API int Qitnn_Init(void) {
    return qitnn_init_cuda() ? 1 : 0;
}

extern "C" QITNN_API int Qitnn_IsReady(void) {
    return g_cuda_ready ? 1 : 0;
}

extern "C" QITNN_API void Qitnn_Pin(float* host, int count) {
    if (!qitnn_init_cuda() || host == nullptr || count <= 0) {
        return;
    }
    size_t bytes = (size_t)count * sizeof(float);

    DeviceSlot old = qitnn_lookup_slot(host);
    if (old.dev != nullptr) {
        if (old.bytes != bytes) {
            QITNN_CU(cudaFree(old.dev));
            void* dev = nullptr;
            QITNN_CU(cudaMalloc(&dev, bytes));
            QITNN_CU(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
            qitnn_store_slot(host, dev, bytes);
            return;
        }
        QITNN_CU(cudaMemcpy(old.dev, host, bytes, cudaMemcpyHostToDevice));
        return;
    }

    void* dev = nullptr;
    QITNN_CU(cudaMalloc(&dev, bytes));
    QITNN_CU(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    qitnn_store_slot(host, dev, bytes);
}

extern "C" QITNN_API void Qitnn_Unpin(float* host) {
    if (host == nullptr) {
        return;
    }
    DeviceSlot slot = qitnn_lookup_slot(host);
    if (slot.dev == nullptr) {
        return;
    }
    QITNN_CU(cudaFree(slot.dev));
    qitnn_erase_slot(host);
}

extern "C" QITNN_API void Qitnn_SyncDown(float* host, int count) {
    if (!qitnn_init_cuda() || host == nullptr || count <= 0) {
        return;
    }
    DeviceSlot slot = qitnn_lookup_slot(host);
    if (slot.dev == nullptr) {
        return;
    }
    size_t bytes = (size_t)count * sizeof(float);
    if (slot.bytes < bytes) {
        bytes = slot.bytes;
    }
    QITNN_CU(cudaDeviceSynchronize());
    QITNN_CU(cudaMemcpy(host, slot.dev, bytes, cudaMemcpyDeviceToHost));
}

extern "C" QITNN_API void Qitnn_SyncUp(float* host, int count) {
    if (!qitnn_init_cuda() || host == nullptr || count <= 0) {
        return;
    }
    DeviceSlot slot = qitnn_lookup_slot(host);
    if (slot.dev == nullptr) {
        return;
    }
    size_t bytes = (size_t)count * sizeof(float);
    if (slot.bytes < bytes) {
        bytes = slot.bytes;
    }
    QITNN_CU(cudaMemcpy(slot.dev, host, bytes, cudaMemcpyHostToDevice));
}

extern "C" QITNN_API void Qitnn_Copy(float* dst, float* src, int count) {
    if (!qitnn_init_cuda() || dst == nullptr || src == nullptr || count <= 0) {
        return;
    }

    size_t bytes = (size_t)count * sizeof(float);
    bool src_pinned = qitnn_is_pinned(src);
    bool dst_pinned = qitnn_is_pinned(dst);

    if (src_pinned && dst_pinned) {
        DeviceSlot src_slot = qitnn_lookup_slot(src);
        DeviceSlot dst_slot = qitnn_lookup_slot(dst);
        QITNN_CU(cudaMemcpy(dst_slot.dev, src_slot.dev, bytes, cudaMemcpyDeviceToDevice));
        return;
    }

    if (src_pinned) {
        DeviceSlot src_slot = qitnn_lookup_slot(src);
        QITNN_CU(cudaDeviceSynchronize());
        QITNN_CU(cudaMemcpy(dst, src_slot.dev, bytes, cudaMemcpyDeviceToHost));
        return;
    }

    if (dst_pinned) {
        DeviceSlot dst_slot = qitnn_lookup_slot(dst);
        QITNN_CU(cudaMemcpy(dst_slot.dev, src, bytes, cudaMemcpyHostToDevice));
        return;
    }

    std::memcpy(dst, src, bytes);
}

extern "C" QITNN_API void Qitnn_Sgd(float* host_w, float* host_g, float lr, int n) {
    if (!qitnn_init_cuda() || host_w == nullptr || host_g == nullptr || n <= 0) {
        return;
    }
    size_t bytes = (size_t)n * sizeof(float);
    bool stay = qitnn_all_pinned({host_w, host_g});
    float* dw = (float*)qitnn_dev(host_w, bytes, true);
    float* dg = (float*)qitnn_dev(host_g, bytes, true);
    qitnn_sgd_k<<<(n + 255) / 256, 256>>>(dw, dg, lr, n);
    QITNN_CU(cudaGetLastError());
    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }
    qitnn_undev(host_g, dg, 0, false);
    qitnn_undev(host_w, dw, bytes, true);
}

extern "C" QITNN_API void Qitnn_Prior(float* an, float* az, float* ap, float step, float entropy_floor, int count) {
    if (!qitnn_init_cuda() || an == nullptr || az == nullptr || ap == nullptr) {
        return;
    }
    if (count <= 0 || step <= 0.0f || entropy_floor <= 0.0f) {
        return;
    }
    size_t bytes = (size_t)count * sizeof(float);
    bool stay = qitnn_all_pinned({an, az, ap});
    float* d_an = (float*)qitnn_dev(an, bytes, true);
    float* d_az = (float*)qitnn_dev(az, bytes, true);
    float* d_ap = (float*)qitnn_dev(ap, bytes, true);
    qitnn_prior_k<<<(count + 255) / 256, 256>>>(d_an, d_az, d_ap, step, entropy_floor, count);
    QITNN_CU(cudaGetLastError());
    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }
    qitnn_undev(an, d_an, bytes, true);
    qitnn_undev(az, d_az, bytes, true);
    qitnn_undev(ap, d_ap, bytes, true);
}

extern "C" QITNN_API void Qitnn_Forward3(
    float* input,
    float* a_neg,
    float* a_zero,
    float* a_pos,
    float* out_u,
    float* out_v,
    float* out_cn,
    float* out_cz,
    float* out_cp,
    int rows,
    int in_dim,
    int out_dim
) {
    if (!qitnn_init_cuda()) {
        return;
    }
    size_t sz_in = (size_t)rows * in_dim * sizeof(float);
    size_t sz_w = (size_t)in_dim * out_dim * sizeof(float);
    size_t sz_out = (size_t)rows * out_dim * sizeof(float);
    bool stay = qitnn_all_pinned({input, a_neg, a_zero, a_pos, out_u, out_v, out_cn, out_cz, out_cp});

    float* d_in = (float*)qitnn_dev(input, sz_in, true);
    float* d_an = (float*)qitnn_dev(a_neg, sz_w, true);
    float* d_az = (float*)qitnn_dev(a_zero, sz_w, true);
    float* d_ap = (float*)qitnn_dev(a_pos, sz_w, true);
    float* d_u = (float*)qitnn_dev(out_u, sz_out, false);
    float* d_v = (float*)qitnn_dev(out_v, sz_out, false);
    float* d_cn = (float*)qitnn_dev(out_cn, sz_out, false);
    float* d_cz = (float*)qitnn_dev(out_cz, sz_out, false);
    float* d_cp = (float*)qitnn_dev(out_cp, sz_out, false);

    QITNN_CU(cudaMemset(d_cn, 0, sz_out));
    QITNN_CU(cudaMemset(d_cz, 0, sz_out));
    QITNN_CU(cudaMemset(d_cp, 0, sz_out));

    qitnn_gemm_row_major(d_in, d_an, d_cn, rows, out_dim, in_dim);
    qitnn_gemm_row_major(d_in, d_az, d_cz, rows, out_dim, in_dim);
    qitnn_gemm_row_major(d_in, d_ap, d_cp, rows, out_dim, in_dim);

    int count = rows * out_dim;
    qitnn_normalize3_k<<<(count + 255) / 256, 256>>>(d_cn, d_cz, d_cp, d_u, d_v, count);
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(input, d_in, sz_in, false);
    qitnn_undev(a_neg, d_an, sz_w, false);
    qitnn_undev(a_zero, d_az, sz_w, false);
    qitnn_undev(a_pos, d_ap, sz_w, false);
    qitnn_undev(out_u, d_u, sz_out, true);
    qitnn_undev(out_v, d_v, sz_out, true);
    qitnn_undev(out_cn, d_cn, sz_out, true);
    qitnn_undev(out_cz, d_cz, sz_out, true);
    qitnn_undev(out_cp, d_cp, sz_out, true);
}

extern "C" QITNN_API void Qitnn_BackNorm3(
    float* du,
    float* dv,
    float* cn,
    float* cz,
    float* cp,
    float* dcn,
    float* dcz,
    float* dcp,
    float ent_lambda,
    int count
) {
    if (!qitnn_init_cuda()) {
        return;
    }
    size_t bytes = (size_t)count * sizeof(float);
    bool stay = qitnn_all_pinned({du, dv, cn, cz, cp, dcn, dcz, dcp});

    float* d_du = (float*)qitnn_dev(du, bytes, true);
    float* d_dv = (float*)qitnn_dev(dv, bytes, true);
    float* d_cn = (float*)qitnn_dev(cn, bytes, true);
    float* d_cz = (float*)qitnn_dev(cz, bytes, true);
    float* d_cp = (float*)qitnn_dev(cp, bytes, true);
    float* d_dcn = (float*)qitnn_dev(dcn, bytes, false);
    float* d_dcz = (float*)qitnn_dev(dcz, bytes, false);
    float* d_dcp = (float*)qitnn_dev(dcp, bytes, false);

    qitnn_backnorm3_k<<<(count + 255) / 256, 256>>>(
        d_du,
        d_dv,
        d_cn,
        d_cz,
        d_cp,
        d_dcn,
        d_dcz,
        d_dcp,
        ent_lambda,
        count
    );
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(du, d_du, bytes, false);
    qitnn_undev(dv, d_dv, bytes, false);
    qitnn_undev(cn, d_cn, bytes, false);
    qitnn_undev(cz, d_cz, bytes, false);
    qitnn_undev(cp, d_cp, bytes, false);
    qitnn_undev(dcn, d_dcn, bytes, true);
    qitnn_undev(dcz, d_dcz, bytes, true);
    qitnn_undev(dcp, d_dcp, bytes, true);
}

extern "C" QITNN_API void Qitnn_CenteredSimplex(
    float* u,
    float* v,
    float* out_x,
    float* out_y,
    int count
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    size_t bytes = (size_t)count * sizeof(float);
    bool stay = qitnn_all_pinned({u, v, out_x, out_y});

    float* d_u = (float*)qitnn_dev(u, bytes, true);
    float* d_v = (float*)qitnn_dev(v, bytes, true);
    float* d_x = (float*)qitnn_dev(out_x, bytes, false);
    float* d_y = (float*)qitnn_dev(out_y, bytes, false);

    qitnn_centered_simplex_k<<<(count + 255) / 256, 256>>>(d_u, d_v, d_x, d_y, count);
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(u, d_u, bytes, false);
    qitnn_undev(v, d_v, bytes, false);
    qitnn_undev(out_x, d_x, bytes, true);
    qitnn_undev(out_y, d_y, bytes, true);
}

extern "C" QITNN_API void Qitnn_EmbedLookup(float* table, float* tokens, float* out, int seq_len, int dim, int vocab) {
    if (!qitnn_init_cuda()) {
        return;
    }
    int total = seq_len * dim;
    size_t sz_t = (size_t)vocab * dim * sizeof(float);
    size_t sz_tok = (size_t)seq_len * sizeof(float);
    size_t sz_out = (size_t)total * sizeof(float);
    bool stay = qitnn_all_pinned({table, tokens, out});

    float* d_t = (float*)qitnn_dev(table, sz_t, true);
    float* d_tok = (float*)qitnn_dev(tokens, sz_tok, true);
    float* d_out = (float*)qitnn_dev(out, sz_out, false);

    qitnn_seq_embed_k<<<(total + 255) / 256, 256>>>(d_t, d_tok, d_out, seq_len, dim, vocab);
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(table, d_t, sz_t, false);
    qitnn_undev(tokens, d_tok, sz_tok, false);
    qitnn_undev(out, d_out, sz_out, true);
}

extern "C" QITNN_API void Qitnn_EmbedBackward(
    float* table,
    float* tokens,
    float* grad,
    int seq_len,
    int dim,
    int vocab,
    float lr
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    size_t sz_t = (size_t)vocab * dim * sizeof(float);
    size_t sz_tok = (size_t)seq_len * sizeof(float);
    size_t sz_g = (size_t)seq_len * dim * sizeof(float);
    bool stay = qitnn_all_pinned({table, tokens, grad});

    float* d_t = (float*)qitnn_dev(table, sz_t, true);
    float* d_tok = (float*)qitnn_dev(tokens, sz_tok, true);
    float* d_g = (float*)qitnn_dev(grad, sz_g, true);

    int total = vocab * dim;
    qitnn_embed_backward_k<<<(total + 255) / 256, 256>>>(d_t, d_tok, d_g, seq_len, dim, vocab, lr);
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(table, d_t, sz_t, true);
    qitnn_undev(tokens, d_tok, sz_tok, false);
    qitnn_undev(grad, d_g, sz_g, false);
}

extern "C" QITNN_API void Qitnn_PosAdd(float* x, float* pos, int seq_len, int dim) {
    if (!qitnn_init_cuda()) {
        return;
    }
    int total = seq_len * dim;
    size_t bytes = (size_t)total * sizeof(float);
    bool stay = qitnn_all_pinned({x, pos});

    float* d_x = (float*)qitnn_dev(x, bytes, true);
    float* d_pos = (float*)qitnn_dev(pos, bytes, true);
    qitnn_pos_add_k<<<(total + 255) / 256, 256>>>(d_x, d_pos, total);
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(x, d_x, bytes, true);
    qitnn_undev(pos, d_pos, bytes, false);
}

extern "C" QITNN_API void Qitnn_LayerNormBatch(float* x, float* gamma, float* beta, int seq_len, int dim) {
    if (!qitnn_init_cuda()) {
        return;
    }
    size_t sz_x = (size_t)seq_len * dim * sizeof(float);
    size_t sz_p = (size_t)dim * sizeof(float);
    bool stay = qitnn_all_pinned({x, gamma, beta});

    float* d_x = (float*)qitnn_dev(x, sz_x, true);
    float* d_g = (float*)qitnn_dev(gamma, sz_p, true);
    float* d_b = (float*)qitnn_dev(beta, sz_p, true);

    qitnn_layernorm_batch_k<<<seq_len, 1>>>(d_x, d_g, d_b, seq_len, dim);
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(x, d_x, sz_x, true);
    qitnn_undev(gamma, d_g, sz_p, false);
    qitnn_undev(beta, d_b, sz_p, false);
}

extern "C" QITNN_API void Qitnn_Attention2(
    float* qx,
    float* qy,
    float* kx,
    float* ky,
    float* vx,
    float* vy,
    float* ox,
    float* oy,
    int seq_len,
    int dim
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    float scale = 1.0f / sqrtf(2.0f * (float)dim);
    size_t sz_vec = (size_t)seq_len * dim * sizeof(float);
    size_t sz_scores = (size_t)seq_len * seq_len * sizeof(float);
    bool stay = qitnn_all_pinned({qx, qy, kx, ky, vx, vy, ox, oy});

    float* d_qx = (float*)qitnn_dev(qx, sz_vec, true);
    float* d_qy = (float*)qitnn_dev(qy, sz_vec, true);
    float* d_kx = (float*)qitnn_dev(kx, sz_vec, true);
    float* d_ky = (float*)qitnn_dev(ky, sz_vec, true);
    float* d_vx = (float*)qitnn_dev(vx, sz_vec, true);
    float* d_vy = (float*)qitnn_dev(vy, sz_vec, true);
    float* d_ox = (float*)qitnn_dev(ox, sz_vec, false);
    float* d_oy = (float*)qitnn_dev(oy, sz_vec, false);

    /* scores buffer hangs around. boring but useful. */
    qitnn_ensure_buffer(g_scores, g_scores_cap, sz_scores);

    qitnn_attn_scores2_k<<<((seq_len * seq_len) + 255) / 256, 256>>>(
        d_qx,
        d_qy,
        d_kx,
        d_ky,
        g_scores,
        seq_len,
        dim,
        scale
    );
    qitnn_attn_softmax_rows_k<<<(seq_len + 255) / 256, 256>>>(g_scores, seq_len);
    qitnn_attn_apply2_k<<<((seq_len * dim) + 255) / 256, 256>>>(
        g_scores,
        d_vx,
        d_vy,
        d_ox,
        d_oy,
        seq_len,
        dim
    );
    QITNN_CU(cudaGetLastError());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    qitnn_undev(qx, d_qx, sz_vec, false);
    qitnn_undev(qy, d_qy, sz_vec, false);
    qitnn_undev(kx, d_kx, sz_vec, false);
    qitnn_undev(ky, d_ky, sz_vec, false);
    qitnn_undev(vx, d_vx, sz_vec, false);
    qitnn_undev(vy, d_vy, sz_vec, false);
    qitnn_undev(ox, d_ox, sz_vec, true);
    qitnn_undev(oy, d_oy, sz_vec, true);
}

extern "C" QITNN_API void Qitnn_AttentionBackward2(
    float* dqx,
    float* dqy,
    float* dkx,
    float* dky,
    float* dvx,
    float* dvy,
    float* qx,
    float* qy,
    float* kx,
    float* ky,
    float* vx,
    float* vy,
    float* dox,
    float* doy,
    int seq_len,
    int dim
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    float scale = 1.0f / sqrtf(2.0f * (float)dim);
    size_t sz_vec = (size_t)seq_len * dim * sizeof(float);
    size_t sz_scores = (size_t)seq_len * seq_len * sizeof(float);
    bool stay = qitnn_all_pinned({dqx, dqy, dkx, dky, dvx, dvy, qx, qy, kx, ky, vx, vy, dox, doy});

    float* ddqx = (float*)qitnn_dev(dqx, sz_vec, false);
    float* ddqy = (float*)qitnn_dev(dqy, sz_vec, false);
    float* ddkx = (float*)qitnn_dev(dkx, sz_vec, false);
    float* ddky = (float*)qitnn_dev(dky, sz_vec, false);
    float* ddvx = (float*)qitnn_dev(dvx, sz_vec, false);
    float* ddvy = (float*)qitnn_dev(dvy, sz_vec, false);
    float* d_qx = (float*)qitnn_dev(qx, sz_vec, true);
    float* d_qy = (float*)qitnn_dev(qy, sz_vec, true);
    float* d_kx = (float*)qitnn_dev(kx, sz_vec, true);
    float* d_ky = (float*)qitnn_dev(ky, sz_vec, true);
    float* d_vx = (float*)qitnn_dev(vx, sz_vec, true);
    float* d_vy = (float*)qitnn_dev(vy, sz_vec, true);
    float* d_dox = (float*)qitnn_dev(dox, sz_vec, true);
    float* d_doy = (float*)qitnn_dev(doy, sz_vec, true);

    float* d_attn = nullptr;
    float* d_tmp_attn = nullptr;
    float* d_scores = nullptr;
    QITNN_CU(cudaMalloc(&d_attn, sz_scores));
    QITNN_CU(cudaMalloc(&d_tmp_attn, sz_scores));
    QITNN_CU(cudaMalloc(&d_scores, sz_scores));

    qitnn_attn_scores2_k<<<((seq_len * seq_len) + 255) / 256, 256>>>(d_qx, d_qy, d_kx, d_ky, d_attn, seq_len, dim, scale);
    qitnn_attn_softmax_rows_k<<<(seq_len + 255) / 256, 256>>>(d_attn, seq_len);
    qitnn_attn_dv2_k<<<((seq_len * dim) + 255) / 256, 256>>>(d_attn, d_dox, d_doy, ddvx, ddvy, seq_len, dim);
    qitnn_attn_dattn2_k<<<((seq_len * seq_len) + 255) / 256, 256>>>(d_dox, d_doy, d_vx, d_vy, d_tmp_attn, seq_len, dim);
    qitnn_attn_dscores2_k<<<(seq_len + 255) / 256, 256>>>(d_attn, d_tmp_attn, d_scores, seq_len, scale);
    qitnn_attn_dq2_k<<<((seq_len * dim) + 255) / 256, 256>>>(d_scores, d_kx, d_ky, ddqx, ddqy, seq_len, dim);
    qitnn_attn_dk2_k<<<((seq_len * dim) + 255) / 256, 256>>>(d_scores, d_qx, d_qy, ddkx, ddky, seq_len, dim);
    QITNN_CU(cudaGetLastError());
    QITNN_CU(cudaDeviceSynchronize());

    if (!stay) {
        QITNN_CU(cudaDeviceSynchronize());
    }

    QITNN_CU(cudaFree(d_attn));
    QITNN_CU(cudaFree(d_tmp_attn));
    QITNN_CU(cudaFree(d_scores));

    qitnn_undev(dqx, ddqx, sz_vec, true);
    qitnn_undev(dqy, ddqy, sz_vec, true);
    qitnn_undev(dkx, ddkx, sz_vec, true);
    qitnn_undev(dky, ddky, sz_vec, true);
    qitnn_undev(dvx, ddvx, sz_vec, true);
    qitnn_undev(dvy, ddvy, sz_vec, true);
    qitnn_undev(qx, d_qx, sz_vec, false);
    qitnn_undev(qy, d_qy, sz_vec, false);
    qitnn_undev(kx, d_kx, sz_vec, false);
    qitnn_undev(ky, d_ky, sz_vec, false);
    qitnn_undev(vx, d_vx, sz_vec, false);
    qitnn_undev(vy, d_vy, sz_vec, false);
    qitnn_undev(dox, d_dox, sz_vec, false);
    qitnn_undev(doy, d_doy, sz_vec, false);
}

extern "C" QITNN_API double Qitnn_CELoss(float* logits, float* targets, int seq_len, int vocab) {
    if (!qitnn_init_cuda()) {
        return 0.0;
    }

    size_t sz_logits = (size_t)seq_len * vocab * sizeof(float);
    size_t sz_targets = (size_t)seq_len * sizeof(float);
    size_t sz_loss = (size_t)seq_len * sizeof(float);

    float* d_logits = (float*)qitnn_dev(logits, sz_logits, true);
    float* d_targets = (float*)qitnn_dev(targets, sz_targets, true);
    qitnn_ensure_buffer(g_loss, g_loss_cap, sz_loss);

    qitnn_celoss_rows_k<<<(seq_len + 255) / 256, 256>>>(d_logits, d_targets, g_loss, seq_len, vocab);
    QITNN_CU(cudaGetLastError());
    QITNN_CU(cudaDeviceSynchronize());

    std::vector<float> host_loss(seq_len, 0.0f);
    QITNN_CU(cudaMemcpy(host_loss.data(), g_loss, sz_loss, cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (float v : host_loss) {
        total += (double)v;
    }

    qitnn_undev(logits, d_logits, sz_logits, true);
    qitnn_undev(targets, d_targets, sz_targets, false);
    return total;
}

extern "C" QITNN_API void Qitnn_DeviceForward3(
    const float* d_input,
    const float* d_a_neg,
    const float* d_a_zero,
    const float* d_a_pos,
    float* d_out_u,
    float* d_out_v,
    float* d_out_cn,
    float* d_out_cz,
    float* d_out_cp,
    int rows,
    int in_dim,
    int out_dim
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    size_t sz_out = (size_t)rows * out_dim * sizeof(float);
    QITNN_CU(cudaMemset(d_out_cn, 0, sz_out));
    QITNN_CU(cudaMemset(d_out_cz, 0, sz_out));
    QITNN_CU(cudaMemset(d_out_cp, 0, sz_out));

    qitnn_gemm_row_major(d_input, d_a_neg, d_out_cn, rows, out_dim, in_dim);
    qitnn_gemm_row_major(d_input, d_a_zero, d_out_cz, rows, out_dim, in_dim);
    qitnn_gemm_row_major(d_input, d_a_pos, d_out_cp, rows, out_dim, in_dim);

    int count = rows * out_dim;
    qitnn_normalize3_k<<<(count + 255) / 256, 256>>>(d_out_cn, d_out_cz, d_out_cp, d_out_u, d_out_v, count);
    QITNN_CU(cudaGetLastError());
}

extern "C" QITNN_API void Qitnn_DeviceBackNorm3(
    const float* d_du,
    const float* d_dv,
    const float* d_cn,
    const float* d_cz,
    const float* d_cp,
    float* d_dcn,
    float* d_dcz,
    float* d_dcp,
    float ent_lambda,
    int count
) {
    if (!qitnn_init_cuda()) {
        return;
    }
    qitnn_backnorm3_k<<<(count + 255) / 256, 256>>>(
        d_du,
        d_dv,
        d_cn,
        d_cz,
        d_cp,
        d_dcn,
        d_dcz,
        d_dcp,
        ent_lambda,
        count
    );
    QITNN_CU(cudaGetLastError());
}

extern "C" QITNN_API void Qitnn_DevicePrior(
    float* d_an,
    float* d_az,
    float* d_ap,
    float step,
    float entropy_floor,
    int count
) {
    if (!qitnn_init_cuda()) {
        return;
    }
    if (d_an == nullptr || d_az == nullptr || d_ap == nullptr) {
        return;
    }
    if (count <= 0 || step <= 0.0f || entropy_floor <= 0.0f) {
        return;
    }
    qitnn_prior_k<<<(count + 255) / 256, 256>>>(d_an, d_az, d_ap, step, entropy_floor, count);
    QITNN_CU(cudaGetLastError());
}

extern "C" QITNN_API void Qitnn_DeviceCenteredSimplex(
    const float* d_u,
    const float* d_v,
    float* d_out_x,
    float* d_out_y,
    int count
) {
    if (!qitnn_init_cuda()) {
        return;
    }
    qitnn_centered_simplex_k<<<(count + 255) / 256, 256>>>(d_u, d_v, d_out_x, d_out_y, count);
    QITNN_CU(cudaGetLastError());
}

extern "C" QITNN_API void Qitnn_DeviceAttention2(
    const float* d_qx,
    const float* d_qy,
    const float* d_kx,
    const float* d_ky,
    const float* d_vx,
    const float* d_vy,
    float* d_ox,
    float* d_oy,
    int seq_len,
    int dim
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    float scale = 1.0f / sqrtf(2.0f * (float)dim);
    size_t sz_scores = (size_t)seq_len * seq_len * sizeof(float);
    qitnn_ensure_buffer(g_scores, g_scores_cap, sz_scores);

    qitnn_attn_scores2_k<<<((seq_len * seq_len) + 255) / 256, 256>>>(
        d_qx,
        d_qy,
        d_kx,
        d_ky,
        g_scores,
        seq_len,
        dim,
        scale
    );
    qitnn_attn_softmax_rows_k<<<(seq_len + 255) / 256, 256>>>(g_scores, seq_len);
    qitnn_attn_apply2_k<<<((seq_len * dim) + 255) / 256, 256>>>(
        g_scores,
        d_vx,
        d_vy,
        d_ox,
        d_oy,
        seq_len,
        dim
    );
    QITNN_CU(cudaGetLastError());
}

extern "C" QITNN_API void Qitnn_DeviceAttentionBackward2(
    float* d_dqx,
    float* d_dqy,
    float* d_dkx,
    float* d_dky,
    float* d_dvx,
    float* d_dvy,
    const float* d_qx,
    const float* d_qy,
    const float* d_kx,
    const float* d_ky,
    const float* d_vx,
    const float* d_vy,
    const float* d_dox,
    const float* d_doy,
    int seq_len,
    int dim
) {
    if (!qitnn_init_cuda()) {
        return;
    }

    float scale = 1.0f / sqrtf(2.0f * (float)dim);
    size_t sz_scores = (size_t)seq_len * seq_len * sizeof(float);

    float* d_attn = nullptr;
    float* d_tmp_attn = nullptr;
    float* d_scores = nullptr;
    QITNN_CU(cudaMalloc(&d_attn, sz_scores));
    QITNN_CU(cudaMalloc(&d_tmp_attn, sz_scores));
    QITNN_CU(cudaMalloc(&d_scores, sz_scores));

    qitnn_attn_scores2_k<<<((seq_len * seq_len) + 255) / 256, 256>>>(d_qx, d_qy, d_kx, d_ky, d_attn, seq_len, dim, scale);
    qitnn_attn_softmax_rows_k<<<(seq_len + 255) / 256, 256>>>(d_attn, seq_len);
    qitnn_attn_dv2_k<<<((seq_len * dim) + 255) / 256, 256>>>(d_attn, d_dox, d_doy, d_dvx, d_dvy, seq_len, dim);
    qitnn_attn_dattn2_k<<<((seq_len * seq_len) + 255) / 256, 256>>>(d_dox, d_doy, d_vx, d_vy, d_tmp_attn, seq_len, dim);
    qitnn_attn_dscores2_k<<<(seq_len + 255) / 256, 256>>>(d_attn, d_tmp_attn, d_scores, seq_len, scale);
    qitnn_attn_dq2_k<<<((seq_len * dim) + 255) / 256, 256>>>(d_scores, d_kx, d_ky, d_dqx, d_dqy, seq_len, dim);
    qitnn_attn_dk2_k<<<((seq_len * dim) + 255) / 256, 256>>>(d_scores, d_qx, d_qy, d_dkx, d_dky, seq_len, dim);
    QITNN_CU(cudaGetLastError());
    QITNN_CU(cudaDeviceSynchronize());

    QITNN_CU(cudaFree(d_attn));
    QITNN_CU(cudaFree(d_tmp_attn));
    QITNN_CU(cudaFree(d_scores));
}
