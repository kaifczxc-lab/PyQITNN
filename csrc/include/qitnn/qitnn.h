#pragma once

#include <stddef.h>

#if defined(_WIN32)
#  if defined(QITNN_BUILD)
#    define QITNN_API __declspec(dllexport)
#  else
#    define QITNN_API __declspec(dllimport)
#  endif
#else
#  define QITNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QITNN_API const char* Qitnn_Version(void);
QITNN_API int Qitnn_Init(void);
QITNN_API int Qitnn_IsReady(void);

QITNN_API void Qitnn_Pin(float* host, int count);
QITNN_API void Qitnn_Unpin(float* host);
QITNN_API void Qitnn_SyncDown(float* host, int count);
QITNN_API void Qitnn_SyncUp(float* host, int count);
QITNN_API void Qitnn_Copy(float* dst, float* src, int count);

QITNN_API void Qitnn_Sgd(float* host_w, float* host_g, float lr, int n);
QITNN_API void Qitnn_Prior(float* an, float* az, float* ap, float step, float entropy_floor, int count);

QITNN_API void Qitnn_Forward3(
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
);

QITNN_API void Qitnn_BackNorm3(
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
);

QITNN_API void Qitnn_CenteredSimplex(
    float* u,
    float* v,
    float* out_x,
    float* out_y,
    int count
);

QITNN_API void Qitnn_EmbedLookup(
    float* table,
    float* tokens,
    float* out,
    int seq_len,
    int dim,
    int vocab
);

QITNN_API void Qitnn_EmbedBackward(
    float* table,
    float* tokens,
    float* grad,
    int seq_len,
    int dim,
    int vocab,
    float lr
);

QITNN_API void Qitnn_PosAdd(float* x, float* pos, int seq_len, int dim);
QITNN_API void Qitnn_LayerNormBatch(float* x, float* gamma, float* beta, int seq_len, int dim);

QITNN_API void Qitnn_Attention2(
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
);

QITNN_API void Qitnn_AttentionBackward2(
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
);

QITNN_API double Qitnn_CELoss(float* logits, float* targets, int seq_len, int vocab);

#ifdef __cplusplus
}
#endif
