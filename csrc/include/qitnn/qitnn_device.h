#pragma once

#include "qitnn/qitnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
device-first lane
caller owns device pointers
no pin map
no host copy circus
*/

#define QITNN_DTYPE_FLOAT32 0
#define QITNN_DTYPE_FLOAT16 1
#define QITNN_DTYPE_BFLOAT16 2

QITNN_API void Qitnn_DeviceForward3(
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
);

QITNN_API void Qitnn_DeviceBackNorm3(
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
);

QITNN_API void Qitnn_DeviceBackNorm3Ex(
    const void* d_du,
    const void* d_dv,
    int grad_dtype,
    const float* d_cn,
    const float* d_cz,
    const float* d_cp,
    float* d_dcn,
    float* d_dcz,
    float* d_dcp,
    float ent_lambda,
    int count
);

QITNN_API void Qitnn_DevicePrior(
    float* d_an,
    float* d_az,
    float* d_ap,
    float step,
    float entropy_floor,
    int count
);

QITNN_API void Qitnn_DevicePriorEx(
    void* d_an,
    void* d_az,
    void* d_ap,
    int dtype,
    float step,
    float entropy_floor,
    int count
);

QITNN_API void Qitnn_DeviceCenteredSimplex(
    const float* d_u,
    const float* d_v,
    float* d_out_x,
    float* d_out_y,
    int count
);

QITNN_API void Qitnn_DeviceCenteredSimplexEx(
    const void* d_u,
    const void* d_v,
    int in_dtype,
    void* d_out_x,
    void* d_out_y,
    int out_dtype,
    int count
);

QITNN_API void Qitnn_DeviceAttention2(
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
);

QITNN_API void Qitnn_DeviceAttention2Ex(
    const void* d_qx,
    const void* d_qy,
    const void* d_kx,
    const void* d_ky,
    const void* d_vx,
    const void* d_vy,
    int dtype,
    void* d_ox,
    void* d_oy,
    int out_dtype,
    int seq_len,
    int dim
);

QITNN_API void Qitnn_DeviceAttention2Batched(
    const float* d_qx,
    const float* d_qy,
    const float* d_kx,
    const float* d_ky,
    const float* d_vx,
    const float* d_vy,
    float* d_ox,
    float* d_oy,
    int batch,
    int seq_len,
    int dim
);

QITNN_API void Qitnn_DeviceAttention2BatchedEx(
    const void* d_qx,
    const void* d_qy,
    const void* d_kx,
    const void* d_ky,
    const void* d_vx,
    const void* d_vy,
    int dtype,
    void* d_ox,
    void* d_oy,
    int out_dtype,
    int batch,
    int seq_len,
    int dim
);

QITNN_API void Qitnn_DeviceAttentionBackward2(
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
);

QITNN_API void Qitnn_DeviceAttentionBackward2Batched(
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
    int batch,
    int seq_len,
    int dim
);

QITNN_API void Qitnn_DeviceAttentionBackward2Ex(
    void* d_dqx,
    void* d_dqy,
    void* d_dkx,
    void* d_dky,
    void* d_dvx,
    void* d_dvy,
    int out_dtype,
    const void* d_qx,
    const void* d_qy,
    const void* d_kx,
    const void* d_ky,
    const void* d_vx,
    const void* d_vy,
    const void* d_dox,
    const void* d_doy,
    int in_dtype,
    int seq_len,
    int dim
);

QITNN_API void Qitnn_DeviceAttentionBackward2BatchedEx(
    void* d_dqx,
    void* d_dqy,
    void* d_dkx,
    void* d_dky,
    void* d_dvx,
    void* d_dvy,
    int out_dtype,
    const void* d_qx,
    const void* d_qy,
    const void* d_kx,
    const void* d_ky,
    const void* d_vx,
    const void* d_vy,
    const void* d_dox,
    const void* d_doy,
    int in_dtype,
    int batch,
    int seq_len,
    int dim
);

QITNN_API void Qitnn_DeviceForward3Ex(
    const void* d_input,
    int input_dtype,
    const float* d_a_neg,
    const float* d_a_zero,
    const float* d_a_pos,
    void* d_out_u,
    void* d_out_v,
    int uv_dtype,
    float* d_out_cn,
    float* d_out_cz,
    float* d_out_cp,
    int rows,
    int in_dim,
    int out_dim
);

#ifdef __cplusplus
}
#endif
