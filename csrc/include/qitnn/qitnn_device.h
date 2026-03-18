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

QITNN_API void Qitnn_DevicePrior(
    float* d_an,
    float* d_az,
    float* d_ap,
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

#ifdef __cplusplus
}
#endif
