#include <torch/extension.h>

#include <tuple>

#include "qitnn/qitnn_device.h"

namespace {

void check_cuda_f32_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_amp_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.scalar_type() == torch::kFloat32 ||
        t.scalar_type() == torch::kFloat16 ||
        t.scalar_type() == torch::kBFloat16,
        name,
        " must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_same_shape(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.sizes() == b.sizes(), a_name, " shape must match ", b_name);
}

void check_same_dtype(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), a_name, " dtype must match ", b_name);
}

int tensor_dtype_code(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kFloat32:
            return QITNN_DTYPE_FLOAT32;
        case torch::kFloat16:
            return QITNN_DTYPE_FLOAT16;
        case torch::kBFloat16:
            return QITNN_DTYPE_BFLOAT16;
        default:
            TORCH_CHECK(false, "unsupported dtype");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward3_cuda(
    torch::Tensor input,
    torch::Tensor a_neg,
    torch::Tensor a_zero,
    torch::Tensor a_pos
) {
    check_cuda_amp_2d(input, "input");
    check_cuda_f32_2d(a_neg, "a_neg");
    check_cuda_f32_2d(a_zero, "a_zero");
    check_cuda_f32_2d(a_pos, "a_pos");

    TORCH_CHECK(input.size(1) == a_neg.size(0), "input.size(1) must match a_neg.size(0)");
    TORCH_CHECK(a_neg.sizes() == a_zero.sizes(), "a_zero shape must match a_neg");
    TORCH_CHECK(a_neg.sizes() == a_pos.sizes(), "a_pos shape must match a_neg");
    TORCH_CHECK(input.get_device() == 0, "stage2 bridge currently supports only cuda:0");
    TORCH_CHECK(a_neg.get_device() == 0, "stage2 bridge currently supports only cuda:0");
    TORCH_CHECK(a_zero.get_device() == 0, "stage2 bridge currently supports only cuda:0");
    TORCH_CHECK(a_pos.get_device() == 0, "stage2 bridge currently supports only cuda:0");

    const auto rows = static_cast<int>(input.size(0));
    const auto in_dim = static_cast<int>(input.size(1));
    const auto out_dim = static_cast<int>(a_neg.size(1));

    auto out_u = torch::empty({rows, out_dim}, input.options());
    auto out_v = torch::empty({rows, out_dim}, input.options());
    auto fp32_opts = input.options().dtype(torch::kFloat32);
    auto out_cn = torch::empty({rows, out_dim}, fp32_opts);
    auto out_cz = torch::empty({rows, out_dim}, fp32_opts);
    auto out_cp = torch::empty({rows, out_dim}, fp32_opts);

    Qitnn_DeviceForward3Ex(
        input.data_ptr(),
        tensor_dtype_code(input),
        a_neg.data_ptr<float>(),
        a_zero.data_ptr<float>(),
        a_pos.data_ptr<float>(),
        out_u.data_ptr(),
        out_v.data_ptr(),
        tensor_dtype_code(out_u),
        out_cn.data_ptr<float>(),
        out_cz.data_ptr<float>(),
        out_cp.data_ptr<float>(),
        rows,
        in_dim,
        out_dim
    );

    return {out_u, out_v, out_cn, out_cz, out_cp};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backnorm3_cuda(
    torch::Tensor du,
    torch::Tensor dv,
    torch::Tensor cn,
    torch::Tensor cz,
    torch::Tensor cp,
    double ent_lambda
) {
    check_cuda_amp_2d(du, "du");
    check_cuda_amp_2d(dv, "dv");
    check_same_dtype(du, dv, "du", "dv");
    check_cuda_f32_2d(cn, "cn");
    check_cuda_f32_2d(cz, "cz");
    check_cuda_f32_2d(cp, "cp");

    check_same_shape(du, dv, "du", "dv");
    check_same_shape(du, cn, "du", "cn");
    check_same_shape(du, cz, "du", "cz");
    check_same_shape(du, cp, "du", "cp");

    TORCH_CHECK(du.get_device() == 0, "stage4 bridge currently supports only cuda:0");
    TORCH_CHECK(dv.get_device() == 0, "stage4 bridge currently supports only cuda:0");
    TORCH_CHECK(cn.get_device() == 0, "stage4 bridge currently supports only cuda:0");
    TORCH_CHECK(cz.get_device() == 0, "stage4 bridge currently supports only cuda:0");
    TORCH_CHECK(cp.get_device() == 0, "stage4 bridge currently supports only cuda:0");

    auto dcn = torch::empty_like(cn);
    auto dcz = torch::empty_like(cz);
    auto dcp = torch::empty_like(cp);

    Qitnn_DeviceBackNorm3Ex(
        du.data_ptr(),
        dv.data_ptr(),
        tensor_dtype_code(du),
        cn.data_ptr<float>(),
        cz.data_ptr<float>(),
        cp.data_ptr<float>(),
        dcn.data_ptr<float>(),
        dcz.data_ptr<float>(),
        dcp.data_ptr<float>(),
        static_cast<float>(ent_lambda),
        static_cast<int>(cn.numel())
    );

    return {dcn, dcz, dcp};
}

void prior_cuda(
    torch::Tensor a_neg,
    torch::Tensor a_zero,
    torch::Tensor a_pos,
    double step,
    double entropy_floor
) {
    check_cuda_amp_2d(a_neg, "a_neg");
    check_cuda_amp_2d(a_zero, "a_zero");
    check_cuda_amp_2d(a_pos, "a_pos");

    check_same_shape(a_neg, a_zero, "a_neg", "a_zero");
    check_same_shape(a_neg, a_pos, "a_neg", "a_pos");
    check_same_dtype(a_neg, a_zero, "a_neg", "a_zero");
    check_same_dtype(a_neg, a_pos, "a_neg", "a_pos");

    TORCH_CHECK(a_neg.get_device() == 0, "prior bridge currently supports only cuda:0");
    TORCH_CHECK(a_zero.get_device() == 0, "prior bridge currently supports only cuda:0");
    TORCH_CHECK(a_pos.get_device() == 0, "prior bridge currently supports only cuda:0");

    Qitnn_DevicePriorEx(
        a_neg.data_ptr(),
        a_zero.data_ptr(),
        a_pos.data_ptr(),
        tensor_dtype_code(a_neg),
        static_cast<float>(step),
        static_cast<float>(entropy_floor),
        static_cast<int>(a_neg.numel())
    );
}

std::tuple<torch::Tensor, torch::Tensor> centered_simplex_cuda(
    torch::Tensor u,
    torch::Tensor v
) {
    check_cuda_amp_2d(u, "u");
    check_cuda_amp_2d(v, "v");
    check_same_shape(u, v, "u", "v");
    check_same_dtype(u, v, "u", "v");

    TORCH_CHECK(u.get_device() == 0, "centered_simplex bridge currently supports only cuda:0");
    TORCH_CHECK(v.get_device() == 0, "centered_simplex bridge currently supports only cuda:0");

    auto out_x = torch::empty_like(u);
    auto out_y = torch::empty_like(v);

    Qitnn_DeviceCenteredSimplexEx(
        u.data_ptr(),
        v.data_ptr(),
        tensor_dtype_code(u),
        out_x.data_ptr(),
        out_y.data_ptr(),
        tensor_dtype_code(out_x),
        static_cast<int>(u.numel())
    );

    return {out_x, out_y};
}

std::tuple<torch::Tensor, torch::Tensor> attention2_cuda(
    torch::Tensor qx,
    torch::Tensor qy,
    torch::Tensor kx,
    torch::Tensor ky,
    torch::Tensor vx,
    torch::Tensor vy
) {
    check_cuda_amp_2d(qx, "qx");
    check_cuda_amp_2d(qy, "qy");
    check_cuda_amp_2d(kx, "kx");
    check_cuda_amp_2d(ky, "ky");
    check_cuda_amp_2d(vx, "vx");
    check_cuda_amp_2d(vy, "vy");

    check_same_shape(qx, qy, "qx", "qy");
    check_same_shape(qx, kx, "qx", "kx");
    check_same_shape(qx, ky, "qx", "ky");
    check_same_shape(qx, vx, "qx", "vx");
    check_same_shape(qx, vy, "qx", "vy");
    check_same_dtype(qx, qy, "qx", "qy");
    check_same_dtype(qx, kx, "qx", "kx");
    check_same_dtype(qx, ky, "qx", "ky");
    check_same_dtype(qx, vx, "qx", "vx");
    check_same_dtype(qx, vy, "qx", "vy");

    TORCH_CHECK(qx.get_device() == 0, "attention2 bridge currently supports only cuda:0");
    TORCH_CHECK(qy.get_device() == 0, "attention2 bridge currently supports only cuda:0");
    TORCH_CHECK(kx.get_device() == 0, "attention2 bridge currently supports only cuda:0");
    TORCH_CHECK(ky.get_device() == 0, "attention2 bridge currently supports only cuda:0");
    TORCH_CHECK(vx.get_device() == 0, "attention2 bridge currently supports only cuda:0");
    TORCH_CHECK(vy.get_device() == 0, "attention2 bridge currently supports only cuda:0");

    const auto seq_len = static_cast<int>(qx.size(0));
    const auto dim = static_cast<int>(qx.size(1));

    auto ox = torch::empty_like(qx);
    auto oy = torch::empty_like(qy);

    Qitnn_DeviceAttention2Ex(
        qx.data_ptr(),
        qy.data_ptr(),
        kx.data_ptr(),
        ky.data_ptr(),
        vx.data_ptr(),
        vy.data_ptr(),
        tensor_dtype_code(qx),
        ox.data_ptr(),
        oy.data_ptr(),
        tensor_dtype_code(ox),
        seq_len,
        dim
    );

    return {ox, oy};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> attention_backward2_cuda(
    torch::Tensor dox,
    torch::Tensor doy,
    torch::Tensor qx,
    torch::Tensor qy,
    torch::Tensor kx,
    torch::Tensor ky,
    torch::Tensor vx,
    torch::Tensor vy
) {
    check_cuda_amp_2d(dox, "dox");
    check_cuda_amp_2d(doy, "doy");
    check_cuda_amp_2d(qx, "qx");
    check_cuda_amp_2d(qy, "qy");
    check_cuda_amp_2d(kx, "kx");
    check_cuda_amp_2d(ky, "ky");
    check_cuda_amp_2d(vx, "vx");
    check_cuda_amp_2d(vy, "vy");

    check_same_shape(dox, doy, "dox", "doy");
    check_same_shape(dox, qx, "dox", "qx");
    check_same_shape(qx, qy, "qx", "qy");
    check_same_shape(qx, kx, "qx", "kx");
    check_same_shape(qx, ky, "qx", "ky");
    check_same_shape(qx, vx, "qx", "vx");
    check_same_shape(qx, vy, "qx", "vy");
    check_same_dtype(dox, doy, "dox", "doy");
    check_same_dtype(dox, qx, "dox", "qx");
    check_same_dtype(qx, qy, "qx", "qy");
    check_same_dtype(qx, kx, "qx", "kx");
    check_same_dtype(qx, ky, "qx", "ky");
    check_same_dtype(qx, vx, "qx", "vx");
    check_same_dtype(qx, vy, "qx", "vy");

    TORCH_CHECK(dox.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(doy.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(qx.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(qy.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(kx.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(ky.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(vx.get_device() == 0, "attention backward bridge currently supports only cuda:0");
    TORCH_CHECK(vy.get_device() == 0, "attention backward bridge currently supports only cuda:0");

    const auto seq_len = static_cast<int>(qx.size(0));
    const auto dim = static_cast<int>(qx.size(1));

    auto dqx = torch::empty_like(qx);
    auto dqy = torch::empty_like(qy);
    auto dkx = torch::empty_like(kx);
    auto dky = torch::empty_like(ky);
    auto dvx = torch::empty_like(vx);
    auto dvy = torch::empty_like(vy);

    Qitnn_DeviceAttentionBackward2Ex(
        dqx.data_ptr(),
        dqy.data_ptr(),
        dkx.data_ptr(),
        dky.data_ptr(),
        dvx.data_ptr(),
        dvy.data_ptr(),
        tensor_dtype_code(dqx),
        qx.data_ptr(),
        qy.data_ptr(),
        kx.data_ptr(),
        ky.data_ptr(),
        vx.data_ptr(),
        vy.data_ptr(),
        dox.data_ptr(),
        doy.data_ptr(),
        tensor_dtype_code(qx),
        seq_len,
        dim
    );

    return {dqx, dqy, dkx, dky, dvx, dvy};
}

}  /* namespace */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward3_cuda", &forward3_cuda, "libQITNN forward3 (CUDA)");
    m.def("backnorm3_cuda", &backnorm3_cuda, "libQITNN backnorm3 (CUDA)");
    m.def("prior_cuda", &prior_cuda, "libQITNN prior (CUDA)");
    m.def("centered_simplex_cuda", &centered_simplex_cuda, "libQITNN centered simplex (CUDA)");
    m.def("attention2_cuda", &attention2_cuda, "libQITNN attention2 (CUDA)");
    m.def("attention_backward2_cuda", &attention_backward2_cuda, "libQITNN attention backward2 (CUDA)");
}
