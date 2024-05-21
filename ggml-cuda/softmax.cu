#include "common.cuh"
#include "softmax.cuh"

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

template <bool vals_smem, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(const float * x, const T * mask, float * dst, const int ncols_par, const int nrows_y, const float scale, const float max_bias, const float m0, const float m1, uint32_t n_head_log2) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(max_bias, rowx/nrows_y, n_head_log2, m0, m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_max = data_soft_max_f32; // shared memory buffer for inter-warp communication (for max values)
    float * buf_den = data_soft_max_f32 + WARP_SIZE; // shared memory buffer for inter-warp communication (for sum(exp(v - max)))
    // shared memory buffer to cache values between iterations:
    float * vals = vals_smem ? buf_max + 2 * WARP_SIZE : dst + (int64_t)rowx*ncols;

    float max_val = -INFINITY;
    float den = 1;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = x[ix]*scale + (mask ? slope*t2f32(mask[iy]) : 0.0f);

        if (vals_smem) {
            vals[col] = val;
        }

        float last_max_val = max_val;
        max_val = max(max_val, val);
        den = (isinf(last_max_val) ? 0.F : den * expf(last_max_val - max_val)) +
              expf(val - max_val);
    }

#pragma unroll
    for (int radix = 1; radix < 32; radix <<= 1) {
        constexpr auto shfl_mask = static_cast<uint32_t>(-1);
        float butterflied_v = __shfl_xor_sync(shfl_mask, max_val, radix);
        float butterflied_d = __shfl_xor_sync(shfl_mask, den, radix);

        float this_max_val = max(max_val, butterflied_v);

        den = isinf(this_max_val) ? 0
                           : (den * exp(max_val - this_max_val) +
                              butterflied_d * exp(butterflied_v - this_max_val));
        max_val = this_max_val;
    }

    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_max[lane_id] = -INFINITY;
            buf_den[lane_id] = 1;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_max[warp_id] = max_val;
            buf_den[warp_id] = den;
        }
        __syncthreads();

        max_val = buf_max[lane_id];
        den = buf_den[lane_id];
        
#pragma unroll
        for (int radix = 1; radix < 32; radix <<= 1) {
            constexpr auto shfl_mask = static_cast<uint32_t>(-1);
            float butterflied_v = __shfl_xor_sync(shfl_mask, max_val, radix);
            float butterflied_d = __shfl_xor_sync(shfl_mask, den, radix);

            float this_max_val = max(max_val, butterflied_v);

            den = isinf(this_max_val) ? 0
                               : (den * exp(max_val - this_max_val) +
                                  butterflied_d * exp(butterflied_v - this_max_val));
            max_val = this_max_val;
        }
    }

    const float inv_sum = 1.0f / den;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t idst = (int64_t)rowx * ncols + col;
        const int64_t ix = (int64_t)rowx * ncols + col;
        const int64_t iy = (int64_t)rowy * ncols + col;

        const float val =
            vals_smem
                ? vals[col]
                : (x[ix] * scale + (mask ? slope * t2f32(mask[iy]) : 0.0f));

        dst[idst] = expf(val - max_val) * inv_sum;
    }
}

template<typename T>
static void soft_max_f32_cuda(const float * x, const T * mask, float * dst, const int ncols_x, const int nrows_x, const int nrows_y, const float scale, const float max_bias, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = (GGML_PAD(ncols_x, WARP_SIZE) + 2 * WARP_SIZE) * sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    const uint32_t n_head      = nrows_x/nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    if (shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        switch (ncols_x) {
            case 32:
                soft_max_f32<true, 32, 32><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 64:
                soft_max_f32<true, 64, 64><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 128:
                soft_max_f32<true, 128, 128><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 256:
                soft_max_f32<true, 256, 256><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 512:
                soft_max_f32<true, 512, 512><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 1024:
                soft_max_f32<true, 1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 2048:
                soft_max_f32<true, 2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 4096:
                soft_max_f32<true, 4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            default:
                soft_max_f32<true, 0, 0><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
        }
    } else {
        const size_t shmem_low = 2 * WARP_SIZE * sizeof(float);
        soft_max_f32<false, 0, 0><<<block_nums, block_dims, shmem_low, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
    }
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const void  * src1_d = src1 ? (const void *)src1->data : nullptr;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    if (use_f16) {
        const half * src1_dd = (const half *)src1_d;

        soft_max_f32_cuda(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    } else {
        const float * src1_dd = (const float *)src1_d;

        soft_max_f32_cuda(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    }
}
