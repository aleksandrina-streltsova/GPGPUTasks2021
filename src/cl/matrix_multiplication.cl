#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define GROUP_SIZE 16
__kernel void matrix_multiplication(__global const float *as,
                                    __global const float *bs,
                                    __global       float *cs,
                                    unsigned int M, unsigned int K, unsigned int N) {
    __local float local_as[GROUP_SIZE][GROUP_SIZE];
    __local float local_bs[GROUP_SIZE][GROUP_SIZE];

    float sum = 0.f;

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    for (int k = 0; k * GROUP_SIZE < K; ++k) {
        if (j < M && k * GROUP_SIZE + local_i < K) {
            local_as[local_j][(local_i + local_j) & 0xf] = as[j * K + k * GROUP_SIZE + local_i];
        } else {
            local_as[local_j][(local_i + local_j) & 0xf] = 0.f;
        }

        if (k * GROUP_SIZE + local_j < K && i < N) {
            local_bs[local_j][local_i] = bs[(k * GROUP_SIZE + local_j) * N + i];
        } else {
            local_bs[local_j][local_i] = 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < GROUP_SIZE; ++l) {
            sum += local_as[local_j][(local_j + local_i + l) & 0xf] * local_bs[(l + local_i) & 0xf][local_i];
        }
    }

    if (j < M && i < N) {
        cs[j * N + i] = sum;
    }
}