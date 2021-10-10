#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define GROUP_SIZE 16
__kernel void matrix_transpose(__global const float *as,
                               __global       float *as_t,
                               unsigned int M, unsigned int K) {
    __local float local_as[GROUP_SIZE][GROUP_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < K && j < M) {
        local_as[local_j][(local_i + local_j) & 0xf] = as[j * K + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int new_i = j - local_j + local_i;
    int new_j = i - local_i + local_j;
    if (new_i < M && new_j < K) {
        as_t[new_j * M + new_i] = local_as[local_i][(local_i + local_j) & 0xf];
    }
}