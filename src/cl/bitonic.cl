#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6
#define WORK_GROUP_SIZE 256u

__kernel void bitonic(__global float *as, unsigned int seq_length, unsigned int l) {
    const unsigned int id = get_global_id(0);
    unsigned int idx1 = 2 * id - (id & (l - 1));
    unsigned int idx2 = idx1 + l;
    bool ascending = (idx1 & seq_length) == 0;
    float a1 = as[idx1];
    float a2 = as[idx2];
    if ((a1 <= a2) != ascending) {
        as[idx1] = a2;
        as[idx2] = a1;
    }
}

__kernel void bitonic_start(__global float *as) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local float local_as[2 * WORK_GROUP_SIZE];
    local_as[2 * local_id] = as[2 * id];
    local_as[2 * local_id + 1] = as[2 * id + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int seq_length = 2; seq_length <= 2 * WORK_GROUP_SIZE; seq_length *= 2) {
        for (unsigned int l = seq_length / 2; l >= 1; l /= 2) {
            unsigned int local_idx1 = 2 * local_id - (local_id & (l - 1));
            unsigned int local_idx2 = local_idx1 + l;
            bool ascending = ((2 * id - (id & (l - 1))) & seq_length) == 0;
            float a1 = local_as[local_idx1];
            float a2 = local_as[local_idx2];
            if ((a1 <= a2) != ascending) {
                local_as[local_idx1] = a2;
                local_as[local_idx2] = a1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    as[2 * id] = local_as[2 * local_id];
    as[2 * id + 1] = local_as[2 * local_id + 1];
}

__kernel void bitonic_end(__global float *as, unsigned int seq_length) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local float local_as[2 * WORK_GROUP_SIZE];
    local_as[2 * local_id] = as[2 * id];
    local_as[2 * local_id + 1] = as[2 * id + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int l = WORK_GROUP_SIZE; l >= 1; l /= 2) {
        unsigned int local_idx1 = 2 * local_id - (local_id & (l - 1));
        unsigned int local_idx2 = local_idx1 + l;
        bool ascending = ((2 * id - (id & (l - 1))) & seq_length) == 0;
        float a1 = local_as[local_idx1];
        float a2 = local_as[local_idx2];
        if ((a1 <= a2) != ascending) {
            local_as[local_idx1] = a2;
            local_as[local_idx2] = a1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as[2 * id] = local_as[2 * local_id];
    as[2 * id + 1] = local_as[2 * local_id + 1];
}