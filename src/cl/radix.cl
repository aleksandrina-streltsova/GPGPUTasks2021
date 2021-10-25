#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6
#define N_VALUES 16
#define WORK_GROUP_SIZE 256

__kernel void count(__global const unsigned int *as,
                    __global       unsigned int *count_table,
                    unsigned int offset,
                    unsigned int n_groups) {
    const unsigned int group_id = get_group_id(0);
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_count_table[WORK_GROUP_SIZE][N_VALUES];

    for (int i = 0; i < N_VALUES; ++i) {
        local_count_table[local_id][i] = 0;
    }

    unsigned int a = as[id];
    unsigned int value = (a >> offset) & (N_VALUES - 1);
    local_count_table[local_id][value]++;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id <= N_VALUES) {
        for (int i = 1; i < WORK_GROUP_SIZE; ++i) {
            local_count_table[0][local_id] += local_count_table[i][local_id];
        }
        count_table[local_id * n_groups + group_id] = local_count_table[0][local_id];
    }
}

__kernel void local_scan(__global const int *as,
                         __global       int *prefix_sum,
                         __global       int *sum,
                         unsigned int n) {
    const unsigned int group_id = get_group_id(0);
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_sum[2 * WORK_GROUP_SIZE];
    __local unsigned int local_prefix_sum[WORK_GROUP_SIZE];

    if (id < n) {
        local_sum[local_id + 256] = as[id];
    } else {
        local_sum[local_id + 256] = 0;
    }
    local_sum[local_id] = 0;
    local_prefix_sum[local_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 128) {
        local_sum[local_id + 128] = local_sum[2 * local_id + 256] + local_sum[2 * local_id + 257];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 64) {
        local_sum[local_id + 64] = local_sum[2 * local_id + 128] + local_sum[2 * local_id + 129];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 32) {
        local_sum[local_id + 32] = local_sum[2 * local_id + 64] + local_sum[2 * local_id + 65];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 16) {
        local_sum[local_id + 16] = local_sum[2 * local_id + 32] + local_sum[2 * local_id + 33];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 8) {
        local_sum[local_id + 8] = local_sum[2 * local_id + 16] + local_sum[2 * local_id + 17];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 4) {
        local_sum[local_id + 4] = local_sum[2 * local_id + 8] + local_sum[2 * local_id + 9];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 2) {
        local_sum[local_id + 2] = local_sum[2 * local_id + 4] + local_sum[2 * local_id + 5];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 1) {
        local_sum[local_id + 1] = local_sum[2 * local_id + 2] + local_sum[2 * local_id + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int idx = local_id + 1;
    if (idx & 1) {
        local_prefix_sum[local_id] += local_sum[255 + idx];
    }
    if (idx & 2) {
        local_prefix_sum[local_id] += local_sum[127 + (idx >> 1)];
    }
    if (idx & 4) {
        local_prefix_sum[local_id] += local_sum[63 + (idx >> 2)];
    }
    if (idx & 8) {
        local_prefix_sum[local_id] += local_sum[31 + (idx >> 3)];
    }
    if (idx & 16) {
        local_prefix_sum[local_id] += local_sum[15 + (idx >> 4)];
    }
    if (idx & 32) {
        local_prefix_sum[local_id] += local_sum[7 + (idx >> 5)];
    }
    if (idx & 64) {
        local_prefix_sum[local_id] += local_sum[3 + (idx >> 6)];
    }
    if (idx & 128) {
        local_prefix_sum[local_id] += local_sum[1 + (idx >> 7)];
    }
    if (idx & 256) {
        local_prefix_sum[local_id] += local_sum[idx >> 8];
    }

    // write to global memory
    if (id < n) {
        prefix_sum[id] = local_prefix_sum[local_id];
    }
    if (local_id == 255) {
        sum[group_id] = local_prefix_sum[local_id];
    }
}

__kernel void scan_last(__global const int *as,
                        __global       int *prefix_sum,
                        unsigned int n) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    __local unsigned int local_prefix_sum[WORK_GROUP_SIZE];

    for (unsigned int offset = 0; offset < n; offset += WORK_GROUP_SIZE) {
        if (id < n) {
            local_as[local_id] = as[id];
        } else {
            local_as[local_id] = 0;
        }
        local_prefix_sum[local_id] = 0;

        if (local_id == 0) {
            local_prefix_sum[0] = local_as[0];
            for (int i = 1; i < WORK_GROUP_SIZE; ++i) {
                local_prefix_sum[i] = local_prefix_sum[i - 1] + local_as[i];
            }
        }
        if (id < n) {
            prefix_sum[id] = local_prefix_sum[local_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

}

__kernel void scan(__global const unsigned int *level_0,
                   __global const unsigned int *level_1,
                   __global const unsigned int *level_2,
                   __global       unsigned int *prefix_sum,
                   unsigned int n) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local int local_prefix_sum[WORK_GROUP_SIZE];

    local_prefix_sum[local_id] = 0;

    unsigned int idx = id + 1;
    if (idx & 255) {
        local_prefix_sum[local_id] += level_0[idx - 1];
    }
    if ((idx >> 8) & 255) {
        local_prefix_sum[local_id] += level_1[(idx >> 8) - 1];
    }
    if ((idx >> 16) & 255) {
        local_prefix_sum[local_id] += level_2[(idx >> 16) - 1];
    }
    prefix_sum[id] = local_prefix_sum[local_id];
}

__kernel void reorder(__global const unsigned int *as,
                      __global const unsigned int *prefix_sum,
                      __global       unsigned int *as_sorted,
                      unsigned int offset, unsigned int n_groups) {
    const unsigned int group_id = get_group_id(0);
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_count_table[WORK_GROUP_SIZE][N_VALUES];

    for (int i = 0; i < N_VALUES; ++i) {
        local_count_table[local_id][i] = 0;
    }

    unsigned int a = as[id];
    unsigned int value = (a >> offset) & (N_VALUES - 1);
    local_count_table[local_id][value]++;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < N_VALUES) {
        for (int i = 1; i < WORK_GROUP_SIZE; ++i) {
            local_count_table[i][local_id] += local_count_table[i - 1][local_id];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int idx = prefix_sum[value * n_groups + group_id] - local_count_table[WORK_GROUP_SIZE - 1][value] +
                       local_count_table[local_id][value] - 1;
    as_sorted[idx] = a;
}
