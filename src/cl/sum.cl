#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 256
__kernel void sum(__global const unsigned int* as,
                  __global       unsigned int* result,
                  unsigned int n)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_as[local_id] = as[global_id];
    } else {
        local_as[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m = WORK_GROUP_SIZE / 2; m > 0; m /= 2) {
        if (local_id < m) {
            local_as[local_id] += local_as[local_id + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(result, local_as[0]);
    }
}
