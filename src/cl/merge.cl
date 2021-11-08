#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define MAX 1000.f
__kernel void merge(__global const float *as,
                    __global       float *buffer,
                    unsigned int n, unsigned int k) {
    const unsigned int id = get_global_id(0);
    const unsigned int diag = id & (2 * k - 1);

    unsigned int l = (diag >= k) ? diag + 1 - k : 0;
    unsigned int r = (diag >= k - 1) ? k : diag + 1;
    unsigned int m;
    float a, b, result;

    while (l < r) {
        m = (l + r) / 2;
        a = as[id - diag + m];
        b = as[id + k - m];
        if (a <= b) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    a = (l == 0) ? -MAX : as[id - diag + l - 1];
    b = (l == diag + 1) ? -MAX : as[id + k - l];

    buffer[id] = (a <= b) ? b : a;
}
