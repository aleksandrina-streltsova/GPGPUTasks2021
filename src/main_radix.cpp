#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#define N_VALUES 16
#define WORK_GROUP_SIZE 256

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

unsigned int round_up(unsigned int num, unsigned int den) {
    return (num + den - 1) / den;
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    gpu::gpu_mem_32u as_gpu, as_sorted_gpu, count_table_gpu, prefix_sum_gpu;
    unsigned int block_size = WORK_GROUP_SIZE;
    unsigned int n_gpu = (n + block_size - 1) / block_size * block_size;
    std::vector<unsigned int> as_max(n_gpu, std::numeric_limits<unsigned int>::max());
    as_gpu.resizeN(n_gpu);
    as_gpu.writeN(as_max.data(), n_gpu);
    as_sorted_gpu.resizeN(n_gpu);
    as_sorted_gpu.writeN(as_max.data(), n_gpu);
    unsigned int count_table_size = n_gpu / block_size * N_VALUES;
    count_table_gpu.resizeN(count_table_size);
    prefix_sum_gpu.resizeN(count_table_size);

    gpu::gpu_mem_32u level_0_gpu, level_1_gpu, level_2_gpu, group_0_gpu, group_1_gpu;
    unsigned int level_0_size = count_table_size;
    unsigned int level_1_size = round_up(level_0_size, WORK_GROUP_SIZE);
    unsigned int level_2_size = round_up(level_1_size, WORK_GROUP_SIZE);

    level_0_gpu.resizeN(level_0_size);
    level_1_gpu.resizeN(level_1_size);
    level_2_gpu.resizeN(level_2_size);
    group_0_gpu.resizeN(level_1_size);
    group_1_gpu.resizeN(level_2_size);

    {
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();
        ocl::Kernel local_scan(radix_kernel, radix_kernel_length, "local_scan");
        local_scan.compile();
        ocl::Kernel scan_last(radix_kernel, radix_kernel_length, "scan_last");
        scan_last.compile();
        ocl::Kernel scan(radix_kernel, radix_kernel_length, "scan");
        scan.compile();
        ocl::Kernel reorder(radix_kernel, radix_kernel_length, "reorder");
        reorder.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = WORK_GROUP_SIZE;
            for (unsigned int i = 0; i < 32; i += 4) {
                count.exec(gpu::WorkSize(workGroupSize, n_gpu), as_gpu, count_table_gpu, i,n_gpu / block_size);

                local_scan.exec(gpu::WorkSize(workGroupSize, level_0_size), count_table_gpu, level_0_gpu, group_0_gpu, level_0_size);
                local_scan.exec(gpu::WorkSize(workGroupSize, level_1_size), group_0_gpu, level_1_gpu, group_1_gpu, level_1_size);
                scan_last.exec(gpu::WorkSize(workGroupSize, workGroupSize), group_1_gpu, level_2_gpu, level_2_size);
                scan.exec(gpu::WorkSize(workGroupSize, level_0_size), level_0_gpu, level_1_gpu, level_2_gpu, prefix_sum_gpu, count_table_size);

                reorder.exec(gpu::WorkSize(workGroupSize, n_gpu), as_gpu, prefix_sum_gpu, as_sorted_gpu, i, n_gpu / block_size);

                std::swap(as_gpu, as_sorted_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
