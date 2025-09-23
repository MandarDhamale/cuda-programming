#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// Data structure for a 3D point
typedef struct {
    float x, y, z;
} point;

// Data structure for a histogram bucket
typedef struct {
    unsigned long long d_cnt;
} bucket;

// NOTE: We will pass num_buckets as a kernel argument instead of using a global
// int num_buckets; 

// *****************************************************************************
// KERNEL: Computes the Spatial Distance Histogram on the GPU
// *****************************************************************************
__global__ void sdh_kernel(point *d_points, bucket *d_histogram, int N, float w, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        point p_i = d_points[idx];

        for (int j = idx + 1; j < N; j++) {
            point p_j = d_points[j];
            float dist = sqrtf(powf(p_i.x - p_j.x, 2) + powf(p_i.y - p_j.y, 2) + powf(p_i.z - p_j.z, 2));
            int h_pos = floorf(dist / w);
            
            // This now works because num_buckets is passed as an argument
            if (h_pos < num_buckets) {
                atomicAdd(&(d_histogram[h_pos].d_cnt), (unsigned long long)1);
            }
        }
    }
}

// *****************************************************************************
// UTILITY: Compares CPU and GPU histograms and prints the differences
// *****************************************************************************
void compare_histograms(bucket *cpu_hist, bucket *gpu_hist, int num_buckets) {
    printf("\nDifference between CPU and GPU histograms:\n");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = (long long)cpu_hist[i].d_cnt - (long long)gpu_hist[i].d_cnt;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n\n");
}


// *****************************************************************************
// MAIN FUNCTION
// *****************************************************************************
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <num_points> <bucket_width>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    float w = atof(argv[2]);
    // num_buckets is now a local variable in main
    int num_buckets = (int)(1000.0 / w) + 1;

    // ========= Host Memory Allocation =========
    point *h_points = (point *)malloc(sizeof(point) * N);
    bucket *cpu_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    bucket *gpu_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_points[i].x = (float)(rand() % 1000);
        h_points[i].y = (float)(rand() % 1000);
        h_points[i].z = (float)(rand() % 1000);
    }

    for (int i = 0; i < num_buckets; i++) {
        cpu_histogram[i].d_cnt = 0;
        gpu_histogram[i].d_cnt = 0;
    }

    // ========= 1. CPU HISTOGRAM COMPUTATION =========
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float dist = sqrt(pow(h_points[i].x - h_points[j].x, 2) +
                              pow(h_points[i].y - h_points[j].y, 2) +
                              pow(h_points[i].z - h_points[j].z, 2));
            int h_pos = floor(dist / w);
            if (h_pos < num_buckets) {
                cpu_histogram[h_pos].d_cnt++;
            }
        }
    }

    printf("CPU Histogram:\n");
    for (int i = 0; i < num_buckets; i++) {
        if (i % 5 == 0) printf("\n%02d: ", i);
        printf("%15llu ", cpu_histogram[i].d_cnt);
        if (i != num_buckets - 1) printf("| ");
    }
    printf("\n");

    // ========= 2. GPU HISTOGRAM COMPUTATION =========
    point *d_points;
    bucket *d_histogram;

    cudaMalloc((void **)&d_points, sizeof(point) * N);
    cudaMalloc((void **)&d_histogram, sizeof(bucket) * num_buckets);

    cudaMemcpy(d_points, h_points, sizeof(point) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, gpu_histogram, sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel with the added 'num_buckets' argument
    sdh_kernel<<<gridSize, blockSize>>>(d_points, d_histogram, N, w, num_buckets);

    cudaDeviceSynchronize();

    cudaMemcpy(gpu_histogram, d_histogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);

    printf("\nGPU Histogram:\n");
    for (int i = 0; i < num_buckets; i++) {
        if (i % 5 == 0) printf("\n%02d: ", i);
        printf("%15llu ", gpu_histogram[i].d_cnt);
        if (i != num_buckets - 1) printf("| ");
    }
    printf("\n");


    // ========= 3. COMPARISON =========
    compare_histograms(cpu_histogram, gpu_histogram, num_buckets);


    // ========= Cleanup =========
    cudaFree(d_points);
    cudaFree(d_histogram);
    free(h_points);
    free(cpu_histogram);
    free(gpu_histogram);

    return 0;
}