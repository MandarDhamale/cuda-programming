#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // Required for malloc, srand, etc.
#include <math.h>   // Required for sqrt

typedef struct hist_entry{
    unsigned long long d_cnt;
} bucket;

typedef struct atomdesc {
    double x_pos;
    double y_pos;
    double z_pos;
} atom;

__global__ void sdh_gpu_kernel(atom *atom_list_device, bucket *histogram_device, int N, double w, int num_buckets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        for (int j = i + 1; j < N; j++) {
            double x1 = atom_list_device[i].x_pos;
            double x2 = atom_list_device[j].x_pos;
            double y1 = atom_list_device[i].y_pos;
            double y2 = atom_list_device[j].y_pos;
            double z1 = atom_list_device[i].z_pos;
            double z2 = atom_list_device[j].z_pos;
            double dist = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
            int h_pos = (int)(dist / w);
            if (h_pos < num_buckets) {
                atomicAdd(&(histogram_device[h_pos].d_cnt), (unsigned long long)1);
            }
        }
    }
}

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

int main(int argc, char **argv) {
    // 1. Define problem size and parse command-line arguments
    if (argc != 3) {
        printf("Usage: %s <num_points> <bucket_width>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    double w = atof(argv[2]);
    int num_buckets = (int)(23000 * 1.732 / w) + 1;

    // 2. Declare and Allocate Host Pointers
    atom *atom_list_host = (atom*)malloc(sizeof(atom) * N);
    bucket *cpu_histogram = (bucket*)malloc(sizeof(bucket) * num_buckets);
    bucket *gpu_histogram = (bucket*)malloc(sizeof(bucket) * num_buckets);

    // 3. Initialize the host data
    srand(1); // Use a fixed seed for consistent results
    for (int i = 0; i < N; i++) {
        atom_list_host[i].x_pos = ((double)(rand()) / RAND_MAX) * 23000;
        atom_list_host[i].y_pos = ((double)(rand()) / RAND_MAX) * 23000;
        atom_list_host[i].z_pos = ((double)(rand()) / RAND_MAX) * 23000;
    }

    // --- CPU-side Calculation ---
    printf("CPU Histogram:\n");
    for (int i = 0; i < num_buckets; i++) {
        cpu_histogram[i].d_cnt = 0;
    }
    for(int i = 0; i < N; i++) {
        for(int j = i + 1; j < N; j++) {
            double x1 = atom_list_host[i].x_pos;
            double x2 = atom_list_host[j].x_pos;
            double y1 = atom_list_host[i].y_pos;
            double y2 = atom_list_host[j].y_pos;
            double z1 = atom_list_host[i].z_pos;
            double z2 = atom_list_host[j].z_pos;
            double dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2) + (z1 - z2)*(z1-z2));
            int h_pos = (int) (dist / w);
            if (h_pos < num_buckets) {
                cpu_histogram[h_pos].d_cnt++;
            }
        } 
    }
    for (int i = 0; i < num_buckets; i++) {
        if (i % 5 == 0) printf("\n%02d: ", i);
        printf("%15llu ", cpu_histogram[i].d_cnt);
        if (i != num_buckets - 1) printf("| ");
    }
    printf("\n");
    
    // --- GPU-side Calculation ---
    atom *atom_list_device;
    bucket *histogram_device;
    
    // Initialize GPU histogram to zero before copying
    for (int i = 0; i < num_buckets; i++) {
        gpu_histogram[i].d_cnt = 0;
    }

    cudaMalloc((void**)&atom_list_device, sizeof(atom) * N);
    cudaMalloc((void**)&histogram_device, sizeof(bucket) * num_buckets);

    cudaMemcpy(atom_list_device, atom_list_host, sizeof(atom) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(histogram_device, gpu_histogram, sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);

    int blockSize = 256; 
    int gridSize = (N + blockSize - 1) / blockSize;

    sdh_gpu_kernel<<<gridSize, blockSize>>>(atom_list_device, histogram_device, N, w, num_buckets);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_histogram, histogram_device, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);

    // --- Print GPU Results ---
    printf("\nGPU Histogram:\n");
    for (int i = 0; i < num_buckets; i++) {
        if (i % 5 == 0) printf("\n%02d: ", i);
        printf("%15llu ", gpu_histogram[i].d_cnt); 
        if (i != num_buckets - 1) printf("| ");
    }
    printf("\n");

    // --- Compare Results ---
    compare_histograms(cpu_histogram, gpu_histogram, num_buckets);

    // --- Clean Up ---
    printf("\nCleaning up memory...\n");
    free(atom_list_host);
    free(cpu_histogram);
    free(gpu_histogram);
    cudaFree(atom_list_device);
    cudaFree(histogram_device);

    return 0;
}