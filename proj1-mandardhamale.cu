/* ==================================================================
	Programmer: Mandar Dhamale (mandardhamale@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the GAIVI machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/* Compute and display the difference between the CPU and GPU histograms 
*/
void compare_histograms(bucket *cpu_hist, bucket *gpu_hist, int num_buckets) {
    printf("\nDifference between CPU and GPU histograms:\n");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
}

__global__ void sdh_kernel(atom *d_atom_list, bucket *d_histogram, int n, double w) {
    // Calculate the global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure the thread ID is within the bounds of the atom list
    if (i < n) {
        double dist;
        int h_pos;

        // Each thread calculates the distance from its assigned point 'i' 
        // to all subsequent points 'j' to avoid double counting.
        for (int j = i + 1; j < n; j++) {
            // Calculate Euclidean distance between point i and point j
            double dx = d_atom_list[i].x_pos - d_atom_list[j].x_pos;
            double dy = d_atom_list[i].y_pos - d_atom_list[j].y_pos;
            double dz = d_atom_list[i].z_pos - d_atom_list[j].z_pos;
            dist = sqrt(dx*dx + dy*dy + dz*dz);

            // Determine the histogram bucket index
            h_pos = (int)(dist / w);
            
            // Use atomicAdd to safely increment the bucket counter from multiple threads
            atomicAdd(&(d_histogram[h_pos].d_cnt), (unsigned long long)1);
        }
    }
}

int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram();

	// --- GPU Implementation Starts Here ---
    printf("\n--- Starting GPU Computation ---\n");

    // Declare device (GPU) pointers
    atom *d_atom_list;
    bucket *d_histogram;

    // Allocate memory for a new histogram to store GPU results on the host
    bucket *gpu_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);

	// Calculate the size of our data arrays in bytes
    size_t atom_list_size = sizeof(atom) * PDH_acnt;
    size_t histogram_size = sizeof(bucket) * num_buckets;

    // Allocate memory on the GPU device
    cudaMalloc((void**)&d_atom_list, atom_list_size);
    cudaMalloc((void**)&d_histogram, histogram_size);

	// Copy the atom list from host memory (atom_list) to device memory (d_atom_list)
    cudaMemcpy(d_atom_list, atom_list, atom_list_size, cudaMemcpyHostToDevice);
    
    // Initialize the histogram on the device to all zeros
    cudaMemset(d_histogram, 0, histogram_size);

	// Set the number of threads per block
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed in the grid
    int blocksPerGrid = (PDH_acnt + threadsPerBlock - 1) / threadsPerBlock;

	// Launch the CUDA kernel
    printf("Launching kernel with %d blocks and %d threads per block...\n", blocksPerGrid, threadsPerBlock);
    sdh_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);

    // Wait for the GPU to finish all its work
    cudaDeviceSynchronize();
    printf("Kernel execution finished.\n");

	// Copy the resulting histogram from device memory back to host memory
    cudaMemcpy(gpu_histogram, d_histogram, histogram_size, cudaMemcpyDeviceToHost);

	// Temporarily point the global histogram pointer to the GPU results to reuse the output function
    bucket * temp = histogram; // save original CPU histogram pointer
    histogram = gpu_histogram; // point to GPU results

    printf("\n--- GPU Histogram Results ---\n");
    output_histogram();

    histogram = temp; // restore pointer to CPU results

    // Compare the two histograms
    compare_histograms(histogram, gpu_histogram, num_buckets);

	// Free the allocated GPU memory
    cudaFree(d_atom_list);
    cudaFree(d_histogram);
    free(gpu_histogram); // Free the host memory for the gpu histogram
    printf("\nGPU memory freed. Program finished.\n");
	
	return 0;
}

