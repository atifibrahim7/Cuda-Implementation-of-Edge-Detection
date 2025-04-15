/*******************************************************************************
* --------------------------------------------
*(c) 2001 University of South Florida, Tampa
* Use, or copying without permission prohibited.
* ... [Copyright and permission notice remains unchanged] ...
*******************************************************************************/

typedef long long fixed;
#define fixeddot 16

#define BOOSTBLURFACTOR 90.0
#define VERBOSE 0
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <math.h>
#include <string.h>

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA call caught error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)
int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval);
void canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth_gpu(unsigned char *image, int rows, int cols, float sigma, short int **smoothdem, double *time_taken);
void gaussian_smooth_cpu(unsigned char *image, int rows, int cols, float sigma, short int **smoothdem, double *time_taken);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y_gpu(short int *smoothdem, int rows, int cols, short int **delta_x, short int **delta_y, double *time_taken);
void derrivative_x_y_cpu(short int *smoothdem, int rows, int cols, short int **delta_x, short int **delta_y, double *time_taken);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge);
void radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result);
__global__ void gaussianSmoothXKernel(unsigned char *d_image, float *d_tempim, int rows, int cols, float *d_kernel, int windowsize);
__global__ void gaussianSmoothYKernel(float *d_tempim, short *d_smoothedim, int rows, int cols, float *d_kernel, int windowsize);
__global__ void derivativeXKernel(short *d_smoothedim, short *d_delta_x, int rows, int cols);
__global__ void derivativeYKernel(short *d_smoothedim, short *device_d_y, int rows, int cols);
int main(int argc, char *argv[])
{
   char *infilename = NULL;
   char *dirfilename = NULL;
   char outfilename[128];
   char composedfname[128];
   unsigned char *image;
   unsigned char *edge;
   int rows, cols;
   float sigma, tlow, thigh;

   if (argc < 5) {
      fprintf(stderr, "\n<USAGE> %s image sigma tlow thigh [writedirim]\n", argv[0]);
      fprintf(stderr, "\n      image:      An image to process. Must be in PGM format.\n");
      fprintf(stderr, "      sigma:      Standard deviation of the gaussian blur kernel.\n");
      fprintf(stderr, "      tlow:       Fraction (0.0-1.0) of the high edge strength threshold.\n");
      fprintf(stderr, "      thigh:      Fraction (0.0-1.0) of the distribution of non-zero edge strengths.\n");
      fprintf(stderr, "      writedirim: Optional argument to output a floating point direction image.\n\n");
      exit(1);
   }

   infilename = argv[1];
   sigma = atof(argv[2]);
   tlow = atof(argv[3]);
   thigh = atof(argv[4]);

   if (argc == 6) dirfilename = infilename;

   if (VERBOSE) printf("Reading the image %s.\n", infilename);
   if (read_pgm_image(infilename, &image, &rows, &cols) == 0) {
      fprintf(stderr, "Error reading the input image, %s.\n", infilename);
      exit(1);
   }

   if (VERBOSE) printf("Starting Canny edge detection with CPU and GPU comparison.\n");
   if (dirfilename != NULL) {
      sprintf(composedfname, "%s_s_%3.3f_l_%3.3f_h_%3.3f.fim", infilename, sigma, tlow, thigh);
      dirfilename = composedfname;
   }

   canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);

   sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename, sigma, tlow, thigh);
   if (VERBOSE) printf("Writing the edge image to the file %s.\n", outfilename);
   if (write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0) {
      fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
      exit(1);
   }

   free(image);
   free(edge);
   return 0;
}

void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
{
   FILE *fpdir = NULL;
   unsigned char *nms;
   short int *smoothedim_cpu, *smoothedim_gpu, *delta_x_cpu, *delta_x_gpu, *delta_y_cpu, *delta_y_gpu, *magnitude;
   float *dir_radians = NULL;
   double gaussian_time_cpu = 0.0, gaussian_time_gpu = 0.0, derivative_time_cpu = 0.0, derivative_time_gpu = 0.0;

   if (VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
   gaussian_smooth_cpu(image, rows, cols, sigma, &smoothedim_cpu, &gaussian_time_cpu);
   gaussian_smooth_gpu(image, rows, cols, sigma, &smoothedim_gpu, &gaussian_time_gpu);

   int mismatch_count = 0;
   for (int i = 0; i < rows * cols; i++) {
      if (smoothedim_cpu[i] != smoothedim_gpu[i]) {
         mismatch_count++;
         if (VERBOSE && mismatch_count <= 10) {
            printf("Gaussian mismatch at index %d: CPU=%d, GPU=%d\n", i, smoothedim_cpu[i], smoothedim_gpu[i]);
         }
      }
   }
   printf("Gaussian Smoothing Comparison: %d mismatches out of %d pixels (%.2f%%)\n",
          mismatch_count, rows * cols, (float)mismatch_count / (rows * cols) * 100);

   if (VERBOSE) printf("Computing the X and Y first derivatives.\n");
   derrivative_x_y_cpu(smoothedim_cpu, rows, cols, &delta_x_cpu, &delta_y_cpu, &derivative_time_cpu);
   derrivative_x_y_gpu(smoothedim_cpu, rows, cols, &delta_x_gpu, &delta_y_gpu, &derivative_time_gpu);

   mismatch_count = 0;
   for (int i = 0; i < rows * cols; i++) {
      if (delta_x_cpu[i] != delta_x_gpu[i]) {
         mismatch_count++;
         if (VERBOSE && mismatch_count <= 10) {
            printf("Delta_x mismatch at index %d: CPU=%d, GPU=%d\n", i, delta_x_cpu[i], delta_x_gpu[i]);
         }
      }
   }
   printf("Delta_x Comparison: %d mismatches out of %d pixels (%.2f%%)\n",
          mismatch_count, rows * cols, (float)mismatch_count / (rows * cols) * 100);

   mismatch_count = 0;
   for (int i = 0; i < rows * cols; i++) {
      if (delta_y_cpu[i] != delta_y_gpu[i]) {
         mismatch_count++;
         if (VERBOSE && mismatch_count <= 10) {
            printf("Delta_y mismatch at index %d: CPU=%d, GPU=%d\n", i, delta_y_cpu[i], delta_y_gpu[i]);
         }
      }
   }
   printf("Delta_y Comparison: %d mismatches out of %d pixels (%.2f%%)\n",
          mismatch_count, rows * cols, (float)mismatch_count / (rows * cols) * 100);
   printf("\nTOTAL TIMING Results :\n");
   printf("Gaussian Soothing GPU: %.6f seconds\n", gaussian_time_gpu);
   printf("Gaussian Soothing CPU: %.6f seconds\n", gaussian_time_cpu);
   printf("Derivatives (GPU): %.5f seconds\n", derivative_time_gpu);
   printf("Derivatives (CPU): %.5f seconds\n", derivative_time_cpu);
   printf("Time Total (GPU): %.5f seconds\n", gaussian_time_gpu + derivative_time_gpu);
   printf("Time Total (CPU): %.5f seconds\n", gaussian_time_cpu + derivative_time_cpu);
   printf("Speedup (Gaussian): %.3fx\n", gaussian_time_cpu / gaussian_time_gpu);
   printf("Speedup (Derivatives): %.3fx\n", derivative_time_cpu / derivative_time_gpu);
   printf("Speedup (Total): %.3fx\n", (gaussian_time_cpu + derivative_time_cpu) / (gaussian_time_gpu + derivative_time_gpu));

   // Proceed with CPU results for remaining steps
   if (fname != NULL) {
      radian_direction(delta_x_cpu, delta_y_cpu, rows, cols, &dir_radians, -1, -1);
      if ((fpdir = fopen(fname, "wb")) == NULL) {
         fprintf(stderr, "Error opening the file %s for writing.\n", fname);
         exit(1);
      }
      fwrite(dir_radians, sizeof(float), rows * cols, fpdir);
      fclose(fpdir);
      free(dir_radians);
   }
   if (VERBOSE) printf("Computing the magnitude of the gradient.\n");
   magnitude_x_y(delta_x_cpu, delta_y_cpu, rows, cols, &magnitude);
   if (VERBOSE) printf("Doing the non-maximal suppression.\n");
   if ((nms = (unsigned char *)malloc(rows * cols * sizeof(unsigned char))) == NULL) {
      fprintf(stderr, "Error allocating the nms image.\n");
      exit(1);
   }
   non_max_supp(magnitude, delta_x_cpu, delta_y_cpu, rows, cols, nms);
   if (VERBOSE) printf("Doing hysteresis thresholding.\n");
   if ((*edge = (unsigned char *)malloc(rows * cols * sizeof(unsigned char))) == NULL) {
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
   }
   apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);
   free(smoothedim_gpu);
   free(smoothedim_cpu);
   free(delta_x_gpu);
   free(delta_x_cpu);
   free(delta_y_gpu);
   free(delta_y_cpu);
   free(magnitude);
   free(nms);
}

__global__ void gaussianSmoothXKernel(unsigned char *d_image, float *d_tempim, int rows, int cols, float *d_kernel, int windowsize) {
   int center = windowsize / 2;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if (x < cols && y < rows) {
       float dot = 0.0f;
       float sum = 0.0f;
       for (int cc = -center; cc <= center; cc++) {
           int idx = x + cc;
           if (idx >= 0 && idx < cols) {
               dot += (float)d_image[y * cols + idx] * d_kernel[center + cc];
               sum += d_kernel[center + cc];
           }
       }
       d_tempim[y * cols + x] = (sum > 0.0f) ? (dot / sum) : 0.0f;
   }
}
__global__ void gaussianSmoothYKernel(float *d_tempim, short *d_smoothedim, int rows, int cols, float *d_kernel, int windowsize) {
   int center = windowsize / 2;
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < cols && y < rows) {
       float dot = 0.0f;
       float sum = 0.0f;
       for (int rr = -center; rr <= center; rr++) {
           int idx = y + rr;
           if (idx >= 0 && idx < rows) {
               dot += d_tempim[idx * cols + x] * d_kernel[center + rr];
               sum += d_kernel[center + rr];
           }
       }
       d_smoothedim[y * cols + x] = (sum > 0.0f) ? (short)(dot * BOOSTBLURFACTOR / sum + 0.5f) : 0;
   }
}

void gaussian_smooth_gpu(unsigned char *image, int rows, int cols, float sigma, short int **smoothdem, double *time_taken) {
   float *kernel;
   unsigned char *d_image;
   float *d_tempim;
   short *d_smoothedim;
   float *d_kernel;
   int windowsize;

   if (VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, &kernel, &windowsize);

   *smoothdem = (short int *)malloc(rows * cols * sizeof(short int));
   if (*smoothdem == NULL) {
      fprintf(stderr, "Error allocating host memory for smoothdem.\n");
      exit(1);
   }

   CUDA_ERROR_CHECK(cudaMalloc((void **)&d_smoothedim, rows * cols * sizeof(short)));
   CUDA_ERROR_CHECK(cudaMalloc((void **)&d_tempim, rows * cols * sizeof(float)));
   CUDA_ERROR_CHECK(cudaMalloc((void **)&d_kernel, windowsize * sizeof(float)));
   CUDA_ERROR_CHECK(cudaMalloc((void **)&d_image, rows * cols * sizeof(unsigned char)));
   cudaEvent_t startH2D, stopH2D, startK, stopK, startD2H, stopD2H;
   CUDA_ERROR_CHECK(cudaEventCreate(&stopH2D));
   CUDA_ERROR_CHECK(cudaEventCreate(&startK));
   CUDA_ERROR_CHECK(cudaEventCreate(&startH2D));
   CUDA_ERROR_CHECK(cudaEventCreate(&startD2H));
   CUDA_ERROR_CHECK(cudaEventCreate(&stopD2H));
   CUDA_ERROR_CHECK(cudaEventCreate(&stopK));

   float elapsedH2D = 0.0f, elapsedK = 0.0f, elapsedD2H = 0.0f;
   CUDA_ERROR_CHECK(cudaEventRecord(startH2D, 0));
   CUDA_ERROR_CHECK(cudaMemcpy(d_image, image, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
   CUDA_ERROR_CHECK(cudaMemcpy(d_kernel, kernel, windowsize * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_ERROR_CHECK(cudaEventRecord(stopH2D, 0));
   CUDA_ERROR_CHECK(cudaEventSynchronize(stopH2D));
   CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedH2D, startH2D, stopH2D));

   dim3 block(16, 16);
   dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
   CUDA_ERROR_CHECK(cudaEventRecord(startK, 0));
   if (VERBOSE) printf("   Blurr the image h in the X-direction (CUDA).\n");
   gaussianSmoothXKernel<<<grid, block>>>(d_image, d_tempim, rows, cols, d_kernel, windowsize);
   CUDA_ERROR_CHECK(cudaGetLastError());
   CUDA_ERROR_CHECK(cudaDeviceSynchronize());
   if (VERBOSE) printf("   Blurring the image in the Y-direction (CUDA).\n");
   gaussianSmoothYKernel<<<grid, block>>>(d_tempim, d_smoothedim, rows, cols, d_kernel, windowsize);
   CUDA_ERROR_CHECK(cudaGetLastError());
   CUDA_ERROR_CHECK(cudaDeviceSynchronize());
   CUDA_ERROR_CHECK(cudaEventRecord(stopK, 0));
   CUDA_ERROR_CHECK(cudaEventSynchronize(stopK));
   CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedK, startK, stopK));
   CUDA_ERROR_CHECK(cudaEventRecord(startD2H, 0));
   CUDA_ERROR_CHECK(cudaMemcpy(*smoothdem, d_smoothedim, rows * cols * sizeof(short), cudaMemcpyDeviceToHost));
   CUDA_ERROR_CHECK(cudaEventRecord(stopD2H, 0));
   CUDA_ERROR_CHECK(cudaEventSynchronize(stopD2H));
   CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedD2H, startD2H, stopD2H));
   *time_taken = (double)(elapsedH2D + elapsedK + elapsedD2H) / 1000.0;
   CUDA_ERROR_CHECK(cudaEventDestroy(startH2D));
   CUDA_ERROR_CHECK(cudaEventDestroy(stopH2D));
   CUDA_ERROR_CHECK(cudaEventDestroy(startK));
   CUDA_ERROR_CHECK(cudaEventDestroy(stopK));
   CUDA_ERROR_CHECK(cudaEventDestroy(startD2H));
   CUDA_ERROR_CHECK(cudaEventDestroy(stopD2H));
   CUDA_ERROR_CHECK(cudaFree(d_image));
   CUDA_ERROR_CHECK(cudaFree(d_tempim));
   CUDA_ERROR_CHECK(cudaFree(d_smoothedim));
   CUDA_ERROR_CHECK(cudaFree(d_kernel));
   free(kernel);
}

void gaussian_smooth_cpu(unsigned char *image, int rows, int cols, float sigma, short int **smoothdem, double *time_taken)
{
   int r, c, rr, cc, windowsize, center;
   float *temporaryim, *kernel, dot, sum;
   auto start = std::chrono::high_resolution_clock::now();
   if (VERBOSE) printf("    the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, &kernel, &windowsize);
   center = windowsize / 2;
   if ((temporaryim = (float *)malloc(rows * cols * sizeof(float))) == NULL) {
      fprintf(stderr, "Error allocating the buffer.\n");
      exit(1);
   }
   if (((*smoothdem) = (short int *)malloc(rows * cols * sizeof(short int))) == NULL) {
      fprintf(stderr, "Cant allocating the smoothed image.\n");
      exit(1);
   }
   if (VERBOSE) printf("   Blurring the image in the X-dir (HOST).\n");
   for (r = 0; r < rows; r++) {
      for (c = 0; c < cols; c++) {
         dot = 0.0;
         sum = 0.0;
         for (cc = (-center); cc <= center; cc++) {
            if (((c + cc) >= 0) && ((c + cc) < cols)) {
               dot += (float)image[r * cols + (c + cc)] * kernel[center + cc];
               sum += kernel[center + cc];
            }
         }
         temporaryim[r * cols + c] = dot / sum;
      }
   }
   if (VERBOSE) printf("Blurring the image in the Y-direction (CPU).\n");
   for (c = 0; c < cols; c++) {
      for (r = 0; r < rows; r++) {
         sum = 0.0;
         dot = 0.0;
         for (rr = (-center); rr <= center; rr++) {
            if (((r + rr) >= 0) && ((r + rr) < rows)) {
               dot += temporaryim[(r + rr) * cols + c] * kernel[center + rr];
               sum += kernel[center + rr];
            }
         }
         (*smoothdem)[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
      }
   }
   free(kernel);
   free(temporaryim);
   auto end = std::chrono::high_resolution_clock::now();
   *time_taken = std::chrono::duration<double>(end - start).count();
}

__global__ void derivativeXKernel(short *d_smoothedim, short *d_delta_x, int rows, int cols) {
   int ttxx = threadIdx.x;
   int ty = threadIdx.y;
   int bx = blockIdx.x;
   int by = blockIdx.y;
   int x = bx * blockDim.x + ttxx;
   int y = by * blockDim.y + ty;
   extern __shared__ short dim_s_smooth[];
   int loadWidth = blockDim.x + 2;
   int s_x = ttxx + 1;
   int s_y = ty;
   if (x < cols && y < rows) {
       dim_s_smooth[s_y * loadWidth + s_x] = d_smoothedim[y * cols + x];
   }
   if (ttxx == 0 && x > 0 && y < rows) {
       dim_s_smooth[s_y * loadWidth] = d_smoothedim[y * cols + (x - 1)];
   }
   if (ttxx == blockDim.x - 1 && x < cols - 1 && y < rows) {
       dim_s_smooth[s_y * loadWidth + loadWidth - 1] = d_smoothedim[y * cols + (x + 1)];
   }
   __syncthreads();

   if (x < cols && y < rows) {
       short left = (x == 0) ? dim_s_smooth[s_y * loadWidth + s_x] : dim_s_smooth[s_y * loadWidth + s_x - 1];
       short right = (x == cols - 1) ? dim_s_smooth[s_y * loadWidth + s_x] : dim_s_smooth[s_y * loadWidth + s_x + 1];
       d_delta_x[y * cols + x] = right - left;
   }
}

__global__ void derivativeYKernel(short *d_smoothedim, short *device_d_y, int rows, int cols) {
   int bx = blockIdx.x;
   int ttxx = threadIdx.x;
   int ty = threadIdx.y;
   int by = blockIdx.y;
   int x = bx * blockDim.x + ttxx;
   int y = by * blockDim.y + ty;

   extern __shared__ short dim_s_smooth[];
   int loadHeight = blockDim.y + 2;
   int s_x = ttxx;
   int s_y = ty + 1;

   if (x < cols && y < rows) {
       dim_s_smooth[s_y * blockDim.x + s_x] = d_smoothedim[y * cols + x];
   }
   if (ty == 0 && y > 0 && x < cols) {
       dim_s_smooth[s_x] = d_smoothedim[(y - 1) * cols + x];
   }
   if (ty == blockDim.y - 1 && y < rows - 1 && x < cols) {
       dim_s_smooth[(loadHeight - 1) * blockDim.x + s_x] = d_smoothedim[(y + 1) * cols + x];
   }
   __syncthreads();

   if (x < cols && y < rows) {
       short top = (y == 0) ? dim_s_smooth[s_y * blockDim.x + s_x] : dim_s_smooth[(s_y - 1) * blockDim.x + s_x];
       short bottom = (y == rows - 1) ? dim_s_smooth[s_y * blockDim.x + s_x] : dim_s_smooth[(s_y + 1) * blockDim.x + s_x];
       device_d_y[y * cols + x] = bottom - top;
   }
}
void derrivative_x_y_gpu(short int *smoothdem, int rows, int cols, 
   short int **delta_x, short int **delta_y, double *time_taken) {
float elapsedX = 0.0f, elapsedY = 0.0f, elapsedMemH2D = 0.0f, elapsedMemD2H = 0.0f;
cudaEvent_t startMemH2D, stopMemH2D, startX, stopX, startY, stopY, startMemD2H, stopMemD2H;
CUDA_ERROR_CHECK(cudaEventCreate(&startMemH2D));
CUDA_ERROR_CHECK(cudaEventCreate(&stopMemH2D));
CUDA_ERROR_CHECK(cudaEventCreate(&startX));
CUDA_ERROR_CHECK(cudaEventCreate(&startY));
CUDA_ERROR_CHECK(cudaEventCreate(&stopY));
CUDA_ERROR_CHECK(cudaEventCreate(&stopX));
CUDA_ERROR_CHECK(cudaEventCreate(&startMemD2H));
CUDA_ERROR_CHECK(cudaEventCreate(&stopMemD2H));
short *d_smoothedim, *d_delta_x, *device_d_y;
CUDA_ERROR_CHECK(cudaMalloc(&d_smoothedim, rows * cols * sizeof(short)));
CUDA_ERROR_CHECK(cudaMalloc(&d_delta_x, rows * cols * sizeof(short)));
CUDA_ERROR_CHECK(cudaMalloc(&device_d_y, rows * cols * sizeof(short)));
*delta_x = (short *)malloc(rows * cols * sizeof(short));
*delta_y = (short *)malloc(rows * cols * sizeof(short));
if (!*delta_x || !*delta_y) {
fprintf(stderr, "Error allocating Drivtiv images.\n");
exit(1);
}
dim3 block(16, 16);
dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
size_t smemX = block.y * (block.x + 2) * sizeof(short);
size_t smemY = (block.y + 2) * block.x * sizeof(short);
CUDA_ERROR_CHECK(cudaEventRecord(startMemH2D, 0));
CUDA_ERROR_CHECK(cudaMemcpy(d_smoothedim, smoothdem, rows * cols * sizeof(short), cudaMemcpyHostToDevice));
CUDA_ERROR_CHECK(cudaEventRecord(stopMemH2D, 0));
CUDA_ERROR_CHECK(cudaEventSynchronize(stopMemH2D));
CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedMemH2D, startMemH2D, stopMemH2D));
CUDA_ERROR_CHECK(cudaEventRecord(startX, 0));
derivativeXKernel<<<grid, block, smemX>>>(d_smoothedim, d_delta_x, rows, cols);
CUDA_ERROR_CHECK(cudaEventRecord(stopX, 0));
CUDA_ERROR_CHECK(cudaEventSynchronize(stopX));
CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedX, startX, stopX));
CUDA_ERROR_CHECK(cudaEventRecord(startY, 0));
derivativeYKernel<<<grid, block, smemY>>>(d_smoothedim, device_d_y, rows, cols);
CUDA_ERROR_CHECK(cudaEventRecord(stopY, 0));
CUDA_ERROR_CHECK(cudaEventSynchronize(stopY));
CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedY, startY, stopY));
CUDA_ERROR_CHECK(cudaEventRecord(startMemD2H, 0));
CUDA_ERROR_CHECK(cudaMemcpy(*delta_x, d_delta_x, rows * cols * sizeof(short), cudaMemcpyDeviceToHost));
CUDA_ERROR_CHECK(cudaMemcpy(*delta_y, device_d_y, rows * cols * sizeof(short), cudaMemcpyDeviceToHost));
CUDA_ERROR_CHECK(cudaEventRecord(stopMemD2H, 0));
CUDA_ERROR_CHECK(cudaEventSynchronize(stopMemD2H));
CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedMemD2H, startMemD2H, stopMemD2H));
*time_taken = (double)(elapsedMemH2D + elapsedX + elapsedY + elapsedMemD2H) / 1000.0;
CUDA_ERROR_CHECK(cudaFree(d_smoothedim));
CUDA_ERROR_CHECK(cudaFree(d_delta_x));
CUDA_ERROR_CHECK(cudaFree(device_d_y));
CUDA_ERROR_CHECK(cudaEventDestroy(stopMemH2D));
CUDA_ERROR_CHECK(cudaEventDestroy(startMemH2D));
CUDA_ERROR_CHECK(cudaEventDestroy(stopX));
CUDA_ERROR_CHECK(cudaEventDestroy(startY));
CUDA_ERROR_CHECK(cudaEventDestroy(stopY));
CUDA_ERROR_CHECK(cudaEventDestroy(startX));
CUDA_ERROR_CHECK(cudaEventDestroy(startMemD2H));
CUDA_ERROR_CHECK(cudaEventDestroy(stopMemD2H));
}


void derrivative_x_y_cpu(short int *smoothdem, int rows, int cols, short int **delta_x, short int **delta_y, double *time_taken)
{
   auto start = std::chrono::high_resolution_clock::now();

   int r, c, pos;

   if (((*delta_x) = (short *)malloc(rows * cols * sizeof(short))) == NULL) {
      fprintf(stderr, "Error allocating the delta_x image.\n");
      exit(1);
   }
   if (((*delta_y) = (short *)malloc(rows * cols * sizeof(short))) == NULL) {
      fprintf(stderr, "Error allocating the delta_y image.\n");
      exit(1);
   }

   for (r = 0; r < rows; r++) {
      pos = r * cols;
      (*delta_x)[pos] = smoothdem[pos + 1] - smoothdem[pos];
      pos++;
      for (c = 1; c < (cols - 1); c++, pos++) {
         (*delta_x)[pos] = smoothdem[pos + 1] - smoothdem[pos - 1];
      }
      (*delta_x)[pos] = smoothdem[pos] - smoothdem[pos - 1];
   }

   for (c = 0; c < cols; c++) {
      pos = c;
      (*delta_y)[pos] = smoothdem[pos + cols] - smoothdem[pos];
      pos += cols;
      for (r = 1; r < (rows - 1); r++, pos += cols) {
         (*delta_y)[pos] = smoothdem[pos + cols] - smoothdem[pos - cols];
      }
      (*delta_y)[pos] = smoothdem[pos] - smoothdem[pos - cols];
   }

   auto end = std::chrono::high_resolution_clock::now();
   *time_taken = std::chrono::duration<double>(end - start).count();
}

/*******************************************************************************
* PROCEDURE: magnitude_x_y
*******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude)
{
   int r, c, pos, sq1, sq2;

   if ((*magnitude = (short *)malloc(rows * cols * sizeof(short))) == NULL) {
      fprintf(stderr, "Error allocating the magnitude image.\n");
      exit(1);
   }

   for (r = 0, pos = 0; r < rows; r++) {
      for (c = 0; c < cols; c++, pos++) {
         sq1 = (int)delta_x[pos] * (int)delta_x[pos];
         sq2 = (int)delta_y[pos] * (int)delta_y[pos];
         (*magnitude)[pos] = (short)(0.50 + sqrt((float)sq1 + (float)sq2));
      }
   }
}

/*******************************************************************************
* PROCEDURE: radian_direction
*******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag)
{
   int r, c, pos;
   float *dirim = NULL;
   double dx, dy;

   if ((dirim = (float *)malloc(rows * cols * sizeof(float))) == NULL) {
      fprintf(stderr, "Error allocating the gradient direction image.\n");
      exit(1);
   }
   *dir_radians = dirim;

   for (r = 0, pos = 0; r < rows; r++) {
      for (c = 0; c < cols; c++, pos++) {
         dx = (double)delta_x[pos];
         dy = (double)delta_y[pos];

         if (xdirtag == 1) dx = -dx;
         if (ydirtag == -1) dy = -dy;

         dirim[pos] = (float)angle_radians(dx, dy);
      }
   }
}

/*******************************************************************************
* FUNCTION: angle_radians
*******************************************************************************/
double angle_radians(double x, double y)
{
   double xu, yu, ang;

   xu = fabs(x);
   yu = fabs(y);

   if ((xu == 0) && (yu == 0)) return (0);

   ang = atan(yu / xu);

   if (x >= 0) {
      if (y >= 0) return (ang);
      else return (2 * M_PI - ang);
   } else {
      if (y >= 0) return (M_PI - ang);
      else return (M_PI + ang);
   }
}

/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum = 0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if (VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
   if ((*kernel = (float *)malloc((*windowsize) * sizeof(float))) == NULL) {
      fprintf(stderr, "Error allocating the gaussian kernel array.\n");
      exit(1);
   }

   for (i = 0; i < (*windowsize); i++) {
      x = (float)(i - center);
      fx = pow(2.71828, -0.50 * x * x / (sigma * sigma)) / (sigma * sqrt(6.2831853));
      (*kernel)[i] = fx;
      sum += fx;
   }

   for (i = 0; i < (*windowsize); i++) (*kernel)[i] /= sum;

   if (VERBOSE) {
      printf("The filter coefficients are:\n");
      for (i = 0; i < (*windowsize); i++)
         printf("kernel[%d] = %f\n", i, (*kernel)[i]);
   }
}