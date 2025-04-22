#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
//#include "../gpu-metrics/gpu-metrics.hpp"
#include <iomanip>
#include <iostream>

using namespace std;

using dtype = float;
dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = dtype(1.1);
  }
}

template <int N, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int blockRun) {
  dtype localSum = dtype(0);

  for (int i = 0; i < N / 2; i++) {
    int idx =
        (blockDim.x * blockRun * i + (blockIdx.x % blockRun) * BLOCKSIZE) * 2 +
        threadIdx.x;
    localSum += B[idx] * B[idx + BLOCKSIZE];
  }

  localSum *= (dtype)1.3;
  if (threadIdx.x > 1233 || localSum == (dtype)23.12)
    A[threadIdx.x] += localSum;
}
template <int N, int blockSize>
double callKernel(int blockCount, int blockRun) {
  sumKernel<N, blockSize><<<blockCount, blockSize>>>((dtype*)dA, (dtype*)dB, blockRun);
  GPU_ERROR(cudaPeekAtLastError());
  return 0.0;
}
template <int N> void measure(int blockRun, int random, int run_long) {

  const int blockSize = 1024;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, blockSize>, blockSize, 0));

  int blockCount = 200000;

  // GPU_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  double* c;
  c = (double*)malloc(sizeof(double)*(blockRun * blockSize * N ));
  #pragma omp parallel for
  for (int i = 0; i < (blockRun * blockSize * N ) ; i++) {
    if (random) c[i] = 1.0 + ( (double)(rand()) / (double)(RAND_MAX) );
    else c[i] = 0.0;
  }
 
  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 1; i++) {
    const size_t bufferCount = blockRun * blockSize * N + i * 128;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    GPU_ERROR( cudaMemset(dA, 0, bufferCount*sizeof(dtype)) );  // initialize to zeros
    GPU_ERROR( cudaMemcpy(dA, c, bufferCount*sizeof(dtype), cudaMemcpyHostToDevice) );
    //initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(dtype)));
    GPU_ERROR( cudaMemset(dB, 0, bufferCount*sizeof(dtype)) );  // initialize to zeros
    GPU_ERROR( cudaMemcpy(dB, c, bufferCount*sizeof(dtype), cudaMemcpyHostToDevice) );
    //initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(cudaDeviceSynchronize());

    int iter=1000;
    if ((N * blockSize * sizeof(dtype)) * blockRun / 1024 > 40000) iter=900;
    if (!run_long) iter=1;

    double t1 = dtime();
    for (int ii = 0; ii < iter; ii++) {
      callKernel<N, blockSize>(blockCount, blockRun);
    }
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1)/iter);

    /* measureDRAMBytesStart();
     callKernel<N, blockSize>(blockCount, blockRun);
     auto metrics = measureDRAMBytesStop();
     dram_read.add(metrics[0]);
     dram_write.add(metrics[1]);

     measureL2BytesStart();
     callKernel<N, blockSize>(blockCount, blockRun);
     metrics = measureL2BytesStop();
     L2_read.add(metrics[0]);
     L2_write.add(metrics[1]);*/
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
  }

  double blockDV = N * blockSize * sizeof(dtype);

  double bw = blockDV * blockCount / time.minValue() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << fixed << setprecision(0) << setw(10) << blockDV * blockRun / 1024
       << " kB"                                                           //
       << setprecision(0) << setw(10) << time.minValue() * 1000.0 << "ms" //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
       << setw(10) << bw << " GB/s   "                                    //
       << setprecision(0) << setw(6)
       << dram_read.median() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(6)
       << dram_write.median() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(6)
       << L2_read.median() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(6)
       << L2_write.median() / time.minValue() / 1.0e9 << " GB/s " << endl; //
}

size_t constexpr expSeries(size_t N) {
  size_t val = 20;
  for (size_t i = 0; i < N; i++) {
    val = val * 1.04 + 1;
  }
  return val;
}

int main(int argc, char **argv) {
  //initMeasureMetric();
  int random=0;
  int run_long=0;
  if (argc!=3 || std::atoi(argv[1])>1 || std::atoi(argv[1])<0 || std::atoi(argv[2])>1 || std::atoi(argv[2])<0) {
    std::cout << "Test requires two args. "<< std::endl;
    std::cout << "First arg (0/1) ==> zero/random data "<< std::endl;
    std::cout << "Second arg (0/1) ==> orig/running longer for power measurement "<< std::endl;
    exit (0);
  }
  for (int i = 1; i < argc; ++i) {
    if (i==1 && std::atoi(argv[i])==1) random=1;
    if (i==2 && std::atoi(argv[i])==1) run_long=1;
  }
  std::cout << "Test with ";
  if (random) std::cout <<"random data. ";
  else std::cout <<"zero data. ";
  std::cout <<std::endl;
  if (run_long) std::cout <<"Power profile mode."<<std::endl;

  cout << setw(13) << "data set"   //
       << setw(12) << "exec time"  //
       << setw(11) << "spread"     //
       << setw(15) << "Eff. bw\n"; //

  for (int i = 3; i < 1000; i += 4*max(1.0, i * 0.1)) {
#ifdef __NVCC__
    measure<128>(i, random, run_long);
#else
    measure<128>(i, random, run_long);
#endif
  }
}
