#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
//#include "../gpu-metrics/gpu-metrics.hpp"

#include <iomanip>
#include <iostream>

using namespace std;

#ifdef __NVCC__
using dtype = float;
#else
using dtype = float4;
#endif

dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = (dtype)1.1;
  }
}

template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int zero) {
  dtype localSum = (dtype)0;

  B += threadIdx.x;

#pragma unroll N / BLOCKSIZE> 32   ? 1 : 32 / (N / BLOCKSIZE)
  for (int iter = 0; iter < iters; iter++) {
    B += zero;
    //auto B2 = B + N;
#pragma unroll N / BLOCKSIZE >= 64 ? 32 : N / BLOCKSIZE
    for (int i = 0; i < N; i += BLOCKSIZE) {
      localSum += B[i];
    }
    localSum *= (dtype)1.3;
  }
  if (localSum == (dtype)1233)
    A[threadIdx.x] += localSum;
}

template <int N, int iters, int blockSize> double callKernel(int blockCount) {
  sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, 0);
  return 0.0;
}

template <int N> void measure(int random, int run_long) {
  const size_t iters = (size_t)1000000000 / N + 2;

  const int blockSize = 256;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, iters, blockSize>, blockSize, 0));

  int blockCount = smCount * 1; // maxActiveBlocks;

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

    dtype* c;
    c = (dtype*)malloc(sizeof(dtype)*(N ));
    #pragma omp parallel for
    for (int i = 0; i < N  ; i++) {
      if (random) c[i] = 1.0 + ( (dtype)(rand()) / (dtype)(RAND_MAX) );
      else c[i]=0.0;
    }

  GPU_ERROR(cudaDeviceSynchronize());

  for (int i = 0; i < 1; i++) {
    const size_t bufferCount = N; // + i * 1282;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    GPU_ERROR( cudaMemset(dA, 0, bufferCount*sizeof(dtype)) );  // initialize to zeros
    GPU_ERROR( cudaMemcpy(dA, c, bufferCount*sizeof(dtype), cudaMemcpyHostToDevice) );
    //initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(dtype)));
    GPU_ERROR( cudaMemset(dB, 0, bufferCount*sizeof(dtype)) );  // initialize to zeros
    GPU_ERROR( cudaMemcpy(dB, c, bufferCount*sizeof(dtype), cudaMemcpyHostToDevice) );
    //initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(cudaDeviceSynchronize());

    int iter=900;
    if (!run_long) iter =1;
 
    double t1 = dtime();
    for (int ii = 0; ii < iter; ii++) {
      callKernel<N, iters, blockSize>(blockCount);
    }
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1)/iter);
    /*
    measureDRAMBytesStart();
    callKernel<N, iters, blockSize>(blockCount);
    auto metrics = measureDRAMBytesStop();
    dram_read.add(metrics[0]);
    dram_write.add(metrics[1]);

    measureL2BytesStart();
    callKernel<N, iters, blockSize>(blockCount);
    metrics = measureL2BytesStop();
    L2_read.add(metrics[0]);
    L2_write.add(metrics[1]);
    */
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
  }
  double blockDV = N * sizeof(dtype);

  double bw = blockDV * blockCount * iters / time.minValue() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
       << setw(10) << bw << " GB/s"                                       //
       << setprecision(0) << setw(10)
       << dram_read.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << dram_write.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << L2_read.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << L2_write.value() / time.minValue() / 1.0e9 << " GB/s " << endl; //
}

size_t constexpr expSeries(size_t N) {
  size_t val = 32 * 512;
  for (size_t i = 0; i < N; i++) {
    val *= 1.17;
  }
  return (val / 512) * 512;
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

  unsigned int clock = getGPUClock();
  cout << setw(13) << "data set"   //
       << setw(12) << "exec time"  //
       << setw(11) << "spread"     //
       << setw(15) << "Eff. bw"    //
       << setw(16) << "DRAM read"  //
       << setw(16) << "DRAM write" //
       << setw(16) << "L2 read"    //
       << setw(16) << "L2 store\n";

  //initMeasureMetric();

  const int ct = 1;
  measure<ct*256>(random, run_long);
  measure<ct*512>(random, run_long);
  measure<ct*3 * 256>(random, run_long);
  measure<ct*5 * 512>(random, run_long);
  measure<ct*9 * 512>(random, run_long);
  measure<ct*13 * 512>(random, run_long);
  measure<ct*17 * 512>(random, run_long);
  measure<ct*21 * 512>(random, run_long);
  measure<ct*25 * 512>(random, run_long);
  measure<ct*29 * 512>(random, run_long);

  measure<ct*expSeries(1)>(random, run_long);
  measure<ct*expSeries(5)>(random, run_long);
  measure<ct*expSeries(9)>(random, run_long);
  measure<ct*expSeries(13)>(random, run_long);
  measure<ct*expSeries(18)>(random, run_long);
  measure<ct*expSeries(22)>(random, run_long);
  measure<ct*expSeries(26)>(random, run_long);
}
