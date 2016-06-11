#ifndef STENCIL2D_CUH
#define STENCIL2D_CUH

#include <iostream>
#include <stdio.h>
#define BWIDTH 32
#define BHEIGHT 32
#define X 128

//<template int D>
//__global__ void sten_msk(float* output, float* input, int width, int height, float* mask)
//{
//	int size = BHEIGHT+4;
//	__shared__ float data[BHEIGHT+4][BWIDTH+4];
//
//	int column = blockIdx.x*blockDim.x + threadIdx.x;
//	int line = blockIdx.y*blockDim.y + threadIdx.y;
//	int scolumn = (column%BWIDTH);
//	int sline = (line%BHEIGHT);
//
//	// Dodanie pol od
//	if (column-2 < 0 || line-2 < 0)
//		data[sline][scolumn] = 0;
//	else
//		data[sline][scolumn] = input[(line-2)*width+column-2];
//
//	if (scolumn + blockDim.x < size) {
//		scolumn += blockDim.x;
//		column += blockDim.x;
//		
//		if (line-2 < 0 || column-2 >= width)
//			data[sline][scolumn] = 0;
//		else
//			data[sline][scolumn] = input[(line-2)*width+column-2];
//		if (sline + blockDim.y < size) {
//			sline += blockDim.y;
//			line += blockDim.y;
//			if (line-2 >= height || column-2 >= width  )
//				data[sline][scolumn] = 0;
//			else
//				data[sline][scolumn] = input[(line-2)*width+column-2];
//			sline -= blockDim.y;
//			line -= blockDim.y;
//		}
//		scolumn -= blockDim.x;
//		column -= blockDim.x;
//	}
//
//	if (sline + blockDim.y < size) {
//		sline += blockDim.y;
//		line += blockDim.y;
//		if (column-2 < 0 || line-2 >= height )
//			data[sline][scolumn] = 0;
//		else
//			data[sline][scolumn] = input[(line-2)*width+column-2];
//		sline -= blockDim.y;
//		line -= blockDim.y;
//	}
//
//	__syncthreads();
//
//	float result;	
//	for (int i = 0; i <  5; i++)
//		for (int k = 0; k < 5; k++)
//			result += data[sline+i][scolumn+k];
//	output[line*width+column] = result/25;
//}

template<int D>
__global__ void sten_avg(float* output, float* input, int width, int height)
{
	int size = BHEIGHT+4;
	__shared__ float data[BHEIGHT+2*D][BWIDTH+2*D];

	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int line = blockIdx.y*blockDim.y + threadIdx.y;
	int scolumn = (column%BWIDTH);
	int sline = (line%BHEIGHT);

	// Dodanie pol od
	if (column-2 < 0 || line-2 < 0)
		data[sline][scolumn] = 0;
	else
		data[sline][scolumn] = input[(line-2)*width+column-2];

	if (scolumn + blockDim.x < size) {
		scolumn += blockDim.x;
		column += blockDim.x;
		
		if (line-2 < 0 || column-2 >= width)
			data[sline][scolumn] = 0;
		else
			data[sline][scolumn] = input[(line-2)*width+column-2];
		if (sline + blockDim.y < size) {
			sline += blockDim.y;
			line += blockDim.y;
			if (line-2 >= height || column-2 >= width  )
				data[sline][scolumn] = 0;
			else
				data[sline][scolumn] = input[(line-2)*width+column-2];
			sline -= blockDim.y;
			line -= blockDim.y;
		}
		scolumn -= blockDim.x;
		column -= blockDim.x;
	}

	if (sline + blockDim.y < size) {
		sline += blockDim.y;
		line += blockDim.y;
		if (column-2 < 0 || line-2 >= height )
			data[sline][scolumn] = 0;
		else
			data[sline][scolumn] = input[(line-2)*width+column-2];
		sline -= blockDim.y;
		line -= blockDim.y;
	}

	__syncthreads();

	float result;	
	for (int i = 0; i <  5; i++)
		for (int k = 0; k < 5; k++)
			result += data[sline+i][scolumn+k];
	output[line*width+column] = result/25;
}

template<int D, bool AVG>
cudaError stencil2D(float* output, float* input, int width, int height, float* mask)
{
	float *devOut, *devInput;
	cudaMalloc(&devOut, sizeof(float)*width*height);
	cudaMalloc(&devInput, sizeof(float)*width*height);
	cudaMemcpy(devInput, input, sizeof(float)*width*height, cudaMemcpyHostToDevice);
	dim3 blockDim(BWIDTH, BHEIGHT);
	dim3 gridDim(width/BWIDTH, height/BHEIGHT);
	sten_avg<2><<<gridDim, blockDim>>>(devOut, devInput, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(output, devOut, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOut);
	return cudaGetLastError();
}

#endif
