#ifndef STENCIL2D_CUH
#define STENCIL2D_CUH

#include <iostream>
#include <stdio.h>
#define BWIDTH 32
#define BHEIGHT 32

template<int D>
__global__ void sten_mas(float* output, float* input, int width, int height, float* mas)
{
	int size = BHEIGHT+2*D;

	__shared__ float data[BHEIGHT+2*D][BWIDTH+2*D];
	__shared__ float mask[2*D+1][2*D+1];

	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int line = blockIdx.y*blockDim.y + threadIdx.y;
	int scolumn = (column%BWIDTH);
	int sline = (line%BHEIGHT);
	
	if (sline < 2*D+1 && scolumn < 2*D+1)
		mask[sline][scolumn] = mas[sline*(2*D+1)+scolumn];

	if (column-D < 0 || line-D < 0)
		data[sline][scolumn] = 0;
	else
		data[sline][scolumn] = input[(line-D)*width+column-D];

	if (scolumn + blockDim.x < size) {
		scolumn += blockDim.x;
		column += blockDim.x;
		
		if (line-D < 0 || column-D >= width)
			data[sline][scolumn] = 0;
		else
			data[sline][scolumn] = input[(line-D)*width+column-D];
		if (sline + blockDim.y < size) {
			sline += blockDim.y;
			line += blockDim.y;
			if (line-D >= height || column-D >= width  )
				data[sline][scolumn] = 0;
			else
				data[sline][scolumn] = input[(line-D)*width+column-D];
			sline -= blockDim.y;
			line -= blockDim.y;
		}
		scolumn -= blockDim.x;
		column -= blockDim.x;
	}

	if (sline + blockDim.y < size) {
		sline += blockDim.y;
		line += blockDim.y;
		if (column-D < 0 || line-D >= height )
			data[sline][scolumn] = 0;
		else
			data[sline][scolumn] = input[(line-D)*width+column-D];
		sline -= blockDim.y;
		line -= blockDim.y;
	}

	__syncthreads();

	float result = 0;	
	for (int i = 0; i <  2*D+1; i++)
		for (int k = 0; k < 2*D+1; k++)
			result += data[sline+i][scolumn+k] * mask[i][k];
	
	output[line*width+column] = result;

}

template<int D>
__global__ void sten_avg(float* output, float* input, int width, int height)
{
	int size = BHEIGHT+2*D;

	__shared__ float data[BHEIGHT+2*D][BWIDTH+2*D];

	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int line = blockIdx.y*blockDim.y + threadIdx.y;
	int scolumn = (column%BWIDTH);
	int sline = (line%BHEIGHT);

	if (column-D < 0 || line-D < 0)
		data[sline][scolumn] = 0;
	else
		data[sline][scolumn] = input[(line-D)*width+column-D];

	if (scolumn + blockDim.x < size) {
		scolumn += blockDim.x;
		column += blockDim.x;
		
		if (line-D < 0 || column-D >= width)
			data[sline][scolumn] = 0;
		else
			data[sline][scolumn] = input[(line-D)*width+column-D];
		if (sline + blockDim.y < size) {
			sline += blockDim.y;
			line += blockDim.y;
			if (line-D >= height || column-D >= width  )
				data[sline][scolumn] = 0;
			else
				data[sline][scolumn] = input[(line-D)*width+column-D];
			sline -= blockDim.y;
			line -= blockDim.y;
		}
		scolumn -= blockDim.x;
		column -= blockDim.x;
	}

	if (sline + blockDim.y < size) {
		sline += blockDim.y;
		line += blockDim.y;
		if (column-D < 0 || line-D >= height )
			data[sline][scolumn] = 0;
		else
			data[sline][scolumn] = input[(line-D)*width+column-D];
		sline -= blockDim.y;
		line -= blockDim.y;
	}

	__syncthreads();

	float result = 0;
	for (int i = 0; i <  2*D+1; i++)
		for (int k = 0; k < 2*D+1; k++)
			result += data[sline+i][scolumn+k];
	output[line*width+column] = result/((2*D+1)*(2*D+1));
}

template<int D, bool AVG>
cudaError stencil2D(float* output, float* input, int width, int height, float* mask)
{
	float *devOut, *devInput, *devMask;
	cudaMalloc(&devOut, sizeof(float)*width*height);
	cudaMalloc(&devInput, sizeof(float)*width*height);
	cudaMalloc(&devMask, sizeof(float)*(2*D+1)*(2*D+1));
	cudaMemcpy(devInput, input, sizeof(float)*width*height, cudaMemcpyHostToDevice);
	cudaMemcpy(devMask, mask, sizeof(float)*(2*D+1)*(2*D+1), cudaMemcpyHostToDevice);
	dim3 blockDim(BWIDTH, BHEIGHT);
	dim3 gridDim(width/BWIDTH, height/BHEIGHT);
	if (AVG)
		sten_avg<D><<<gridDim, blockDim>>>(devOut, devInput, width, height);
	else
		sten_mas<D><<<gridDim, blockDim>>>(devOut, devInput, width, height, devMask);

	cudaDeviceSynchronize();
	cudaMemcpy(output, devOut, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOut);
	return cudaGetLastError();
}

#endif
