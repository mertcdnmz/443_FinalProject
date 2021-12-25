#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
using namespace std;


void PrintMatrix(float *matrix, int n);


// This kernel function takes the current pivot value and divide all indices of row with this value to make diagonels(pivots) 1 
__global__ void MakePivotsOne(float* matrix, int rowSize, int colSize, int currCol, float pivotValue)
{
	int tID = currCol * colSize + threadIdx.x ;
	if ( tID < rowSize * colSize )
	{
        matrix[tID] = matrix[tID] / pivotValue;
	}
    __syncthreads();
}

// This kernel function makes all top and bottom values of pivot's column zero
__global__ void MakePivotsColumnZero(float* matrix, int rowSize, int colSize, int currCol, int currRow) 
{
    int tID = currRow * colSize +  threadIdx.x;

    float rateWithPivot = matrix[ currRow * colSize + currCol ] ;

    int colNumber = tID % colSize;
     
    if(currRow != currCol)
    {
        matrix[tID] = matrix[tID] - (rateWithPivot * matrix[( currCol * colSize ) + colNumber ]);   
    }
    __syncthreads();
    
}

void PrintMatrix(float *a, int n)
{   
    printf("\n") ;
    for (int i = 0; i < n -1   ; i++) 
    {
        for (int j = 0; j < n; j++){
            printf("%.3f ",  a[ i * n + j ]) ;
        }
        
        printf("\n")  ;
    }
    printf("\n") ;
}


__global__ void PrintMatrixGPU(float *a, int n)
{   
    printf("\n") ;
    for (int i = 0; i < n -1   ; i++) 
    {
        for (int j = 0; j < n; j++){
            printf("%.3f ",  a[ i * n + j ]) ;
        }
        
        printf("\n")  ;
    }
    printf("\n") ;
}


void createMatrix(float *matrix, int rowSize){
    char filename[32];
    snprintf(filename, sizeof(char)*32, "../dataset/matrix_n_%i.txt", rowSize);
    FILE* file = fopen (filename, "r");
    int matrixSize = 0;
    fscanf (file, " %d", &matrixSize);
    for(int i = 0 ; i < rowSize * (rowSize + 1 ); i++)
    {   
        float number = 0 ;
        fscanf(file, "%f",&number);
        matrix[i] = number;
    }
    fclose (file);
}

void saveMatrix(float* matrix, int rowSize){
    char outputFilename[50];
    snprintf(outputFilename, sizeof(char)*32, "../dataset/output_matrix_%i.txt", rowSize);
    
    FILE* file = fopen(outputFilename, "w");
    fprintf(file,"%d", rowSize);
    for(int i = 0 ; i < rowSize * (rowSize+1) ; i++){
        if(i % (rowSize + 1) == 0){
            fprintf(file, "\n");
        }
        fprintf(file, "%1.f" , matrix[i]);
        if(i != ( rowSize * rowSize ) - 1){
            fprintf(file, " ");
        }
    }
    fclose(file);
}

int main(int argc, char const *argv[]) {

{
    //printing device properties
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("***************Device Properties****************\n");
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device name: %s\n", prop.name);
        printf("Max Grid Size X: %d\n", prop.maxGridSize[0]);
        printf("Max Grid Size Y: %d\n", prop.maxGridSize[1]);
        printf("Max Grid Size Z: %d\n", prop.maxGridSize[2]);
        printf("Max Number of Threads X: %d\n", prop.maxThreadsDim[0]);
        printf("Max Number of Threads Y: %d\n", prop.maxThreadsDim[1]);
        printf("Max Number of Threads Z: %d\n", prop.maxThreadsDim[2]);
        printf("Warp size: %d\n\n", prop.warpSize);
        
    }

}   

    int rowSize = stoi(argv[1]);
    int colSize = rowSize + 1 ;
    size_t size = rowSize * colSize * sizeof(float);
  
    float *h_Matrix  = (float*) malloc(size);
    createMatrix(h_Matrix, rowSize);
    
    cudaError_t err = cudaSuccess; // error handling

    // Allocate the device input matrix 
    float *d_Matrix = NULL;
    err = cudaMalloc(&d_Matrix, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy matrix values to GPU memory
    err = cudaMemcpy(d_Matrix, h_Matrix, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; // calculate performance
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventRecord(stop);

    printf("Gauss Jordan Elimination method calculation is started! \n");

    for(int currCol = 0; currCol < rowSize  ; currCol++ )
    {
        float temp = h_Matrix[ currCol * colSize + currCol ];
        MakePivotsOne<<<1, colSize>>>(d_Matrix, rowSize, colSize, currCol, temp);
        cudaDeviceSynchronize();
		// Check for errors
        
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_Matrix);

			return false;
		}
        
        for( int i = 0 ; i < rowSize ; i++){
            MakePivotsColumnZero<<<1, colSize>>>(d_Matrix, rowSize, colSize, currCol, i);
            cudaDeviceSynchronize();
            
        }
        
        cudaDeviceSynchronize();
        err = cudaMemcpy(h_Matrix, d_Matrix,size, cudaMemcpyDeviceToHost); // copy device matrix into host matrix to take correct pivot value 
        
		// Check for errors
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_Matrix);
			return false;
		}

    }

    // Free device global memory
    err = cudaFree(d_Matrix);
    if (err != cudaSuccess)
    {   
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Gauss Jordan Elimination method calculation is finished! \n\n");
    printf("Total milliseconds: %f\n\n", milliseconds);

    saveMatrix(h_Matrix,rowSize);
    
    // Free host memory
    free(h_Matrix);

    printf("******************Program Finished!*****************\n");
return 0;
}