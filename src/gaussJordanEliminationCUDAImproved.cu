#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
using namespace std;


void PrintMatrix(float *matrix, int n);

__global__ void ScaleRowKernel(float* matrix, int rowSize, int colSize, int currCol, float temp)
{
	int tID = currCol * colSize + threadIdx.x ;
    __shared__ float tempPivot ;
    tempPivot = temp ;//temp is pivot value
	if ( tID < rowSize * colSize )
	{
        matrix[tID] = matrix[tID] / tempPivot;
	}
}

__global__ void SubtractRowKernel(float* matrix, int rowSize, int colSize, int currCol, int i) 
{
    int tID = i * colSize +  threadIdx.x;

    __shared__ float temp;
    temp = matrix[ i * colSize + currCol ] ;

    int colNumber = tID % colSize;
     
    if(i != currCol){
        matrix[tID] = matrix[tID] - (temp * matrix[( currCol * colSize ) + colNumber ]);
        __syncthreads();

    }
    
}

void GaussionElimination(float *matrix, int numberOfRows, int numberOfColumns)
{
    float temp = 0 ;
    for(int currCol = 0; currCol < numberOfRows  ; currCol++ )
    {
        temp = matrix[ currCol * numberOfColumns + currCol ];
        for(int j = 0 ; j < numberOfColumns ; j++)
        {
            matrix[ currCol * numberOfColumns + j ] = (matrix[ currCol * numberOfColumns + j ]) / temp;
        }
           
        for(int i = 0 ; i < numberOfRows; i++)
        {
            temp = matrix[ i * numberOfColumns + currCol ];
            for(int j = 0 ; j < numberOfColumns ; j++)
            {
                if(i != currCol)
                {
                    matrix[ i * numberOfColumns + j ] = matrix[ i * numberOfColumns + j ] -  (temp * matrix[ currCol * numberOfColumns + j ]);
                }
            }
        }
    }
}

void PrintMatrix(float *a, int n)
{   
    cout << endl;
    for (int i = 0; i < n -1   ; i++) 
    {
        for (int j = 0; j < n; j++){
            cout << a[ i * n + j ] << " ";
        }
        
        cout << endl ;
    }
    cout << endl ;
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


void createMatrix(float *matrix, int length){
    char filename[32];
    snprintf(filename, sizeof(char)*32, "../dataset/matrix_n_%i.txt", length);
    FILE* file = fopen (filename, "r");
    int matrixSize = 0;
    fscanf (file, " %d", &matrixSize);
    for(int i = 0 ; i < length * (length + 1 ); i++)
    {   
        float number = 0 ;
        fscanf(file, "%f",&number);
        matrix[i] = number;
    }
    fclose (file);
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
        printf("Warp size: %d\n", prop.warpSize);
        
    }

}   

    int rowSize = stoi(argv[1]);
    int colSize = rowSize + 1 ;
    size_t size = rowSize * colSize * sizeof(float);
  
    float *h_Matrix  = (float*) malloc(size);
    createMatrix(h_Matrix, rowSize);
    //PrintMatrix(h_Matrix, colSize);
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

    for(int currCol = 0; currCol < rowSize  ; currCol++ )
    {
        float temp = h_Matrix[ currCol * colSize + currCol ];
        ScaleRowKernel<<<1, colSize>>>(d_Matrix, rowSize, colSize, currCol, temp);
        cudaDeviceSynchronize();
		// Check for errors
        //PrintMatrixGPU<<<1, 1>>>(d_Matrix,colSize);
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_Matrix);

			return false;
		}
        
        for( int i = 0 ; i < rowSize ; i++){
            SubtractRowKernel<<<1, colSize>>>(d_Matrix, rowSize, colSize, currCol, i);
            cudaDeviceSynchronize();
            
        }
        //PrintMatrixGPU<<<1, 1>>>(d_Matrix,colSize);
        cudaDeviceSynchronize();
        //err = cudaMemcpy(h_Matrix, d_Matrix,size, cudaMemcpyDeviceToHost);
        printf("Copy Pivot \n");
        if((currCol + 1)* colSize + (currCol+1) < colSize * rowSize){
            err = cudaMemcpy(&h_Matrix[(currCol + 1)* colSize + (currCol+1)], &d_Matrix[(currCol + 1)* colSize + (currCol+1)], sizeof(float), cudaMemcpyDeviceToHost);
        }
        
		// Check for errors
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_Matrix);
			return false;
		}

    }
    cudaEvent_t start, stop; // calculate performance
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventRecord(stop);
    
    err = cudaGetLastError();
    if( err != cudaSuccess){
        fprintf(stderr, "Fail to last error (%s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //Copy back output (C) values into CPU memory
    err = cudaMemcpy(h_Matrix, d_Matrix, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy device matrix C to host matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);


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

    printf("Total milliseconds: %f\n", milliseconds);

    //Write down the reduced row echelon matrix
    PrintMatrix(h_Matrix,colSize);
    
    // Free host memory
    free(h_Matrix);

    

    printf("******************Reduce Done!*****************\n");
return 0;
}




    //kernel code
    //int blockX = TILE_WIDTH ; // each 2d block should contain threads as much as TILE_WIDHT in each dimension(X,Y) 
    //int blockY = TILE_WIDTH ; 
    //int gridX = matrixSize / TILE_WIDTH; // we need matrixSize*matrixSize threads. 
    //int gridY = matrixSize / TILE_WIDTH; // So, we should use (matrixSize/TILE_WIDTH, matrixSize/TILE_WIDTH) blocks in our grid.
    //dim3 gridShape(gridX,gridY) ;
    //dim3 blockShape(blockX,blockY); //multi dimensional 
    
    // We should use matrixSize*matrixSize threads.
    // With this way, each thread can calculate one output element. 
    //printf("CUDA kernel launch with (%d,%d) blocks of (%d,%d) threads\n", gridX,gridY, blockX,blockY); 
    