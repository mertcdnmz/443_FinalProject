#include <time.h>
#include <cuda_runtime.h>
#include "util.h"
using namespace std;


// This kernel function takes the current pivot value and divide all indices of row with this value to make diagonels(pivots) 1 
__global__ void MakePivotsOne(double* matrix, int rowSize, int colSize, int currCol, double pivotValue)
{     
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y ;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y ; 
    int threadsPerBlock  = blockDim.x * blockDim.y ;
    int index = blockNumInGrid * threadsPerBlock + threadNumInBlock ;
    
	int tID = currCol * colSize + index ;

    __shared__ double sharedPivot ;
    sharedPivot = pivotValue ; //make pivot value shared
	if ( tID < rowSize * colSize && index < colSize )
	{
        matrix[tID] = matrix[tID] / sharedPivot;
    }
    __syncthreads();
}

// This kernel function makes all top and bottom values of pivot's column zero
__global__ void MakePivotsColumnZero(double* matrix, int rowSize, int colSize, int currCol) 
{    
    int index = threadIdx.x ;
    int currRow = blockIdx.x;

    int tID = currRow * colSize +  index;
  
    __shared__ double rateWithPivot; 
    rateWithPivot = matrix[ currRow * colSize + currCol ] ;

    int colNumber = tID % colSize;
     
    if(currRow != currCol && index < colSize ){
        matrix[tID] = matrix[tID] - (rateWithPivot * matrix[( currCol * colSize ) + colNumber ]);
        __syncthreads();
    }
    
}

__global__ void PrintMatrixGPU(double *a, int n)
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
    size_t size = rowSize * colSize * sizeof(double);
  
    double *h_Matrix  = (double*) malloc(size);
    createMatrix(h_Matrix, rowSize, true);
    
    cudaError_t err = cudaSuccess; // error handling

    // Allocate the device input matrix 
    double *d_Matrix = NULL;
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
    
    
    printf("Gauss Jordan Elimination method calculation is started! \n");
    cudaEventRecord(start);
  
    clock_t start_time = clock(); 
    for(int currCol = 0; currCol < rowSize  ; currCol++ )
    {   
        
        dim3 gridShape(1,1) ;
        dim3 blockShape(colSize,1 ) ; //multi dimensional 
        double currentPivotValue = h_Matrix[ currCol * colSize + currCol ];
        MakePivotsOne<<<gridShape, blockShape>>>(d_Matrix, rowSize, colSize, currCol, currentPivotValue);
        cudaDeviceSynchronize();

		// Check for errors  
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_Matrix);

			return false;
		}
        
        MakePivotsColumnZero<<<rowSize, colSize>>>(d_Matrix, rowSize, colSize, currCol);
        
        // copy only next pivot to host to calculate temp accurately 
        if((currCol + 1)* colSize + (currCol+1) < colSize * rowSize)
        {
            err = cudaMemcpy(&h_Matrix[(currCol + 1)* colSize + (currCol+1)], &d_Matrix[(currCol + 1)* colSize + (currCol+1)], sizeof(double), cudaMemcpyDeviceToHost);
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

    clock_t total_time = clock()- start_time; 
   
    err = cudaMemcpy(h_Matrix, d_Matrix,size, cudaMemcpyDeviceToHost); // copy device matrix into host matrix to take correct pivot value 
    
    
    // Free device global memory
    err = cudaFree(d_Matrix);
    if (err != cudaSuccess)
    {   
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Gauss Jordan Elimination method calculation is finished! \n\n");
    
    float total_ms =  float( total_time) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %f milliseconds for %d x %d matrix with improved CUDA implementation!\n\n",  total_ms, rowSize, rowSize);

    saveMatrix(h_Matrix,rowSize);
    checkMatrix(rowSize);
    
    // Free host memory
    free(h_Matrix);

    printf("\n******************Program Finished!*****************\n");
return 0;
}