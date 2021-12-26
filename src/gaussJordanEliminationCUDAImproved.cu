#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
using namespace std;


void PrintMatrix(double *matrix, int n);
__global__ void PrintMatrixGPU(double *matrix, int n);


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
__global__ void MakePivotsColumnZero(double* matrix, int rowSize, int colSize, int currCol, int currRow) 
{
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y ;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y ; 
    int threadsPerBlock  = blockDim.x * blockDim.y ;
    int index = blockNumInGrid * threadsPerBlock + threadNumInBlock ;

    int tID = currRow * colSize +  index;

    __shared__ double rateWithPivot; 
    rateWithPivot = matrix[ currRow * colSize + currCol ] ;

    int colNumber = tID % colSize;
     
    if(currRow != currCol && index < colSize ){
        matrix[tID] = matrix[tID] - (rateWithPivot * matrix[( currCol * colSize ) + colNumber ]);
        __syncthreads();
    }
    
}

void PrintMatrix(double *a, int n)
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


void createMatrix(double *matrix, int rowSize, bool input){
    char filename[50];
    if(input){
    snprintf(filename, sizeof(char)*50, "../dataset/matrix_n_%i.txt", rowSize);    
    }else{
    snprintf(filename, sizeof(char)*50, "../dataset/output_matrix_%i.txt", rowSize);
    }
    FILE* file = fopen (filename, "r");
    int matrixSize = 0;
    fscanf (file, " %d", &matrixSize);
    for(int i = 0 ; i < rowSize * (rowSize + 1 ); i++)
    {   
        double number = 0 ;
        fscanf(file, "%lf",&number);
        matrix[i] = number;
    }
    fclose (file);
}

void saveMatrix(double* matrix, int rowSize){
    char outputFilename[50];
    snprintf(outputFilename, sizeof(char)*50, "../dataset/output_matrix_%i.txt", rowSize);
    
    FILE* file = fopen(outputFilename, "w");
    fprintf(file,"%d", rowSize);
    for(int i = 0 ; i < rowSize * (rowSize+1) ; i++){
        if(i % (rowSize + 1) == 0){
            fprintf(file, "\n");
        }
        if(matrix[i] == 0){
            fprintf(file, "%f", abs(matrix[i]));
        }else{
            fprintf(file, "%f" , matrix[i]);
        }
        fprintf(file, " ");
    }
    fclose(file);
}

// This function verify the results of variants with putting the output values to input matrix and compare with last column. 
void checkMatrix(int rowSize){
    size_t size = rowSize * (rowSize + 1) * sizeof(double);
    double* inputMatrix = (double*) malloc(size);
    double* outputMatrix = (double*) malloc(size)   ;

    createMatrix(inputMatrix, rowSize, true);
    createMatrix(outputMatrix, rowSize, false);
    int colSize = rowSize + 1 ;
    for(int i = 0 ; i < rowSize ; i++){
        double actualValue = 0; 
        for(int j = 0; j < rowSize ; j++ )
        { 
            actualValue += inputMatrix[ i * colSize + j ] * outputMatrix[ j * colSize + colSize - 1] ;
        }
        double expectedValue = inputMatrix[i * colSize + colSize - 1] ; 
        if(actualValue - expectedValue >= 1e-2){
            printf("There is a  difference( > 1e-2 )!. Row Number: %d\n", i+1);
            printf("Actual value -> %f\nExpected value -> %f\n", actualValue, expectedValue);
        }
    }
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
  

    for(int currCol = 0; currCol < rowSize  ; currCol++ )
    {   
        
        dim3 gridShape(1,1) ;
        dim3 blockShape(colSize,1 ) ; //multi dimensional 
        double currentPivotValue = h_Matrix[ currCol * colSize + currCol ];
        MakePivotsOne<<<gridShape, blockShape>>>(d_Matrix, rowSize, colSize, currCol, currentPivotValue);
        cudaDeviceSynchronize();
        
        
        
        // PrintMatrixGPU<<<1,1>>>(d_Matrix, colSize);
		// Check for errors
        
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_Matrix);

			return false;
		}
        
        for( int i = 0 ; i < rowSize ; i++){
            MakePivotsColumnZero<<<gridShape, blockShape>>>(d_Matrix, rowSize, colSize, currCol, i);
            cudaDeviceSynchronize();
            
        }
        
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
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    err = cudaMemcpy(h_Matrix, d_Matrix,size, cudaMemcpyDeviceToHost); // copy device matrix into host matrix to take correct pivot value 
    
    
    // Free device global memory
    err = cudaFree(d_Matrix);
    if (err != cudaSuccess)
    {   
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Gauss Jordan Elimination method calculation is finished! \n\n");
    printf("Total milliseconds: %f\n\n", milliseconds);

    //printf("Row echelon form of input matrix:\n") ;
    //PrintMatrix(h_Matrix,colSize);

    saveMatrix(h_Matrix,rowSize);
    checkMatrix(rowSize);
    
    // Free host memory
    free(h_Matrix);

    printf("******************Program Finished!*****************\n");
return 0;
}