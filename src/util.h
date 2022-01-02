#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

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
    printf("\n");
}
