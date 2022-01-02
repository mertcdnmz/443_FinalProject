
#include <iostream>
#include <fstream>
#include <time.h>   

using namespace std;

void randomRowOperation(double *src, double *dest, int length){
    double randomScalar = rand() % 3 + 1 ;    
    int randomOperation = rand() % 4 + 1;

    double newNumber = 0 ;
    for(int i = 0; i < length ; i++){
        
        if(randomOperation == 1) 
        {   
            newNumber = src[i] + randomScalar*dest[i] ; 
        }
        else if( randomOperation == 2)
        {
            newNumber = src[i] - randomScalar*dest[i] ;
        }else if(randomOperation == 3)
        {
            newNumber = src[i] + dest[i] / randomScalar ;
        }else{
            newNumber = src[i] - dest[i]/randomScalar ;
        }
             
        src[i] = newNumber ;
    }
}

void saveMatrix(double** matrix, int rowSize){
    char outputFilename[50];
    snprintf(outputFilename, sizeof(char)*50, "../dataset/matrix_n_%i.txt", rowSize);
    
    FILE* file = fopen(outputFilename, "w");
    fprintf(file,"%d", rowSize);
    for(int i = 0 ; i < rowSize; i++){
        for(int j = 0 ; j < rowSize + 1; j++){
            if( j == 0 )   {
                fprintf(file, "\n");
            }
            if(matrix[i][j] == 0){
                fprintf(file, "%f", abs(matrix[i][j]));
            }else{
                fprintf(file, "%f" , matrix[i][j]);
            }
            fprintf(file, " ");

        }
    }
    fclose(file);
}

int main(int argc, char const *argv[]) {
    int rowSize = stoi(argv[1]);
    srand(time(NULL));
    int colSize = rowSize + 1 ;
    
    double* matrix[rowSize];
    for (int i = 0; i < rowSize; i++)
    {
        matrix[i] = (double*)malloc(colSize * sizeof(double));
    }

    for(int i = 0 ; i< rowSize; i++)
    {
        for (int j = 0 ; j < colSize ; j++)
        {
            if(i == j){
                matrix[i][j] = 1;
            }else if (j == rowSize){
                matrix[i][j] = rand() % 3;
            }
            else{
                matrix[i][j] = 0;
            }
        }
    }   

    for(int i= 0 ; i < rowSize * 8 ; i++){
        int row1 = rand() % rowSize ;
        int row2 = rand() % rowSize ;
        while(row1 == row2){
            row1 = rand() % rowSize;
        }
        randomRowOperation(matrix[row1], matrix[row2], colSize);
    }
    
    saveMatrix(matrix, rowSize);

    return 0 ;
}

