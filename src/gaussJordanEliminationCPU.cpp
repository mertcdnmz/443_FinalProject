#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "util.h"
using namespace std;

void GaussionElimination(double *matrix, int numberOfRows, int numberOfColumns)
{
    double temp = 0 ;
    for(int currCol = 0; currCol < numberOfRows  ; currCol++ )
    {

        temp = matrix[ currCol * numberOfColumns + currCol ]; 
        for(int j = 0 ; j < numberOfColumns ; j++)
        {
            matrix[ currCol * numberOfColumns + j ] = (matrix[ currCol * numberOfColumns + j ]) / temp;
        }
            
        for(int i = 0 ; i < numberOfRows; i++)
        {
            temp = matrix[ i * numberOfColumns + currCol ]; // column element that related with current pivot 
            for(int j = 0 ; j < numberOfColumns ; j++)
            {
                if(i != currCol)
                {
                    matrix[ i * numberOfColumns + j ] = matrix[ i * numberOfColumns + j ] -  (temp * matrix[ currCol * numberOfColumns + j ]); // e.g R(i) = R(i) - temp * R(currentPivot)
                }
            }
        }
    }

}

int main(int argc, char const *argv[]) {
    int rowSize = stoi(argv[1]);
    int colSize = rowSize + 1 ;
  
    double *matrix  = (double*) malloc(rowSize * colSize * sizeof(double));
    
    createMatrix(matrix, rowSize,true);
    
    //PrintMatrix(matrix, colSize) ;
    
    clock_t start_time = clock();
    printf("Gauss Jordan Elimination method calculation is started! \n\n");
    GaussionElimination(matrix,rowSize,colSize);
    clock_t total_time = clock() - start_time ;
    printf("Gauss Jordan Elimination method calculation is finished! \n\n");

    float total_ms =  float( total_time) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %f milliseconds for %d x %d matrix with CPU implementation!\n",  total_ms, rowSize, rowSize);
    saveMatrix(matrix,rowSize);
    
    checkMatrix(rowSize);

    //cout << "Final Augmented Matrix is : " << endl;
    //PrintMatrix(matrix, colSize);

    return 0;
}