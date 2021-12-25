#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

using namespace std;


void PrintMatrix(float *matrix, int n);





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
    int rowSize = stoi(argv[1]);
    int colSize = rowSize + 1 ;
  
    float *matrix  = (float*) malloc(rowSize * colSize * sizeof(float));
    
    createMatrix(matrix, rowSize);
    
    PrintMatrix(matrix, colSize) ;
    
    GaussionElimination(matrix,rowSize,colSize);
 
    cout << "Final Augmented Matrix is : " << endl;
    PrintMatrix(matrix, colSize);

    return 0;
}