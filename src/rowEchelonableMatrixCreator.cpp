
#include <iostream>
#include <fstream>
#include <time.h>   

using namespace std;

void randomRowOperation(float *src, float *dest, int length){
    float randomScalar = rand() % 3 + 1 ;    
    int randomOperation = rand() % 4 + 1;

    float newNumber = 0 ;
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

int main(int argc, char const *argv[]) {
    int rowSize = stoi(argv[1]);
    srand(time(NULL));
    int colSize = rowSize + 1 ;
    
    float* matrix[rowSize];
    for (int i = 0; i < rowSize; i++)
    {
        matrix[i] = (float*)malloc(colSize * sizeof(float));
    }

    for(int i = 0 ; i< rowSize; i++)
    {
        for (int j = 0 ; j < colSize ; j++)
        {
            if(i == j){
                matrix[i][j] = 1;
            }else if (j == rowSize){
                matrix[i][j] = rand() % 5;
            }
            else{
                matrix[i][j] = 0;
            }
        }
    }   

    for(int i = 0 ; i < rowSize ; i++ ) {

        for(int j = 0 ; j < colSize; j++){
            cout<<matrix[i][j] <<" " ;
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
    
    for(int i = 0 ; i < rowSize ; i++ ) {

        for(int j = 0 ; j < colSize; j++){
            printf ("%f ", matrix[i][j]);
        }
    
    }


    ofstream matrixFile;
    matrixFile.open ("../dataset/matrix_n_"+to_string(rowSize)+".txt", ios::trunc);
    matrixFile << rowSize << endl ;
    for(int i = 0 ; i< rowSize ; i++)
    {
        for(int j = 0; j< colSize ; j++)
        {
            matrixFile << matrix[i][j] << " ";
        }
        matrixFile << endl ;

    }
    matrixFile.close();
    


}

