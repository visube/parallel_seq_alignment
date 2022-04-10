#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


void alignCuda(int N1, int N2, int* seq1, int*seq2, int* matrix);
void printCudaInfo();

// return GB/s
float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void usage(const char *progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char **argv) {
    /*
    int N = 20 * 1000 * 1000;

    // Parse commandline options
    int opt;
    static struct option long_options[] = {
        {"arraysize", 1, 0, 'n'},
        {"help", 0, 0, '?'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // End parsing of commandline options

    const float alpha = 2.0f;
    float *xarray = new float[N];
    float *yarray = new float[N];
    float *resultarray = new float[N];

    // Load X, Y, store result
   // int totalBytes = sizeof(float) * 3 * N;

    for (int i = 0; i < N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
    }

    printCudaInfo();

    for (int i = 0; i < 3; i++) {
        saxpyCuda(N, alpha, xarray, yarray, resultarray);
    }

    delete[] xarray;
    delete[] yarray;
    delete[] resultarray;
    */

    printCudaInfo();

    //Manually set test input for now
    int N1 = 7;
    int N2 = 7;
    int* seq1 = (int*)calloc(7, sizeof(int));
    int* seq2 = (int*)calloc(7, sizeof(int));
    seq1[0] = 3;
    seq1[1] = 0;
    seq1[2] = 1;
    seq1[3] = 1;
    seq1[4] = 0;
    seq1[5] = 2;
    seq1[6] = 0;
    seq2[0] = 3;
    seq2[1] = 2;
    seq2[2] = 0;
    seq2[3] = 1;
    seq2[4] = 3;
    seq2[5] = 2;
    seq2[6] = 3;

    /*
    seq1 = {3, 0, 1, 1, 0, 2, 0};
    seq2 = {3, 2, 0, 1, 3, 2, 3};*/
    
    int *matrix = (int*)calloc((N1+1)*(N2+1), sizeof(int));

    for(int row = 0; row < N1 + 1; row++){
        matrix[row*(N2+1)] = -1*row;
    }
    for(int col = 1; col < N2 + 1; col++){
        matrix[col] = -1*col;
    }

    printf("original matrix:\n");
    for(int row = 0; row < N1 + 1; row++){
        for(int col = 0; col < N2 + 1; col++){
            if(matrix[row*(N2+1)+col] >=0){
                printf("  %d", matrix[row*(N2+1)+col]);
            }else{
                printf(" %d", matrix[row*(N2+1)+col]);
            }
            
        }
        printf("\n");
    }

    alignCuda(N1, N2, seq1, seq2, matrix);

    printf("final matrix:\n");
    for(int row = 0; row < N1 + 1; row++){
        for(int col = 0; col < N2 + 1; col++){
            if(matrix[row*(N2+1)+col] >=0){
                printf("  %d", matrix[row*(N2+1)+col]);
            }else{
                printf(" %d", matrix[row*(N2+1)+col]);
            }
            
        }
        printf("\n");
    }

    return 0;

}
