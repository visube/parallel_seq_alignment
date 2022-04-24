#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


void alignCuda(int N1, int N2, int* seq1, int*seq2, int* matrix);
void printCudaInfo();

// return GB/s
float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void usage(const char *progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  --file    Input filename\n");
    printf("  -?  --help    This message\n");
}

int main(int argc, char **argv) {
    std::string input_file;

    // Parse commandline options
    int opt;
    int file_specified = 0;
    static struct option long_options[] = {
        {"file", 0, 0, 'f'},
        {"help", 0, 0, '?'},
        {0, 0, 0, 0}
    };
    
    while ((opt = getopt_long(argc, argv, "f:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'f':
            input_file = optarg;
            file_specified = 1;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // End parsing of commandline options
    
    if(file_specified){
        cout << "Input file name: " << input_file << "\n";
    }else{
        usage(argv[0]);
        return 1;
    }
    
    string line;
    ifstream infile("input_seq");
    
    // first line is sizes of two input sequences
    getline(infile,line);
    if(line.size() != 2){
        cout << "First line of input file has size " << line.size() << " exiting\n";
        return 1;
    }
    int N1, N2;
    N1 = (int)line[0] - 48;
    N2 = (int)line[1] - 48;
    int* seq1 = (int*)calloc(N1, sizeof(int));
    int* seq2 = (int*)calloc(N2, sizeof(int));

    // second line is sequence 1
    getline(infile,line);
    for(string::size_type i = 0; i < line.size(); ++i){
        seq1[i] = (int)line[i] - 48;
    }
    // third line is sequence 2
    getline(infile,line);
    for(string::size_type i = 0; i < line.size(); ++i){
        seq2[i] = (int)line[i] - 48;
    }

    printCudaInfo();
    
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
    for(int i = 0; i < N2 + 1; i++){
        if(i == 0){
            printf("    ");
        }else{
            printf("  %d", seq2[i]);
        }
    }
    printf("\n");
    for(int row = 0; row < N1 + 1; row++){
        if(row != 0){
            printf("%d", seq1[row-1]);
        }else{
            printf(" ");
        }
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
