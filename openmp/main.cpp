#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <errno.h> 
#include <algorithm>
#include <stdlib.h>
#include <getopt.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

static void show_help(const char *program_path) {
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
}

void printMatrix(int* matrix, int x, int y);
void compute(int* matrix, int n1, int n2, int*seq1, int*seq2);

int main(int argc, char **argv) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    // time initialization

    auto init_start = Clock::now();
    double init_time = 0;
    string input_file;

    // Parse commandline options
    int opt;
    int file_specified = 0;
    static struct option long_options[] = {
        {"file", 0, 0, 'f'},
        {"threads", 0, 0, 'n'},
        {"help", 0, 0, '?'},
        {0, 0, 0, 0},
    };
    int num_of_threads = 1;
    while ((opt = getopt_long(argc, argv, "f:n:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'f':
            input_file = optarg;
            file_specified = 1;
            break;
        case 'n':
            num_of_threads = stoi(optarg);
            break;
        case '?':
        default:
            show_help(argv[0]);
            return 1;
        }
    }
    // End parsing of commandline options
    
    if(file_specified){
        cout << "Input file name: " << input_file << "\n";
        cout << "Number of threads: " << num_of_threads << "\n";
    }else{
        show_help(argv[0]);
        return 1;
    }
    
    string line;
    ifstream infile(input_file);
    // first line is sequence 1
    getline(infile,line);
    cout << "Input Sequence 1 " << line << "\n";
    int N1 = line.size();
    int* seq1 = (int*)calloc(N1, sizeof(int));

    for(string::size_type i = 0; i < (unsigned)N1; ++i){
        seq1[i] = (int)line[i] - 48;
    }
    // second line is sequence 2
    getline(infile,line);
    cout << "Input Sequence 2 " << line << "\n";
    int N2 = line.size();
    int* seq2 = (int*)calloc(N2, sizeof(int));
    for(string::size_type i = 0; i < (unsigned)N2; ++i){
        seq2[i] = (int)line[i] - 48;
    }
    N1++;
    N2++;
    omp_set_num_threads(num_of_threads);
    int* score_matrix = (int*)calloc(N1 * N2, sizeof(int));

    // initialize the first row of the matrix
    for (int i = 0; i < N1; i++) {
        score_matrix[i] = -i;
    }
    // initialize the first column of the matrix
    for (int i = 0; i < N2; i++) {
        score_matrix[i * N1] = -i;
    }
    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto compute_start = Clock::now();
    double compute_time = 0;
    compute(score_matrix, N1, N2, seq1, seq2);
    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    ofstream outfile;
    outfile.open("openmp_output_matrix");
    if(!outfile){
        cout << "Output file creation failed\n";
    }else{
        cout << "Output file creation succeeded\n";
        for(int i = 0; i < N2; i++){
            for(int j = 0; j < N1; j++){
                outfile << score_matrix[i * N1 + j] << " ";
            }
            outfile << "\n";
        }
        
    }
    free(score_matrix);
    free(seq1);
    free(seq2);
}

void printMatrix(int* matrix, int n1, int n2) {
    for(int i = 0; i < n2; i++){
        for(int j = 0; j < n1; j++){
           printf(" %d    ", matrix[i * n1 + j]);
        }
        printf("\n");
    }
}

void compute(int* matrix, int n1, int n2, int*seq1, int*seq2) {
    #pragma omp parallel 
    for(int i = 1; i < n1 + n2 - 1; i++) {
        #pragma omp for schedule(auto)
        for(int j = max(1, i-n1+2); j < min(n2, i+1); j++) {
            int y_index = j;
            int x_index = i - j + 1;
            int match = seq1[x_index - 1] == seq2[y_index - 1]? 1 : -1;
            int match_score = matrix[(y_index - 1)* n1 + x_index - 1] + match;
            int gap_score_left = matrix[y_index * n1 + x_index - 1] -1;
            int gap_score_top = matrix[(y_index - 1) * n1 + x_index] -1;
            int max_score = max(gap_score_left, gap_score_top);
            max_score = max(match_score, max_score);
            matrix[y_index * n1 + x_index] = max_score;
        }
    }
}