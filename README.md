# parallel_seq_alignment

### TITLE
Parallel sequence alignment algorithm

### SUMMARY
We are interested in parallelizing the Needleman-Wunsch algorithm, which was originated to align DNA sequences. We will be implementing the parallel version of the algorithm utilizing CUDA on GPUs. 

### BACKGROUND 
The Needleman-Wunsch algorithm is used in bioinformatics to align protein sequences; it is widely used for optimal global alignment of DNA sequences. Global alignment means finding a matching sequence from a potentially very long sequence. Since the algorithm scales proportionally to the product of the length of two sequences in a sequential implementation, it can get very slow for large input sequences which is a common case for input to this algorithm. There have been efforts to speed up the algorithm sequentially, but for large inputs it would be wise to come up with a parallel algorithm that speeds up the operation. The motivation is further promoted by the fact the algorithm involves simple computations repeated a large number of times, which can be a good candidate for CUDA. 
	
### THE CHALLENGE
Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what do you hope to learn by doing the project? 
Each cell update is dependent on the cells nearby. We need to figure out how to increase the portion of parallel computation. Additionally, since the computation of a single cell has certain dependencies, there may be a lot of communication from the kernel to the device. We may need some effort to reduce this communication. Through this project, we hope to experience more with performance tuning with CUDA.

### RESOURCES
Our solution requires GPU utilization, and both the GHC and PSC machines support that. 
We will be starting from the bioPython open-source code base, which contains a C sequential implementation of the pairwise alignment. <br>
Link: https://github.com/biopython/biopython/blob/master/Bio/cpairwise2module.c
### GOALS AND DELIVERABLES
We plan to have a solid parallel CUDA version of the algorithm, which would hopefully produce 10x speedup. If we have additional time, we would hope to explore another parallel system like openMP with this algorithm. If we fall short in time, we would still have to achieve at least 5x speedup with our CUDA implementation.
In the poster session, we will demonstrate the numeric data like speedup and the amount of communication in graphs. We would show the speedup of different versions of CUDA programs, and it should be a demonstration of our incremental effort and deeper understanding of the parallel model.

### PLATFORM CHOICE
We will be using C++ and the CUDA interface to implement our solution. We are going to implement and test our solution on the GHP machines first, and then we will move to the PSC machines for further testing with additional hardware resources. We will develop the SPMD parallelism through CUDA because the update to each cell is identical.

### SCHEDULE:
Due Date <br>
3/23 <br>
Come up with a proposal <br>
3/26 <br>
Study the algorithm and read the sequential reference code <br>
4/2 <br>
Finish and test the sequential implementation <br>
4/9 <br>
Finish and test the first CUDA parallelized version on GHC machine <br>
4/11 <br>
Finish milestone report <br>
4/23 <br>
Performance tuning and performance evaluation on PSC machine. Compare the performance with the baseline implementation provided in the bioPython library <br>
4/29 <br>
Finish final report <br>
5/1 <br>
Finish poster <br>

### 4/11 MILESTONE
Current implementation takes input sequences seq1 and seq2 of lengths N1 and N2, and spawns min(N1,N2) workers. Each of the worker will start at matrix(1,workerIdx). The worker index will also be the delay for it to start, and each will work for N1 iters. This effectively updates the matrix elements forming lines that are parallel to the second diagonal. In each of the iteration, the calculations and updates are independent of other components calcuated concurrectly but relies on results from previous 2 iterations. 

Next step would be to compare performance versus the baselayer matrix calculation of the biopython library. To do so, input file parsing and file output need to be added, a python script can be created to randomly generate sequences and compare correctness and speed. 

### Presentation Video
https://drive.google.com/file/d/1wntaAA4nB8WSI_fvRgzSzyST4taIW35i/view?usp=sharing
