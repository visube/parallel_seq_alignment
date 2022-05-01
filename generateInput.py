import random

input_length = 100
seq1 = []
seq2 = []

with open('input_seq','w') as f:
    for n in range(input_length):
        x = random.randint(0,3)
        f.write(str(x))
        seq1.append(x)
    f.write("\n")
    for n in range(input_length):
        x = random.randint(0,3)
        f.write(str(x))
        seq2.append(x)
    f.write("\n")

matrix = [ [0] * (input_length+1) for i in range(input_length+1)]

for n in range(input_length+1):
    matrix[n][0] = -n
    matrix[0][n] = -n

for n in range(input_length):
    for m in range(input_length):
        match = 0
        if(seq1[n] == seq2[m]):
            match = 1
        else:
            match = -1
        best = matrix[n+1][m] - 1
        if(matrix[n][m] + match > best):
            best = matrix[n][m]+match
        if(matrix[n][m+1] - 1 > best):
            best = matrix[n][m+1] -1
        matrix[n+1][m+1] = best

with open('output_matrix_ref','w') as f:
    for n in range(input_length+1):
        for m in range(input_length+1):
            f.write(str(matrix[n][m]))
            f.write(" ")
        f.write("\n")



