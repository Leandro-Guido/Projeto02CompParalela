Leandro Guido e Matheus Sinis

O Código da MLP sequencial foi adaptado de: https://github.com/Cr4ckC4t/neural-network-from-scratch/tree/main

A explicação da paralelização está comentada em cada código

A execução foi feita em um notebook com as seguintes especificações:

Processador:
    Intel Core i5 - 13th Gen (2,10 GHz)
    12 Threads

Memória:
    16GB (DDR4, 3200 MHz)
    
GPU:
    RTX 3050 
    VRAM: 6 GB DDR6

Todas as execuções podem ser encontradas em resultados_benchmark.csv

COMPILAÇÃO E EXECUÇÃO DOS CÓDIGOS

## OpenMP CPU:

    FLAGS DE COMPILAÇÃO:
    1) "-DVERBOSE=<num>" (1 = mostra execução por epocas, 2 = mostra também o resultado na network inteira)
    2) "-DSLOW=<num>" (1 = desacelera o código para melhorar a visualização da execução)

    COMPILAR: gcc -fopenmp -o cpu.exe main.c
    EXECUTAR: ./cpu.exe <num_threads> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>
    ex.:      ./cpu.exe 1 "adult_1.csv" 64 0.01 30 5

## OpenMP GPU:

    FLAGS DE COMPILAÇÃO:
    1) "-DVERBOSE=<num>" (1 = mostra execução por epocas, 2 = mostra também o resultado na network inteira)
    2) "-DSLOW=<num>" (1 = desacelera o código para melhorar a visualização da execução)

    COMPILAR: gcc -fopenmp -fcf-protection=none -fno-stack-protector -no-pie -o gpu.exe main_gpu.c
    EXECUTAR: ./gpu.exe <num_teams> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>
    ex.:      ./gpu.exe 8 "adult_1.csv" 64 0.01 30 5

## MPI:

    FLAGS DE COMPILAÇÃO:
    1) "-DVERBOSE=<num>" (1 = mostra execução por epocas)

    COMPILAR: mpicc -fopenmp -o main_mpi.exe main_mpi.c
    EXECUTAR: mpirun -np <num_proc> ./main_mpi.exe <num_teams> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>
    ex:       mpirun -np 1 ./main_mpi.exe 4 adult_1.csv 64 0.01 30 5

## CUDA:

    COMPILAR: nvcc -arch=sm_61 CUDA.cu -o cuda.exe
    EXECUTAR: ./cuda.exe <num_teams> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>
    ex:       ./cuda.exe 8 "adult_1.csv" 64  0.01 30 5