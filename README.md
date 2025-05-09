Leandro Guido e Matheus Sinis

ii) Código da MLP sequencial foi retirado de: https://github.com/Cr4ckC4t/neural-network-from-scratch/tree/main

iii) PARALELIZAÇÃO OPENMP COMENTADA EM main.cpp -> função evaluate_network()

iv) EXECUÇÃO OpenMP NO PARCODE (compilação e tempos)

1 thread:

real    0m14.646s
user    0m14.605s
sys     0m0.036s

2 threads:

real    0m19.969s
user    0m34.381s
sys     0m0.048s

4 threads:

real    0m12.526s
user    0m35.688s
sys     0m0.068s


8 threads:

real    0m12.399s
user    0m35.421s
sys     0m0.100s

v) ler o lembrete

vi) COMPILAÇÃO E EXECUÇÃO DOS CÓDIGOS
OpenMP:
    CONFIGURAR: trocar na função main (main.cpp) o número de threads omp_set_num_threads(<número de threads>);
    COMPILAR: g++ -O3 -Wall -Wpedantic -fopenmp main.cpp NeuralNetwork.cpp -o <nome>.exe
    EXECUTAR: time ./<nome>.exe

LEMBRETE AO PROFESSOR: Na hora da apresentação nós não tínhamos feito ainda os slides e nem a parte do MPI (todo o resto foi apresentado), a tentativa do MPI (com explicações de como seria feito) está junto dos outros códigos como main_mpi.cpp (não conseguimos fazer rodar sem estourar a memória).
