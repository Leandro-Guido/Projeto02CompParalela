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

    mudar os valores entre parentesis

    COMPILAR: g++ -O3 -fopenmp -std=c++11 -DSCHEDULE="schedule(dynamic, 1)" -DVERBOSE=1 main.cpp NeuralNetwork.cpp -o (nome).exe
    FLAG -DVERBOSE (1 = mostra execução por epocas, 2 = mostra também o resultado na network inteira)

    EXECUTAR: time ./(nome).exe <num_threads> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>
    ex.:      time ./seq.exe 1 "adult_1.csv" 4 0.3 70 5
    no caso do parcode é indicado filtrar para usar threads diferentes da 0 e da 1: time taskset -c 2-4 ./seq.exe 2 "adult_1.csv" 8 0.01 100 5