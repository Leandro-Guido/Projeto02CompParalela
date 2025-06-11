#!/bin/bash

# Configurações gerais
EXEC_CPU="./cpu.exe"
EXEC_GPU="./gpu.exe"
EXEC_CUDA="./cuda.exe"
EXEC_MPI="./main_mpi.exe"
DATASET="adult_0.5.csv"
LEARNING_RATE=0.01
EPOCHS=30
HIDDEN=5
RESULT_CSV="resultados_benchmark.csv"

# Escalabilidade forte
FORTE_CPU=(1 2 4 8)

FORTE_GPU=(8 16 32 64)

FORTE_CUDA=(8 16 32 64)

FORTE_MPI_PROC=(1 2 4)
FORTE_MPI_THREADS=(4 2 1)

FOLDS_FORTE=64

# Escalabilidade fraca
FRACA_CPU_THREADS=(1 2 4 8)
FRACA_CPU_FOLDS=(16 32 64 128)

FRACA_GPU_TEAMS=(16 32 64 128)
FRACA_GPU_FOLDS=(16 32 64 128)

FRACA_CUDA_TEAMS=(16 32 64 128)
FRACA_CUDA_FOLDS=(16 32 64 128)

FRACA_MPI_PROC=(1 2 4)
FRACA_MPI_THREADS=(4 2 1)
FRACA_MPI_FOLDS=(16 32 64 128)

# SEQUENCIAL
SEQUENCIAL_FOLDS=(16 32 64 128)

# Cria arquivo CSV de saída
echo "tipo,execucao,n_procs,n_threads,folds,tempo_s" > $RESULT_CSV

# Função para executar um teste e medir tempo
executar() {
    local tipo=$1    # seq | forte | fraca
    local label=$2   # CPU | GPU | CUDA | MPI
    local procs=$3
    local threads=$4
    local folds=$5
    local exec_cmd=$6

    echo "[$tipo][$label] procs=$procs threads=$threads folds=$folds..."
    START=$(date +%s.%N)

    if [[ "$label" == "MPI" ]]; then
        taskset -c 3-11 mpirun -np $procs --bind-to core $exec_cmd $threads $DATASET $folds $LEARNING_RATE $EPOCHS $HIDDEN
    else
        $exec_cmd $threads $DATASET $folds $LEARNING_RATE $EPOCHS $HIDDEN
    fi

    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    echo "$tipo,$label,$procs,$threads,$folds,$TIME" >> $RESULT_CSV
    echo " → Tempo: $TIME s"
    sleep 1
}

# echo "### SEQUENCIAL (CPU)"
# for fold in "${SEQUENCIAL_FOLDS[@]}"; do
#     executar "seq" "CPU" 1 1 $fold $EXEC_CPU
# done

echo "### ESCALABILIDADE FORTE"
# for t in "${FORTE_CPU[@]}"; do
#     executar "forte" "CPU" 1 $t $FOLDS_FORTE $EXEC_CPU
# done
# for t in "${FORTE_GPU[@]}"; do
#     executar "forte" "GPU" 1 $t $FOLDS_FORTE $EXEC_GPU
# done
# for t in "${FORTE_CUDA[@]}"; do
#     executar "forte" "CUDA" 1 $t $FOLDS_FORTE $EXEC_CUDA
# done
for i in "${!FORTE_MPI_PROC[@]}"; do
    executar "forte" "MPI" "${FORTE_MPI_PROC[$i]}" "${FORTE_MPI_THREADS[$i]}" $FOLDS_FORTE $EXEC_MPI
done

echo "### ESCALABILIDADE FRACA"
# for i in "${!FRACA_CPU_THREADS[@]}"; do
#     executar "fraca" "CPU" 1 "${FRACA_CPU_THREADS[$i]}" "${FRACA_CPU_FOLDS[$i]}" $EXEC_CPU
# done
# for i in "${!FRACA_GPU_TEAMS[@]}"; do
#     executar "fraca" "GPU" 1 "${FRACA_GPU_TEAMS[$i]}" "${FRACA_GPU_FOLDS[$i]}" $EXEC_GPU
# done
# for i in "${!FRACA_CUDA_TEAMS[@]}"; do
#     executar "fraca" "CUDA" 1 "${FRACA_CUDA_TEAMS[$i]}" "${FRACA_CUDA_FOLDS[$i]}" $EXEC_CUDA
# done
for i in "${!FRACA_MPI_PROC[@]}"; do
    executar "fraca" "MPI" "${FRACA_MPI_PROC[$i]}" "${FRACA_MPI_THREADS[$i]}" "${FRACA_MPI_FOLDS[$i]}" $EXEC_MPI
done

echo "Resultados salvos em $RESULT_CSV"
