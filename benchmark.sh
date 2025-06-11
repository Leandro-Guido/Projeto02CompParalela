#!/bin/bash

# Configurações gerais
EXEC_CPU="./cpu.exe"
EXEC_GPU="./gpu.exe"
DATASET="adult_0.5.csv"
LEARNING_RATE=0.01
EPOCHS=30
HIDDEN=5
RESULT_CSV="resultados_benchmark.csv"

# Escalabilidade forte
STRONG_CPU=(1 2 4 8)
STRONG_GPU=(8 16 32 64)
FOLDS_FORTE=64

# Escalabilidade fraca
WEAK_CPU_THREADS=(1 2 4 8)
WEAK_CPU_FOLDS=(16 32 64 128)

WEAK_GPU_THREADS=(1 2 4 8)
WEAK_GPU_FOLDS=(16 32 64 128)

# Cria arquivo CSV de saída
echo "tipo,execucao,n_threads,folds,tempo_s" > $RESULT_CSV

# Função para executar um teste e medir tempo
executar() {
    local exec=$1
    local threads=$2
    local folds=$3
    local tipo=$4
    local label=$5
    local log="log_${label}_${threads}t_${folds}f.txt"

    echo "🔧 Executando $label com $threads t e $folds folds..."
    START=$(date +%s.%N)

    $exec $threads $DATASET $folds $LEARNING_RATE $EPOCHS $HIDDEN > /dev/null 2>&1 # trocar para > "$log" se quiser salvar o log

    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    echo "$tipo,$label,$threads,$folds,$TIME" >> $RESULT_CSV
    echo "✅ Tempo: $TIME segundos"
    sleep 2
}

# echo "🚀 ESCALABILIDADE FORTE (CPU)"
# for t in "${STRONG_CPU[@]}"; do
#     executar "$EXEC_CPU" $t $FOLDS_FORTE "forte" "CPU"
# done

echo "🚀 ESCALABILIDADE FORTE (GPU)"
for t in "${STRONG_GPU[@]}"; do
    executar "$EXEC_GPU" $t $FOLDS_FORTE "forte" "GPU"
done

# echo "🚀 ESCALABILIDADE FRACA (CPU)"
# for i in "${!WEAK_CPU_THREADS[@]}"; do
#     t=${WEAK_CPU_THREADS[$i]}
#     f=${WEAK_CPU_FOLDS[$i]}
#     executar "$EXEC_CPU" $t $f "fraca" "CPU"
# done

echo "🚀 ESCALABILIDADE FRACA (GPU)"
for i in "${!WEAK_GPU_THREADS[@]}"; do
    t=${WEAK_GPU_THREADS[$i]}
    f=${WEAK_GPU_FOLDS[$i]}
    executar "$EXEC_GPU" $t $f "fraca" "GPU"
done

echo "📁 Resultados salvos em $RESULT_CSV"
