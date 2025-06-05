#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <regex>
#include <iterator>
#include <map>
#include <numeric>
#include <cmath>
#include "NeuralNetwork.h"
#include <omp.h> // Será removido
#include <cuda_runtime.h> // Para funções CUDA

#ifndef SCHEDULE
	#define SCHEDULE
#endif

// Funções utilitárias (podem ser deixadas como estão se não manipularem diretamente a GPU)
std::vector<std::vector<float>> load_csv_data(std::string filename);
float accuracy_metric(std::vector<int> expect, std::vector<int> predict);

// Forward declarations para as funções que interagirão com a GPU
// Um wrapper para o kernel CUDA para avaliar uma rede
__global__ void train_and_predict_kernel(
    Network* d_networks, // Array de redes na GPU
    float* d_train_data,
    int* d_train_offsets, // Offsets para dados de treino de cada fold
    int* d_train_sizes, // Tamanhos de dados de treino de cada fold
    float* d_test_data,
    int* d_test_offsets, // Offsets para dados de teste de cada fold
    int* d_test_sizes, // Tamanhos de dados de teste de cada fold
    float* d_expected_outputs, // Outputs esperados na GPU
    int* d_expected_offsets,
    int* d_expected_sizes,
    float l_rate,
    int n_epoch,
    int n_outputs,
    int n_inputs_per_row, // Número total de entradas por linha de dado (incluindo o "target" que será removido)
    float* d_scores // Array para armazenar as pontuações de cada fold
);

// Função do host para inicializar uma rede na GPU
__global__ void initialize_network_kernel(Network* d_network, int n_inputs, int n_hidden, int n_outputs);

// Função do host para liberar memória de uma rede na GPU
__global__ void destroy_network_kernel(Network* d_network);


/*
* Este main function será adaptado para o paralelismo CUDA.
*/
int main(int argc, char* argv[]) {
	std::cout << "Neural Network with Backpropagation in C++ from scratch (CUDA)" << std::endl;

	if (argc < 7) {
        std::cerr << "Uso: " << argv[0] << " <num_threads> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>\n";
        return 1;
    }

	// num_threads agora é para o número de threads CUDA se quiser usar, mas não será diretamente omp_set_num_threads
    // int num_threads = std::atoi(argv[1]); // Não usado diretamente para OpenMP
    std::string dataset_path = argv[2];

    int n_folds = std::atoi(argv[3]);
    float l_rate = std::atof(argv[4]);
    int n_epoch = std::atoi(argv[5]);
    int n_hidden = std::atoi(argv[6]);

    // Remover omp_set_num_threads(num_threads);

	std::vector<std::vector<float>> csv_data;
	csv_data = load_csv_data(dataset_path);

	// Teste a rede neural implementada com CUDA
	float mean = evaluate_network_cuda(csv_data, n_folds, l_rate, n_epoch, n_hidden);

	std::cout << "Mean accuracy: " << mean << std::endl;

	return 0;
}

// Nova função para avaliação da rede com CUDA
float evaluate_network_cuda(std::vector<std::vector<float>> dataset, int n_folds, float l_rate, int n_epoch, int n_hidden) {
    // Inicialização do PRNG para CPU
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Preparar os folds como antes (ainda na CPU)
    std::vector<std::vector<std::vector<float>>> dataset_splits;
    size_t fold_size = static_cast<unsigned int>(dataset.size() / n_folds);

    for (int f = 0; f < n_folds; f++) {
        std::vector<std::vector<float>> fold;
        while (fold.size() < fold_size) {
            int n = rand() % dataset.size();
            std::swap(dataset[n], dataset.back());
            fold.push_back(dataset.back());
            dataset.pop_back();
        }
        dataset_splits.push_back(fold);
    }

    // Pré-processamento e preparação dos dados para GPU
    std::vector<std::vector<std::vector<float>>> train_sets_all(dataset_splits.size());
    std::vector<std::vector<std::vector<float>>> test_sets(dataset_splits.size());
    std::vector<std::vector<int>> expected_all(dataset_splits.size());

    // Buffers flat para dados de treino, teste e expected (para GPU)
    std::vector<float> h_train_data_flat;
    std::vector<int> h_train_offsets; // Offsets de início de cada fold no buffer flat
    std::vector<int> h_train_sizes; // Tamanho de cada fold (num_rows * num_cols)

    std::vector<float> h_test_data_flat;
    std::vector<int> h_test_offsets;
    std::vector<int> h_test_sizes;

    std::vector<float> h_expected_outputs_flat;
    std::vector<int> h_expected_offsets;
    std::vector<int> h_expected_sizes;
    
    int n_inputs_per_row = (dataset_splits[0].empty() ? 0 : dataset_splits[0][0].size());

    for (size_t i = 0; i < dataset_splits.size(); ++i) {
        auto splits_copy = dataset_splits;
        std::swap(splits_copy[i], splits_copy.back());

        std::vector<std::vector<float>> test_set = splits_copy.back();
        splits_copy.pop_back();

        std::vector<std::vector<float>> train_set;
        for (auto& s : splits_copy) {
            train_set.insert(train_set.end(), s.begin(), s.end());
        }

        std::vector<int> expected;
        for (auto& row : test_set) {
            expected.push_back(static_cast<int>(row.back()));
            row.back() = 42; // "Limpa" o target para o teste
        }

        train_sets_all[i] = std::move(train_set);
        test_sets[i] = std::move(test_set);
        expected_all[i] = std::move(expected);

        // Flatten os dados para a GPU
        h_train_offsets.push_back(h_train_data_flat.size());
        h_train_sizes.push_back(train_sets_all[i].size()); // Número de linhas
        for (const auto& row : train_sets_all[i]) {
            h_train_data_flat.insert(h_train_data_flat.end(), row.begin(), row.end());
        }

        h_test_offsets.push_back(h_test_data_flat.size());
        h_test_sizes.push_back(test_sets[i].size()); // Número de linhas
        for (const auto& row : test_sets[i]) {
            h_test_data_flat.insert(h_test_data_flat.end(), row.begin(), row.end());
        }

        h_expected_offsets.push_back(h_expected_outputs_flat.size());
        h_expected_sizes.push_back(expected_all[i].size());
        for (int val : expected_all[i]) {
            h_expected_outputs_flat.push_back(static_cast<float>(val));
        }
    }

    // Alocar memória na GPU
    Network* d_networks;
    float* d_train_data_flat;
    int* d_train_offsets;
    int* d_train_sizes;

    float* d_test_data_flat;
    int* d_test_offsets;
    int* d_test_sizes;

    float* d_expected_outputs_flat;
    int* d_expected_offsets;
    int* d_expected_sizes;
    
    float* d_scores;

    cudaMalloc(&d_networks, n_folds * sizeof(Network)); // Um array de objetos Network na GPU
    cudaMalloc(&d_train_data_flat, h_train_data_flat.size() * sizeof(float));
    cudaMalloc(&d_train_offsets, h_train_offsets.size() * sizeof(int));
    cudaMalloc(&d_train_sizes, h_train_sizes.size() * sizeof(int));

    cudaMalloc(&d_test_data_flat, h_test_data_flat.size() * sizeof(float));
    cudaMalloc(&d_test_offsets, h_test_offsets.size() * sizeof(int));
    cudaMalloc(&d_test_sizes, h_test_sizes.size() * sizeof(int));

    cudaMalloc(&d_expected_outputs_flat, h_expected_outputs_flat.size() * sizeof(float));
    cudaMalloc(&d_expected_offsets, h_expected_offsets.size() * sizeof(int));
    cudaMalloc(&d_expected_sizes, h_expected_sizes.size() * sizeof(int));

    cudaMalloc(&d_scores, n_folds * sizeof(float)); // Para armazenar os resultados

    // Copiar dados do host para o device
    cudaMemcpy(d_train_data_flat, h_train_data_flat.data(), h_train_data_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_offsets, h_train_offsets.data(), h_train_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_sizes, h_train_sizes.data(), h_train_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_test_data_flat, h_test_data_flat.data(), h_test_data_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_offsets, h_test_offsets.data(), h_test_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_sizes, h_test_sizes.data(), h_test_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_expected_outputs_flat, h_expected_outputs_flat.data(), h_expected_outputs_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expected_offsets, h_expected_offsets.data(), h_expected_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expected_sizes, h_expected_sizes.data(), h_expected_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Inicializar as redes na GPU
    // Cada thread do bloco 0, grid 0 inicializa uma rede
    initialize_network_kernel<<<1, n_folds>>>(d_networks, n_inputs_per_row -1, n_hidden, dataset_splits[0].empty() ? 0 : (std::set<float>(train_sets_all[0].back().begin() + (train_sets_all[0].back().size()-1), train_sets_all[0].back().end())).size() );

    // Lançar o kernel principal para treinar e prever
    train_and_predict_kernel<<<1, n_folds>>>(
        d_networks,
        d_train_data_flat, d_train_offsets, d_train_sizes,
        d_test_data_flat, d_test_offsets, d_test_sizes,
        d_expected_outputs_flat, d_expected_offsets, d_expected_sizes,
        l_rate, n_epoch,
        dataset_splits[0].empty() ? 0 : (std::set<float>(train_sets_all[0].back().begin() + (train_sets_all[0].back().size()-1), train_sets_all[0].back().end())).size(), // n_outputs, inferido do primeiro fold
        n_inputs_per_row,
        d_scores
    );

    cudaDeviceSynchronize(); // Espera a GPU terminar

    // Copiar resultados de volta para o host
    std::vector<float> h_scores(n_folds);
    cudaMemcpy(h_scores.data(), d_scores, n_folds * sizeof(float), cudaMemcpyDeviceToHost);

    // Calcular a média
    float sum_scores = 0.0f;
    for (float score : h_scores) {
        sum_scores += score;
    }

    // Liberar memória da GPU
    cudaFree(d_networks);
    cudaFree(d_train_data_flat);
    cudaFree(d_train_offsets);
    cudaFree(d_train_sizes);
    cudaFree(d_test_data_flat);
    cudaFree(d_test_offsets);
    cudaFree(d_test_sizes);
    cudaFree(d_expected_outputs_flat);
    cudaFree(d_expected_offsets);
    cudaFree(d_expected_sizes);
    cudaFree(d_scores);

    // Chamar um kernel para destruir as redes na GPU (liberar memória alocada internamente por Network)
    destroy_network_kernel<<<1, n_folds>>>(d_networks);

    return sum_scores / n_folds;
}


// Kernel CUDA para inicializar as redes na GPU
__global__ void initialize_network_kernel(Network* d_networks, int n_inputs, int n_hidden, int n_outputs) {
    int idx = threadIdx.x;
    if (idx < gridDim.x * blockDim.x) { // Garante que não exceda o número de redes
        d_networks[idx].initialize_network(n_inputs, n_hidden, n_outputs);
    }
}

// Kernel CUDA para treinar e prever (este é o equivalente ao loop OpenMP)
__global__ void train_and_predict_kernel(
    Network* d_networks,
    float* d_train_data,
    int* d_train_offsets,
    int* d_train_sizes,
    float* d_test_data,
    int* d_test_offsets,
    int* d_test_sizes,
    float* d_expected_outputs,
    int* d_expected_offsets,
    int* d_expected_sizes,
    float l_rate,
    int n_epoch,
    int n_outputs,
    int n_inputs_per_row,
    float* d_scores
) {
    int fold_idx = threadIdx.x + blockIdx.x * blockDim.x; // Índice do fold (rede)
    if (fold_idx < gridDim.x * blockDim.x) { // Garante que não exceda o número de folds

        // Recuperar dados de treino e teste para este fold
        float* current_train_data = d_train_data + d_train_offsets[fold_idx];
        int num_train_rows = d_train_sizes[fold_idx];

        float* current_test_data = d_test_data + d_test_offsets[fold_idx];
        int num_test_rows = d_test_sizes[fold_idx];

        float* current_expected_outputs = d_expected_outputs + d_expected_offsets[fold_idx];
        int num_expected_outputs = d_expected_sizes[fold_idx];

        // Obter a rede neural para este fold
        Network& network = d_networks[fold_idx];

        // Lógica de treinamento (Simplificada, pois `train` do Network opera em std::vector)
        // Isso precisaria ser reescrito para operar em dados de GPU e ser CUDA_CALLABLE_MEMBER
        for (size_t e = 0; e < n_epoch; ++e) {
            float sum_error = 0;
            for (int r = 0; r < num_train_rows; ++r) {
                float* row_inputs = current_train_data + r * n_inputs_per_row;
                // A última coluna é o target, precisa ser extraída para 'expected'
                float actual_target = row_inputs[n_inputs_per_row - 1]; // Última coluna é o target

                // Array para as saídas da rede
                float outputs_buffer[n_outputs];
                network.forward_propagate(row_inputs, n_inputs_per_row - 1, outputs_buffer);

                float expected_one_hot[n_outputs];
                for (int i = 0; i < n_outputs; ++i) {
                    expected_one_hot[i] = 0.0f;
                }
                expected_one_hot[static_cast<int>(actual_target)] = 1.0f;

                for (size_t x = 0; x < n_outputs; x++) {
                    sum_error += static_cast<float>(pow((expected_one_hot[x] - outputs_buffer[x]), 2));
                }
                network.backward_propagate_error(expected_one_hot, n_outputs);
                network.update_weights(row_inputs, n_inputs_per_row - 1, l_rate);
            }
            // Não imprimir no kernel para evitar sobrecarga de I/O
            // Se VERBOSE > 0, pode-se usar atomicAdd para somar erros globais
        }


        // Lógica de previsão
        // std::vector<int> predicted(num_test_rows); // std::vector não é para device
        int predicted[num_test_rows]; // Usar array C-style
        for (int j = 0; j < num_test_rows; ++j) {
            float* row_test_inputs = current_test_data + j * n_inputs_per_row;
            predicted[j] = network.predict(row_test_inputs, n_inputs_per_row - 1);
        }

        // Calcular accuracy metric (precisa ser adaptada para C-style arrays)
        int correct = 0;
        for (int i = 0; i < num_test_rows; ++i) {
            if (predicted[i] == static_cast<int>(current_expected_outputs[i])) {
                correct++;
            }
        }
        d_scores[fold_idx] = static_cast<float>(correct * 100.0f / num_test_rows);
    }
}

// Kernel para liberar memória alocada por cada objeto Network na GPU
__global__ void destroy_network_kernel(Network* d_networks) {
    int idx = threadIdx.x;
    if (idx < gridDim.x * blockDim.x) {
        d_networks[idx].~Network(); // Chama o destrutor (que deve liberar m_layers)
    }
}

/*
* Load comma separated values from file and normalize the values
* (Permanece no Host)
*/
std::vector<std::vector<float>> load_csv_data(std::string filename) {
    const std::regex comma(",");
    std::ifstream csv_file(filename);
    std::vector<std::vector<float>> data;
    std::string line;
    std::vector<float> mins;
    std::vector<float> maxs;
    bool first = true;

    while (csv_file && std::getline(csv_file, line)) {
        std::vector<std::string> srow{ std::sregex_token_iterator(line.begin(), line.end(), comma, -1), std::sregex_token_iterator() };
        std::vector<float> row(srow.size());
        std::transform(srow.begin(), srow.end(), row.begin(), [](std::string const& val) {return std::stof(val); });
        
        if (first) {
            mins = row;
            maxs = row;
            first = false;
        } else {
            for (size_t t=0; t < row.size(); t++) {
                if (row[t] > maxs[t]) {
                    maxs[t] = row[t];
                } else if (row[t] < mins[t]) {
                    mins[t] = row[t];
                }
            }
        }
        data.push_back(row);
    }

    for (auto& vec : data) {
        for (size_t i = 0; i < vec.size()-1; i++) {
            vec[i] = (vec[i] - mins[i]) / (maxs[i] - mins[i]);
        }
    }
    return data;
}

// A função accuracy_metric pode ser removida se a lógica for incorporada ao kernel,
// ou mantida para cálculo no host se a previsão for transferida de volta.
// Por simplicidade, incorporei a lógica no kernel.
float accuracy_metric(std::vector<int> expect, std::vector<int> predict) {
    int correct = 0;
    for (size_t i = 0; i < predict.size(); i++) {
        if (predict[i] == expect[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct * 100.0f / predict.size());
}