#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <time.h> // Continua sendo usada no host
#include <cuda_runtime.h> // Para funções CUDA
#include <curand_kernel.h> // Para geração de números aleatórios na GPU

#define MAX_ROWS 40000
#define MAX_COLS 100
#define MAX_NEURONS 100 // Usado para buffers temporários de tamanho fixo

#ifndef VERBOSE
#define VERBOSE 0
#endif

// Removido o CUDA_CALLABLE_MEMBER para essas funções, pois elas devem ser apenas __device__
// #define CUDA_CALLABLE_MEMBER __host__ __device__

// Adicionei um macro para verificar erros CUDA
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

typedef struct
{
    float *weights;
    float output, activation, delta;
    int n_weights;
} Neuron;

typedef struct
{
    Neuron *neurons;
    int n_neurons; // Adicionado para rastrear o número de neurônios na camada
} Layer;

typedef struct
{
    Layer *layers;
    int n_layers;
} Network;

// Utils (ainda no host, pois manipulam arquivos e I/O)
float **load_csv_data(const char *filename, int *rows, int *cols);
void print_data(float **data, int rows, int cols);
void normalize_data(float **data, int rows, int cols);
float accuracy_metric_host(int *expected, int *predicted, int size); // Renomeada para evitar conflito

// OPERAÇÕES NETWORK (agora com qualificadores CUDA ajustados)
// A função initialize_network_device precisa receber o estado do gerador
__device__ void initialize_network_device(Network *net, int n_inputs, int n_hidden, int n_outputs, curandState *rand_state);
__device__ void free_network_device(Network *net); // Para liberar memória na GPU

__device__ void forward_propagate_device(Network *net, float *inputs);
__device__ void backward_propagate_error_device(Network *net, float *expected);
__device__ void update_weights_device(Network *net, float *inputs, float l_rate);
__device__ int predict_device(Network *net, float *input);

// Função rand_weight() adaptada para usar cuRAND
__device__ float rand_weight_device(curandState *rand_state) {
    return curand_uniform(rand_state); // Gera um float entre 0.0f e 1.0f
}

// Função do host para lidar com a interação CPU-GPU
float evaluate_network_cuda(float **dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden);

// Kernel CUDA principal que substituirá o loop OpenMP
__global__ void train_and_predict_kernel(
    Network* d_networks, // Array de redes na GPU
    float* d_train_data_flat, // Dados de treino (flat)
    int* d_train_offsets,    // Offsets para os dados de treino de cada fold
    int* d_train_rows,       // Número de linhas de treino para cada fold
    float* d_test_data_flat, // Dados de teste (flat)
    int* d_test_offsets,     // Offsets para os dados de teste de cada fold
    int* d_test_rows,        // Número de linhas de teste para cada fold
    float* d_expected_outputs_flat, // Saídas esperadas (flat)
    int* d_expected_offsets, // Offsets para as saídas esperadas
    int* d_expected_sizes,   // Tamanhos das saídas esperadas
    float l_rate,
    int n_epoch,
    int n_outputs,
    int n_inputs_per_row, // Número total de entradas por linha (incluindo o target)
    float* d_accuracy_scores, // Array para armazenar as pontuações de acurácia de cada fold
    curandState *rand_states // Estado do gerador para cada thread
);

// Kernel para inicializar as redes na GPU
__global__ void initialize_networks_kernel(
    Network* d_networks,
    int n_folds,
    int n_inputs,
    int n_hidden,
    int n_outputs,
    curandState *rand_states, // Estado do gerador para cada thread
    unsigned long long seed_offset // Novo argumento para o seed
);

// Kernel para liberar a memória interna de cada Network na GPU
__global__ void free_networks_kernel(Network* d_networks, int n_folds);


int main(int argc, char **argv)
{
    if (argc < 7)
    {
        printf("Uso: %s <num_threads_ignored> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>\n", argv[0]);
        return 1;
    }

    char *dataset_file = argv[2];
    int n_folds = atoi(argv[3]);
    float l_rate = atof(argv[4]);
    int n_epoch = atoi(argv[5]);
    int n_hidden = atoi(argv[6]);

    srand(time(NULL)); // Inicialização do PRNG para CPU (ainda útil para load_csv_data, se aplicável)

    int rows, cols;
    float **dataset = load_csv_data(dataset_file, &rows, &cols);
    normalize_data(dataset, rows, cols);

#if VERBOSE > 0
    printf("Dataset carregado com %d linhas e %d colunas.\n", rows, cols);
    printf("Dividindo em %d folds.\n", n_folds);
#endif

    float mean_accuracy = evaluate_network_cuda(dataset, rows, cols, n_folds, l_rate, n_epoch, n_hidden);

    printf("Acurácia média: %.3f\n", mean_accuracy);

    // Liberar dataset carregado na CPU
    for (int i = 0; i < rows; ++i) {
        free(dataset[i]);
    }
    free(dataset);

    return 0;
}


float evaluate_network_cuda(float **dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden) {
    float sum_accuracy = 0.0;
    int fold_size = rows / n_folds;
    const int n_outputs = 2; // no nosso caso so tem as classes renda anual <=50K e >50K

    // Preparar os dados para a GPU - "flatten" os dados para um buffer contínuo
    // Usando arrays C-style em vez de std::vector
    float* h_train_data_flat = NULL;
    int* h_train_offsets = NULL;
    int* h_train_rows_per_fold = NULL;

    float* h_test_data_flat = NULL;
    int* h_test_offsets = NULL;
    int* h_test_rows_per_fold = NULL;

    float* h_expected_outputs_flat = NULL;
    int* h_expected_offsets = NULL;
    int* h_expected_sizes = NULL;

    // Pré-alocar buffers temporários no host para os offsets e tamanhos.
    h_train_offsets = (int*)malloc(n_folds * sizeof(int));
    h_train_rows_per_fold = (int*)malloc(n_folds * sizeof(int));
    h_test_offsets = (int*)malloc(n_folds * sizeof(int));
    h_test_rows_per_fold = (int*)malloc(n_folds * sizeof(int));
    h_expected_offsets = (int*)malloc(n_folds * sizeof(int));
    h_expected_sizes = (int*)malloc(n_folds * sizeof(int));
    cudaCheckError();

    // Calcular o tamanho total necessário para os dados flat antes da alocação.
    size_t total_train_data_elements = 0;
    size_t total_test_data_elements = 0;
    size_t total_expected_elements = 0;

    for (int i = 0; i < n_folds; i++) {
        total_train_data_elements += (rows - fold_size) * cols;
        total_test_data_elements += fold_size * cols;
        total_expected_elements += fold_size;
    }

    h_train_data_flat = (float*)malloc(total_train_data_elements * sizeof(float));
    h_test_data_flat = (float*)malloc(total_test_data_elements * sizeof(float));
    h_expected_outputs_flat = (float*)malloc(total_expected_elements * sizeof(float));
    cudaCheckError();

    // Popular os buffers flat e os arrays de offsets/tamanhos
    size_t current_train_flat_idx = 0;
    size_t current_test_flat_idx = 0;
    size_t current_expected_flat_idx = 0;

    for (int i = 0; i < n_folds; i++) {
        int start = i * fold_size;
        int end = start + fold_size;

        float **train_set_cpu = (float**)malloc((rows - fold_size) * sizeof(float *));
        float **test_set_cpu = (float**)malloc(fold_size * sizeof(float *));
        int *expected_cpu = (int*)malloc(fold_size * sizeof(int));
        cudaCheckError();
        int train_idx = 0, test_idx = 0;

        for (int r = 0; r < rows; r++) {
            if (r >= start && r < end) {
                test_set_cpu[test_idx] = (float*)malloc(cols * sizeof(float));
                memcpy(test_set_cpu[test_idx], dataset[r], cols * sizeof(float));
                expected_cpu[test_idx] = (int)test_set_cpu[test_idx][cols - 1];
                test_set_cpu[test_idx][cols - 1] = -1; // "Limpa" o target para o teste
                test_idx++;
            } else {
                train_set_cpu[train_idx] = dataset[r];
                train_idx++;
            }
        }

        h_train_offsets[i] = current_train_flat_idx;
        h_train_rows_per_fold[i] = (rows - fold_size);
        for (int r = 0; r < (rows - fold_size); ++r) {
            memcpy(h_train_data_flat + current_train_flat_idx, train_set_cpu[r], cols * sizeof(float));
            current_train_flat_idx += cols;
        }

        h_test_offsets[i] = current_test_flat_idx;
        h_test_rows_per_fold[i] = fold_size;
        for (int r = 0; r < fold_size; ++r) {
            memcpy(h_test_data_flat + current_test_flat_idx, test_set_cpu[r], cols * sizeof(float));
            current_test_flat_idx += cols;
        }

        h_expected_offsets[i] = current_expected_flat_idx;
        h_expected_sizes[i] = fold_size;
        for (int r = 0; r < fold_size; ++r) {
            h_expected_outputs_flat[current_expected_flat_idx++] = (float)expected_cpu[r];
        }

        for (int j = 0; j < fold_size; j++) free(test_set_cpu[j]);
        free(test_set_cpu);
        free(train_set_cpu);
        free(expected_cpu);
    }

    // Alocar memória na GPU
    Network* d_networks;
    float* d_train_data_flat;
    int* d_train_offsets;
    int* d_train_rows_d;

    float* d_test_data_flat;
    int* d_test_offsets;
    int* d_test_rows_d;

    float* d_expected_outputs_flat;
    int* d_expected_offsets;
    int* d_expected_sizes;

    float* d_accuracy_scores;
    curandState *d_rand_states; // Estados dos geradores de números aleatórios na GPU

    cudaMalloc((void**)&d_networks, n_folds * sizeof(Network)); cudaCheckError();
    cudaMalloc((void**)&d_train_data_flat, total_train_data_elements * sizeof(float)); cudaCheckError();
    cudaMalloc((void**)&d_train_offsets, n_folds * sizeof(int)); cudaCheckError();
    cudaMalloc((void**)&d_train_rows_d, n_folds * sizeof(int)); cudaCheckError();

    cudaMalloc((void**)&d_test_data_flat, total_test_data_elements * sizeof(float)); cudaCheckError();
    cudaMalloc((void**)&d_test_offsets, n_folds * sizeof(int)); cudaCheckError();
    cudaMalloc((void**)&d_test_rows_d, n_folds * sizeof(int)); cudaCheckError();

    cudaMalloc((void**)&d_expected_outputs_flat, total_expected_elements * sizeof(float)); cudaCheckError();
    cudaMalloc((void**)&d_expected_offsets, n_folds * sizeof(int)); cudaCheckError();
    cudaMalloc((void**)&d_expected_sizes, n_folds * sizeof(int)); cudaCheckError();

    cudaMalloc((void**)&d_accuracy_scores, n_folds * sizeof(float)); cudaCheckError();
    cudaMalloc((void**)&d_rand_states, n_folds * sizeof(curandState)); cudaCheckError();


    // Copiar dados do host para o device
    cudaMemcpy(d_train_data_flat, h_train_data_flat, total_train_data_elements * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_train_offsets, h_train_offsets, n_folds * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_train_rows_d, h_train_rows_per_fold, n_folds * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError();

    cudaMemcpy(d_test_data_flat, h_test_data_flat, total_test_data_elements * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_test_offsets, h_test_offsets, n_folds * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_test_rows_d, h_test_rows_per_fold, n_folds * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError();

    cudaMemcpy(d_expected_outputs_flat, h_expected_outputs_flat, total_expected_elements * sizeof(float), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_expected_offsets, h_expected_offsets, n_folds * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_expected_sizes, h_expected_sizes, n_folds * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError();


    // Configuração do lançamento do kernel
    int blocks_per_grid = 1;
    int threads_per_block = n_folds;

    // Gerar um seed base no host para os geradores na GPU
    unsigned long long host_seed = time(NULL);

    // Inicializar as redes na GPU
    initialize_networks_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_networks, n_folds, cols - 1, n_hidden, n_outputs, d_rand_states, host_seed
    );
    cudaDeviceSynchronize(); cudaCheckError(); // Espera a inicialização terminar

    // Lançar o kernel principal de treinamento e previsão
    train_and_predict_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_networks,
        d_train_data_flat, d_train_offsets, d_train_rows_d,
        d_test_data_flat, d_test_offsets, d_test_rows_d,
        d_expected_outputs_flat, d_expected_offsets, d_expected_sizes,
        l_rate, n_epoch, n_outputs, cols,
        d_accuracy_scores,
        d_rand_states // Passa o array de estados para o kernel principal
    );
    cudaDeviceSynchronize(); cudaCheckError(); // Espera o kernel terminar

    // Copiar os resultados de acurácia de volta para o host
    float* h_accuracy_scores_result = (float*)malloc(n_folds * sizeof(float)); cudaCheckError();
    cudaMemcpy(h_accuracy_scores_result, d_accuracy_scores, n_folds * sizeof(float), cudaMemcpyDeviceToHost); cudaCheckError();

    // Calcular a acurácia média
    for (int i = 0; i < n_folds; ++i) {
        sum_accuracy += h_accuracy_scores_result[i];
    }

    // Liberar memória da GPU (Primeiro liberar a memória interna de cada Network)
    free_networks_kernel<<<blocks_per_grid, threads_per_block>>>(d_networks, n_folds);
    cudaDeviceSynchronize(); cudaCheckError(); // Espera a liberação interna terminar

    cudaFree(d_networks); cudaCheckError();
    cudaFree(d_train_data_flat); cudaCheckError();
    cudaFree(d_train_offsets); cudaCheckError();
    cudaFree(d_train_rows_d); cudaCheckError();
    cudaFree(d_test_data_flat); cudaCheckError();
    cudaFree(d_test_offsets); cudaCheckError();
    cudaFree(d_test_rows_d); cudaCheckError();
    cudaFree(d_expected_outputs_flat); cudaCheckError();
    cudaFree(d_expected_offsets); cudaCheckError();
    cudaFree(d_expected_sizes); cudaCheckError();
    cudaFree(d_accuracy_scores); cudaCheckError();
    cudaFree(d_rand_states); cudaCheckError();

    // Liberar a memória do host
    free(h_train_data_flat);
    free(h_train_offsets);
    free(h_train_rows_per_fold);
    free(h_test_data_flat);
    free(h_test_offsets);
    free(h_test_rows_per_fold);
    free(h_expected_outputs_flat);
    free(h_expected_offsets);
    free(h_expected_sizes);
    free(h_accuracy_scores_result);

    free_network_devide(d_networks);

    return sum_accuracy / (float)n_folds;
}

// Kernel CUDA para inicializar as redes na GPU
__global__ void initialize_networks_kernel(
    Network* d_networks,
    int n_folds,
    int n_inputs,
    int n_hidden,
    int n_outputs,
    curandState *rand_states,
    unsigned long long seed_offset // Recebe o seed base do host
) {
    int fold_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fold_idx < n_folds) {
        // Inicializa o estado do gerador de números aleatórios para esta thread
        // O seed é baseado no seed_offset do host e no ID da thread
        curand_init(seed_offset + fold_idx, 0, 0, &rand_states[fold_idx]);

        initialize_network_device(&d_networks[fold_idx], n_inputs, n_hidden, n_outputs, &rand_states[fold_idx]);
    }
}

// Kernel CUDA principal
__global__ void train_and_predict_kernel(
    Network* d_networks,
    float* d_train_data_flat,
    int* d_train_offsets,
    int* d_train_rows,
    float* d_test_data_flat,
    int* d_test_offsets,
    int* d_test_rows,
    float* d_expected_outputs_flat,
    int* d_expected_offsets,
    int* d_expected_sizes,
    float l_rate,
    int n_epoch,
    int n_outputs,
    int n_inputs_per_row,
    float* d_accuracy_scores,
    curandState *rand_states // Recebe o array de estados
) {
    int fold_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fold_idx < gridDim.x * blockDim.x) {

        Network* current_net = &d_networks[fold_idx];
        // curandState *current_rand_state = &rand_states[fold_idx]; // Não é mais usado diretamente aqui, mas em funções chamadas

        // Obter os dados de treino para este fold
        float* train_data_for_fold = d_train_data_flat + d_train_offsets[fold_idx];
        int current_train_rows = d_train_rows[fold_idx];

        // Obter os dados de teste para este fold
        float* test_data_for_fold = d_test_data_flat + d_test_offsets[fold_idx];
        int current_test_rows = d_test_rows[fold_idx];

        // Obter as saídas esperadas para este fold
        float* expected_outputs_for_fold = d_expected_outputs_flat + d_expected_offsets[fold_idx];
        // int current_expected_size = d_expected_sizes[fold_idx]; // Não usado diretamente aqui

        // Lógica de treinamento (loop de épocas e linhas de treino)
        float expected_one_hot[MAX_NEURONS]; // Tamanho máximo de neurônios para um one-hot
        for (int epoch = 0; epoch < n_epoch; epoch++) {
            float sum_error = 0.0f;
            for (int r = 0; r < current_train_rows; r++) {
                float* row = train_data_for_fold + r * n_inputs_per_row;
                forward_propagate_device(current_net, row);

                for (int i = 0; i < n_outputs; i++) {
                    expected_one_hot[i] = 0.0f;
                }
                expected_one_hot[(int)row[n_inputs_per_row - 1]] = 1.0f; // Última coluna é o target

                Layer *output_layer = &current_net->layers[current_net->n_layers - 1];
                for (int i = 0; i < n_outputs; i++) {
                    float diff = expected_one_hot[i] - output_layer->neurons[i].output;
                    sum_error += diff * diff;
                }

                backward_propagate_error_device(current_net, expected_one_hot);
                update_weights_device(current_net, row, l_rate);
            }
            #if VERBOSE > 0
                // Printf em kernels é limitado e pode afetar a performance. Use com moderação.
                // printf("[%d] epoch=%d, error=%.6f\n", fold_idx, epoch, sum_error);
            #endif
        }

        // Lógica de previsão
        int correct_predictions = 0;
        for (int j = 0; j < current_test_rows; j++) {
            float* test_row = test_data_for_fold + j * n_inputs_per_row;
            int predicted_class = predict_device(current_net, test_row);
            if (predicted_class == (int)expected_outputs_for_fold[j]) {
                correct_predictions++;
            }
        }
        d_accuracy_scores[fold_idx] = (float)correct_predictions * 100.0f / current_test_rows;
    }
}

// Kernel para liberar a memória alocada internamente por cada objeto Network na GPU
__global__ void free_networks_kernel(Network* d_networks, int n_folds) {
    int fold_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fold_idx < n_folds) {
        free_network_device(&d_networks[fold_idx]);
    }
}


// --------------------------------------------
// Funções Network (Adaptadas para CUDA)
// --------------------------------------------

__device__ void initialize_network_device(Network *net, int n_inputs, int n_hidden, int n_outputs, curandState *rand_state)
{
    // REDE NEURAL COM 2 CAMADAS
    net->n_layers = 2;
    // Usar cudaMalloc para alocar no device
    cudaMalloc((void**)&net->layers, 2 * sizeof(Layer));

    // Camada Oculta
    net->layers[0].n_neurons = n_hidden;
    cudaMalloc((void**)&net->layers[0].neurons, n_hidden * sizeof(Neuron));
    for (int i = 0; i < n_hidden; i++)
    {
        // Precisamos garantir que a alocação de neurônios seja acessível por ponteiro de device
        // Se initialize_network_device é chamada no device, essas alocações são no device.
        Neuron* current_neuron = &net->layers[0].neurons[i]; // Ponteiro para o neurônio no device
        current_neuron->n_weights = n_inputs + 1;
        cudaMalloc((void**)&current_neuron->weights, (n_inputs + 1) * sizeof(float));
        for (int j = 0; j < n_inputs + 1; j++)
        {
            current_neuron->weights[j] = rand_weight_device(rand_state); // Passa o estado do gerador
        }
    }

    // Camada de Saída
    net->layers[1].n_neurons = n_outputs;
    cudaMalloc((void**)&net->layers[1].neurons, n_outputs * sizeof(Neuron));
    for (int i = 0; i < n_outputs; i++)
    {
        Neuron* current_neuron = &net->layers[1].neurons[i];
        current_neuron->n_weights = n_hidden + 1;
        cudaMalloc((void**)&current_neuron->weights, (n_hidden + 1) * sizeof(float));
        for (int j = 0; j < n_hidden + 1; j++)
        {
            current_neuron->weights[j] = rand_weight_device(rand_state); // Passa o estado do gerador
        }
    }
}

__device__ void forward_propagate_device(Network *net, float *inputs)
{
    // É seguro usar um array estático de tamanho fixo em funções __device__
    // desde que o tamanho seja conhecido em tempo de compilação e pequeno o suficiente
    // para a stack da thread (MAX_NEURONS).
    float new_inputs_buffer[MAX_NEURONS];

    float *in = inputs;

    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *layer = &net->layers[l];

        for (int n = 0; n < layer->n_neurons; n++)
        {
            Neuron *neuron = &layer->neurons[n];

            neuron->activation = neuron->weights[neuron->n_weights - 1]; // Bias
            for (int j = 0; j < neuron->n_weights - 1; j++)
            {
                neuron->activation += neuron->weights[j] * in[j];
            }
            neuron->output = 1.0f / (1.0f + expf(-neuron->activation));
            new_inputs_buffer[n] = neuron->output;
        }

        // Se não for a última camada, atualize `in` para apontar para as saídas da camada atual
        if (l < net->n_layers - 1) {
             in = new_inputs_buffer;
        }
    }
}

__device__ void backward_propagate_error_device(Network *net, float *expected)
{
    for (int l = net->n_layers - 1; l >= 0; l--)
    {
        Layer *layer = &net->layers[l];
        for (int n = 0; n < layer->n_neurons; n++)
        {
            Neuron *neuron = &layer->neurons[n];
            float error = 0.0f;

            if (l == net->n_layers - 1)
            {
                error = expected[n] - neuron->output;
            }
            else
            {
                Layer *next_layer = &net->layers[l + 1];
                for (int k = 0; k < next_layer->n_neurons; k++)
                {
                    error += next_layer->neurons[k].weights[n] * next_layer->neurons[k].delta;
                }
            }
            neuron->delta = error * neuron->output * (1.0f - neuron->output);
        }
    }
}

__device__ void update_weights_device(Network *net, float *inputs, float l_rate)
{
    float new_in_buffer[MAX_NEURONS];

    float *in = inputs;

    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *layer = &net->layers[l];

        // Copiar as entradas corretas para o buffer `new_in_buffer` para esta camada
        if (l == 0) { // Primeira camada usa as entradas originais
            // Aqui, in já aponta para as entradas originais (row)
        } else { // Camadas subsequentes usam as saídas da camada anterior
            Layer *prev_layer = &net->layers[l-1];
            for (int i = 0; i < prev_layer->n_neurons; i++)
            {
                new_in_buffer[i] = prev_layer->neurons[i].output;
            }
            in = new_in_buffer; // Atualiza 'in' para as saídas da camada anterior
        }

        for (int n = 0; n < layer->n_neurons; n++)
        {
            Neuron *neuron = &layer->neurons[n];
            int num_input_weights = neuron->n_weights - 1;

            for (int j = 0; j < num_input_weights; j++)
            {
                neuron->weights[j] += l_rate * neuron->delta * in[j];
            }
            neuron->weights[neuron->n_weights - 1] += l_rate * neuron->delta; // Atualiza o peso do bias
        }
    }
}

__device__ int predict_device(Network *net, float *input)
{
    forward_propagate_device(net, input);
    Layer *output_layer = &net->layers[net->n_layers - 1];

    int max_i = 0;
    float max_v = output_layer->neurons[0].output;
    for (int i = 1; i < output_layer->n_neurons; i++)
    {
        if (output_layer->neurons[i].output > max_v)
        {
            max_v = output_layer->neurons[i].output;
            max_i = i;
        }
    }
    return max_i;
}

void free_network_device(Network *net)
{
    if (net->layers) {
        for (int l = 0; l < net->n_layers; l++)
        {
            if (net->layers[l].neurons) {
                for (int n = 0; n < net->layers[l].n_neurons; n++)
                {
                    if (net->layers[l].neurons[n].weights) {
                        cudaFree(net->layers[l].neurons[n].weights);
                    }
                }
                cudaFree(net->layers[l].neurons);
            }
        }
        cudaFree(net->layers);
    }
}


// --------------------------------------------
// Funções utilitárias (host)
// --------------------------------------------

// carrega os dados do CSV como uma matriz de floats
float **load_csv_data(const char *filename, int *rows, int *cols)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Erro ao abrir CSV");
        exit(1);
    }

    float **data = (float**)malloc(MAX_ROWS * sizeof(float *));
    char line[4096];
    int r = 0, c = 0;

    while (fgets(line, sizeof(line), fp) && r < MAX_ROWS)
    {
        data[r] = (float*)malloc(MAX_COLS * sizeof(float));
        c = 0;
        char *token = strtok(line, ",");
        while (token)
        {
            data[r][c++] = atof(token);
            token = strtok(NULL, ",");
        }
        r++;
    }

    fclose(fp);
    *rows = r;
    *cols = c;
    return data;
}

// normalizar os dados para ficarem entre 0, 1
void normalize_data(float **data, int rows, int cols)
{
    float *mins = (float*)malloc(cols * sizeof(float));
    float *maxs = (float*)malloc(cols * sizeof(float));
    for (int j = 0; j < cols - 1; j++)
    {
        mins[j] = data[0][j];
        maxs[j] = data[0][j];
        for (int i = 1; i < rows; i++)
        {
            if (data[i][j] < mins[j])
                mins[j] = data[i][j];
            if (data[i][j] > maxs[j])
                maxs[j] = data[i][j];
        }
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
        }
        // No caso de maxs[j] == mins[j], para evitar divisão por zero,
        // o valor já deve ser 0 (data[i][j] - mins[j] será 0).
        // Não é estritamente necessário um if, mas pode ser boa prática.
    }
    free(mins);
    free(maxs);
}

void print_data(float **data, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", data[i][j]);
        }
        printf("\n");
    }
}

float accuracy_metric_host(int *expected, int *predicted, int size)
{
    int correct = 0;
    for (int i = 0; i < size; i++)
    {
        if (expected[i] == predicted[i])
            correct++;
    }
    return 100.0f * correct / size;
}