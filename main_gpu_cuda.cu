#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <time.h>
#include <cuda_runtime.h> // Para funções CUDA

#define MAX_ROWS 40000
#define MAX_COLS 100
#define MAX_NEURONS 100 // Usado para buffers temporários de tamanho fixo

#ifndef VERBOSE
#define VERBOSE 0
#endif

// Qualificador para funções que podem ser chamadas tanto do host quanto do device
#define CUDA_CALLABLE_MEMBER __host__ __device__

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

// OPERAÇÕES NETWORK (agora com qualificadores CUDA)
CUDA_CALLABLE_MEMBER void initialize_network_device(Network *net, int n_inputs, int n_hidden, int n_outputs);
CUDA_CALLABLE_MEMBER void free_network_device(Network *net); // Para liberar memória na GPU

CUDA_CALLABLE_MEMBER void forward_propagate_device(Network *net, float *inputs);
CUDA_CALLABLE_MEMBER void backward_propagate_error_device(Network *net, float *expected);
CUDA_CALLABLE_MEMBER void update_weights_device(Network *net, float *inputs, float l_rate);
CUDA_CALLABLE_MEMBER int predict_device(Network *net, float *input);

// Função rand_weight() precisa ser adaptada para GPU
__device__ float rand_weight_device() {
    // Em um kernel real, você usaria uma biblioteca como cuRAND para gerar números aleatórios.
    // Para simplificar, esta é uma implementação básica (e não verdadeiramente aleatória)
    // para demonstrar que o rand() do host não funciona no device.
    // Lembre-se: rand() não é thread-safe nem aleatório o suficiente para GPUs.
    return (float)rand() / (float)RAND_MAX;
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
    float* d_accuracy_scores // Array para armazenar as pontuações de acurácia de cada fold
);

// Kernel para inicializar as redes na GPU
__global__ void initialize_networks_kernel(
    Network* d_networks,
    int n_folds,
    int n_inputs,
    int n_hidden,
    int n_outputs
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

    srand(time(NULL)); // Inicialização do PRNG para CPU

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
    size_t h_train_data_flat_size = 0;
    size_t h_train_offsets_size = 0;

    float* h_test_data_flat = NULL;
    int* h_test_offsets = NULL;
    int* h_test_rows_per_fold = NULL;
    size_t h_test_data_flat_size = 0;
    size_t h_test_offsets_size = 0;

    float* h_expected_outputs_flat = NULL;
    int* h_expected_offsets = NULL;
    int* h_expected_sizes = NULL;
    size_t h_expected_outputs_flat_size = 0;
    size_t h_expected_offsets_size = 0;

    // Pré-alocar buffers temporários no host para os offsets e tamanhos
    // para evitar realocações complexas dentro do loop.
    // O tamanho máximo necessário para cada um é n_folds.
    h_train_offsets = (int*)malloc(n_folds * sizeof(int));
    h_train_rows_per_fold = (int*)malloc(n_folds * sizeof(int));
    h_test_offsets = (int*)malloc(n_folds * sizeof(int));
    h_test_rows_per_fold = (int*)malloc(n_folds * sizeof(int));
    h_expected_offsets = (int*)malloc(n_folds * sizeof(int));
    h_expected_sizes = (int*)malloc(n_folds * sizeof(int));

    // Calcular o tamanho total necessário para os dados flat antes da alocação.
    // Isso é mais eficiente do que reallocar repetidamente.
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

    cudaMalloc((void**)&d_networks, n_folds * sizeof(Network));
    cudaMalloc((void**)&d_train_data_flat, total_train_data_elements * sizeof(float));
    cudaMalloc((void**)&d_train_offsets, n_folds * sizeof(int));
    cudaMalloc((void**)&d_train_rows_d, n_folds * sizeof(int));

    cudaMalloc((void**)&d_test_data_flat, total_test_data_elements * sizeof(float));
    cudaMalloc((void**)&d_test_offsets, n_folds * sizeof(int));
    cudaMalloc((void**)&d_test_rows_d, n_folds * sizeof(int));

    cudaMalloc((void**)&d_expected_outputs_flat, total_expected_elements * sizeof(float));
    cudaMalloc((void**)&d_expected_offsets, n_folds * sizeof(int));
    cudaMalloc((void**)&d_expected_sizes, n_folds * sizeof(int));

    cudaMalloc((void**)&d_accuracy_scores, n_folds * sizeof(float));


    // Copiar dados do host para o device
    cudaMemcpy(d_train_data_flat, h_train_data_flat, total_train_data_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_offsets, h_train_offsets, n_folds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_rows_d, h_train_rows_per_fold, n_folds * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_test_data_flat, h_test_data_flat, total_test_data_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_offsets, h_test_offsets, n_folds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_rows_d, h_test_rows_per_fold, n_folds * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_expected_outputs_flat, h_expected_outputs_flat, total_expected_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expected_offsets, h_expected_offsets, n_folds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expected_sizes, h_expected_sizes, n_folds * sizeof(int), cudaMemcpyHostToDevice);


    // Configuração do lançamento do kernel
    int blocks_per_grid = 1;
    int threads_per_block = n_folds;

    // Inicializar as redes na GPU
    initialize_networks_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_networks, n_folds, cols - 1, n_hidden, n_outputs
    );
    cudaDeviceSynchronize();

    // Lançar o kernel principal de treinamento e previsão
    train_and_predict_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_networks,
        d_train_data_flat, d_train_offsets, d_train_rows_d,
        d_test_data_flat, d_test_offsets, d_test_rows_d,
        d_expected_outputs_flat, d_expected_offsets, d_expected_sizes,
        l_rate, n_epoch, n_outputs, cols,
        d_accuracy_scores
    );
    cudaDeviceSynchronize();

    // Copiar os resultados de acurácia de volta para o host
    float* h_accuracy_scores_result = (float*)malloc(n_folds * sizeof(float));
    cudaMemcpy(h_accuracy_scores_result, d_accuracy_scores, n_folds * sizeof(float), cudaMemcpyDeviceToHost);

    // Calcular a acurácia média
    for (int i = 0; i < n_folds; ++i) {
        sum_accuracy += h_accuracy_scores_result[i];
    }

    // Liberar memória da GPU (Primeiro liberar a memória interna de cada Network)
    free_networks_kernel<<<blocks_per_grid, threads_per_block>>>(d_networks, n_folds);
    cudaDeviceSynchronize();

    cudaFree(d_networks);
    cudaFree(d_train_data_flat);
    cudaFree(d_train_offsets);
    cudaFree(d_train_rows_d);
    cudaFree(d_test_data_flat);
    cudaFree(d_test_offsets);
    cudaFree(d_test_rows_d);
    cudaFree(d_expected_outputs_flat);
    cudaFree(d_expected_offsets);
    cudaFree(d_expected_sizes);
    cudaFree(d_accuracy_scores);

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

    return sum_accuracy / (float)n_folds;
}

// Kernel CUDA para inicializar as redes na GPU
__global__ void initialize_networks_kernel(
    Network* d_networks,
    int n_folds,
    int n_inputs,
    int n_hidden,
    int n_outputs
) {
    int fold_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fold_idx < n_folds) {
        initialize_network_device(&d_networks[fold_idx], n_inputs, n_hidden, n_outputs);
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
    float* d_accuracy_scores
) {
    int fold_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fold_idx < gridDim.x * blockDim.x) {

        Network* current_net = &d_networks[fold_idx];

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
        // int predicted[current_test_rows]; // Não é ideal para alocação variável em kernel
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

CUDA_CALLABLE_MEMBER void initialize_network_device(Network *net, int n_inputs, int n_hidden, int n_outputs)
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
        Neuron* current_neuron = &net->layers[0].neurons[i]; // Ponteiro para o neurônio no device
        current_neuron->n_weights = n_inputs + 1;
        cudaMalloc((void**)&current_neuron->weights, (n_inputs + 1) * sizeof(float));
        for (int j = 0; j < n_inputs + 1; j++)
        {
            current_neuron->weights[j] = rand_weight_device(); // Usar rand_weight_device
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
            current_neuron->weights[j] = rand_weight_device();
        }
    }
}

CUDA_CALLABLE_MEMBER void forward_propagate_device(Network *net, float *inputs)
{
    // Usar um buffer de tamanho fixo na stack ou memória compartilhada se possível.
    // malloc/free no device são para casos mais complexos e podem introduzir overhead.
    // MAX_NEURONS deve ser definido com base no número máximo de neurônios que uma camada pode ter.
    float new_inputs_buffer[MAX_NEURONS]; // Alocação na stack do device (limite de 256KB por thread por padrão)

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

CUDA_CALLABLE_MEMBER void backward_propagate_error_device(Network *net, float *expected)
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

CUDA_CALLABLE_MEMBER void update_weights_device(Network *net, float *inputs, float l_rate)
{
    float new_in_buffer[MAX_NEURONS]; // Alocação na stack do device

    float *in = inputs;

    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *layer = &net->layers[l];

        // Copiar as entradas corretas para o buffer `new_in_buffer` para esta camada
        if (l == 0) { // Primeira camada usa as entradas originais
            // Aqui, in já aponta para as entradas originais (row)
            // Não precisamos copiar para new_in_buffer para usar as entradas da primeira camada
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
            // O número de pesos para as entradas (excluindo o bias) é neuron->n_weights - 1
            int num_input_weights = neuron->n_weights - 1;

            for (int j = 0; j < num_input_weights; j++)
            {
                neuron->weights[j] += l_rate * neuron->delta * in[j];
            }
            neuron->weights[neuron->n_weights - 1] += l_rate * neuron->delta; // Atualiza o peso do bias
        }
    }
}

CUDA_CALLABLE_MEMBER int predict_device(Network *net, float *input)
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

CUDA_CALLABLE_MEMBER void free_network_device(Network *net)
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