#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <assert.h>

#define MAX_ROWS 40000
#define COLS 15
#define MAX_NEURONS COLS

#ifndef VERBOSE
#define VERBOSE 0
#endif

#define CUDA_CHECK(err)                                                                                \
    {                                                                                                  \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

#define IDX_LAYER(fold, layer) ((fold) * n_layers + (layer))
#define IDX_NEURON(fold, layer, neuron) ((fold) * n_layers * MAX_NEURONS + (layer) * MAX_NEURONS + (neuron))
#define IDX_WEIGHT(fold, layer, neuron, weight) ((fold) * n_layers * MAX_NEURONS * MAX_NEURONS + (layer) * MAX_NEURONS * MAX_NEURONS + (neuron) * MAX_NEURONS + (weight))

#define TOTAL_LAYERS (n_folds * n_layers)
#define TOTAL_NEURONS (n_folds * n_layers * MAX_NEURONS)
#define TOTAL_WEIGHTS (n_folds * n_layers * MAX_NEURONS * MAX_NEURONS)

// utils
float *load_csv_data(const char *filename, int *rows, int *cols);
void normalize_data(float *data, int rows, int cols);
float accuracy_metric(int *expected, int *predicted, int size);
float evaluate_network(float *dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden, int num_blocks);
float rand_weight() { return (float)rand() / (float)RAND_MAX; }

// aproximação de expf(x) válida para -10 <= x <= 10
__device__ float fast_exp(float x)
{
    x = 1.0f + x / 256.0f;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}
__device__ float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }
__device__ float sigmoid_derivative(float output) { return output * (1.0f - output); }

// OPERAÇÕES NETWORK (Device)
__device__ void forward_propagate_device(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer);
__device__ void backward_propagate_error_device(int fold, int n_layers, int *n_neurons, float *weights, float *outputs, float *deltas, float *expected);
__device__ void update_weights_device(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *deltas, float *inputs, float l_rate);

// OPERAÇÕES NETWORK (Host)
void initialize_network(int fold_id, int n_inputs, int n_hidden, int n_outputs, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *activations, float *deltas);
int predict(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer);

__global__ void train_networks_kernel(
    float *train_sets, int *n_neurons, int *n_weights, float *weights,
    float *outputs, float *activations, float *deltas, float *layer_buffer,
    int rows, int cols, int fold_size, int n_epoch, float l_rate, int n_layers, int n_outputs,
    int n_folds)
{
    // Cada bloco processa um ou mais folds (grid-stride loop)
    for (int i = blockIdx.x; i < n_folds; i += gridDim.x)
    {

        float expected_output[MAX_NEURONS];

        for (int epoch = 0; epoch < n_epoch; epoch++)
        {
            float sum_error = 0.0f;
            for (int r = 0; r < rows - fold_size; r++)
            {

                float *row = &train_sets[(i * (rows - fold_size) + r) * cols];

                forward_propagate_device(i, n_layers, n_neurons, n_weights, weights, activations, outputs, row, layer_buffer);

                for (int k = 0; k < n_outputs; k++)
                    expected_output[k] = 0.0f;
                expected_output[(int)row[cols - 1]] = 1.0f;

                int last_layer = n_layers - 1;
                for (int j = 0; j < n_outputs; j++)
                {
                    float diff = expected_output[j] - outputs[IDX_NEURON(i, last_layer, j)];
                    sum_error += diff * diff;
                }

                backward_propagate_error_device(i, n_layers, n_neurons, weights, outputs, deltas, expected_output);
                update_weights_device(i, n_layers, n_neurons, n_weights, weights, outputs, deltas, row, l_rate);
            }
#if VERBOSE > 0
            printf("[block %.3d>fold %.3d]\tepoch=%.3d, l_rate=%.3f, error=%.6f\n", blockIdx.x, i, epoch, l_rate, sum_error);
#endif
        }
    }
}

// utils
float *load_csv_data(const char *filename, int *rows, int *cols)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Erro ao abrir CSV");
        exit(1);
    }

    float *data = (float *)malloc(MAX_ROWS * COLS * sizeof(float));
    char line[4096];
    int r = 0, c = 0;

    while (fgets(line, sizeof(line), fp) && r < MAX_ROWS)
    {
        char *token = strtok(line, ",");
        c = 0;
        while (token)
        {
            data[r * COLS + c++] = atof(token);
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
void normalize_data(float *data, int rows, int cols)
{
    float *mins = (float *)malloc(cols * sizeof(float));
    float *maxs = (float *)malloc(cols * sizeof(float));
    for (int j = 0; j < cols - 1; j++)
    {
        mins[j] = data[j];
        maxs[j] = data[j];
        for (int i = 1; i < rows; i++)
        {
            float val = data[i * cols + j];
            if (val < mins[j])
                mins[j] = val;
            if (val > maxs[j])
                maxs[j] = val;
        }
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            data[i * cols + j] = (data[i * cols + j] - mins[j]) / (maxs[j] - mins[j]);
        }
    }
    free(mins);
    free(maxs);
}

float accuracy_metric(int *expected, int *predicted, int size)
{
    int correct = 0;
    for (int i = 0; i < size; i++)
    {
        if (expected[i] == predicted[i])
            correct++;
    }
    return 100.0f * correct / size;
}

float evaluate_network(float *dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden, int num_blocks)
{
    /*
        INICIALIZAÇÃO DOS DADOS E NETWORK
    */
    float sum_accuracy = 0.0f;
    int fold_size = rows / n_folds;
    printf("Fold size: %d\n", fold_size);
    const int n_outputs = 2; // no nosso caso so tem as classes renda anual <=50K e >50K

    // NETWORK LINEAR
    int n_layers = 2;
    int *h_n_neurons;     // [net][layer]
    float *h_weights;     // [net][layer][neuron][weight]
    float *h_outputs;     // [net][layer][neuron]
    float *h_activations; // [net][layer][neuron]
    float *h_deltas;      // [net][layer][neuron]
    int *h_n_weights;     // [net][layer][neuron]
    float *h_layer_buffer;
    float *h_train_sets;
    float *h_test_sets;
    int *expected_labels;
    int *predicted_labels;

    // Alocação de memória no Host (CPU)
    h_n_neurons = (int *)malloc(TOTAL_LAYERS * sizeof(int));
    h_weights = (float *)malloc(TOTAL_WEIGHTS * sizeof(float));
    h_outputs = (float *)malloc(TOTAL_NEURONS * sizeof(float));
    h_activations = (float *)malloc(TOTAL_NEURONS * sizeof(float));
    h_deltas = (float *)malloc(TOTAL_NEURONS * sizeof(float));
    h_n_weights = (int *)malloc(TOTAL_NEURONS * sizeof(int));
    h_layer_buffer = (float *)malloc(TOTAL_NEURONS * sizeof(float));
    h_train_sets = (float *)malloc(n_folds * (rows - fold_size) * cols * sizeof(float));
    h_test_sets = (float *)malloc(n_folds * fold_size * cols * sizeof(float));
    expected_labels = (int *)malloc(n_folds * fold_size * sizeof(int));
    predicted_labels = (int *)malloc(n_folds * fold_size * sizeof(int));

    // Preparação dos dados de treino/teste e inicialização das redes
    for (int i = 0; i < n_folds; i++)
    {
        int start = i * fold_size;
        int end = start + fold_size;
        int train_idx = 0, test_idx = 0;
        for (int r = 0; r < rows; r++)
        {
            if (r >= start && r < end)
            {
                memcpy(&h_test_sets[(i * fold_size + test_idx) * cols], &dataset[r * cols], cols * sizeof(float));
                expected_labels[i * fold_size + test_idx] = (int)h_test_sets[(i * fold_size + test_idx) * cols + (cols - 1)];
                test_idx++;
            }
            else
            {
                memcpy(&h_train_sets[(i * (rows - fold_size) + train_idx) * cols], &dataset[r * cols], cols * sizeof(float));
                train_idx++;
            }
        }
        initialize_network(i, cols - 1, n_hidden, n_outputs, n_layers, h_n_neurons, h_n_weights, h_weights, h_outputs, h_activations, h_deltas);
    }

    /*
        TREINO DA NETWORK
        O paralelismo ocorre aqui. Alocamos memória na GPU, copiamos os dados
        e lançamos o kernel CUDA para treinar todas as redes em paralelo.
    */
    int *d_n_neurons;
    float *d_weights;
    float *d_outputs;
    float *d_activations;
    float *d_deltas;
    int *d_n_weights;
    float *d_layer_buffer;
    float *d_train_sets;

    // Alocação de memória no Device (GPU)
    CUDA_CHECK(cudaMalloc(&d_n_neurons, TOTAL_LAYERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, TOTAL_WEIGHTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, TOTAL_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_activations, TOTAL_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_deltas, TOTAL_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_n_weights, TOTAL_NEURONS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_layer_buffer, TOTAL_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_train_sets, n_folds * (rows - fold_size) * cols * sizeof(float)));

    // Cópia de dados do Host para o Device
    CUDA_CHECK(cudaMemcpy(d_n_neurons, h_n_neurons, TOTAL_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, TOTAL_WEIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outputs, h_outputs, TOTAL_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_activations, h_activations, TOTAL_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_deltas, h_deltas, TOTAL_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_weights, h_n_weights, TOTAL_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_sets, h_train_sets, n_folds * (rows - fold_size) * cols * sizeof(float), cudaMemcpyHostToDevice));

    // Lançamento do Kernel CUDA
    dim3 grid(num_blocks, 1, 1);
    dim3 block(1, 1, 1);
    printf("blocks: %d | folds: %d\n", num_blocks, n_folds);

    train_networks_kernel<<<grid, block>>>(
        d_train_sets, d_n_neurons, d_n_weights, d_weights,
        d_outputs, d_activations, d_deltas, d_layer_buffer,
        rows, cols, fold_size, n_epoch, l_rate, n_layers, n_outputs,
        n_folds);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel finished.\n");

    // Cópia dos resultados (pesos treinados) de volta para o Host
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, TOTAL_WEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outputs, d_outputs, TOTAL_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_activations, d_activations, TOTAL_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));

    /*
        TESTE DAS NETWORKS E LIBERAÇÃO DOS RECURSOS
    */
    for (int i = 0; i < n_folds; i++)
    {
        for (int j = 0; j < fold_size; j++)
        {
            predicted_labels[i * fold_size + j] = predict(i, n_layers, h_n_neurons, h_n_weights, h_weights, h_activations, h_outputs, &h_test_sets[(i * fold_size + j) * cols], h_layer_buffer);
        }
        sum_accuracy += accuracy_metric(&expected_labels[i * fold_size], &predicted_labels[i * fold_size], fold_size);
    }

    // Liberação de memória
    cudaFree(d_n_neurons);
    cudaFree(d_weights);
    cudaFree(d_outputs);
    cudaFree(d_activations);
    cudaFree(d_deltas);
    cudaFree(d_n_weights);
    cudaFree(d_layer_buffer);
    cudaFree(d_train_sets);
    free(h_n_neurons);
    free(h_weights);
    free(h_outputs);
    free(h_activations);
    free(h_deltas);
    free(h_n_weights);
    free(h_layer_buffer);
    free(h_train_sets);
    free(h_test_sets);
    free(expected_labels);
    free(predicted_labels);

    return sum_accuracy / (float)n_folds;
}

__device__ void forward_propagate_device(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
{
    float *in = inputs; // inicia com os inputs
    for (int l = 0; l < n_layers; l++)
    {
        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            int nw = n_weights[IDX_NEURON(fold, l, n)];

            // começa a ativacao com o bias (ultimo peso)
            activations[IDX_NEURON(fold, l, n)] = weights[IDX_WEIGHT(fold, l, n, nw - 1)];
            for (int j = 0; j < nw - 1; j++)
            {
                activations[IDX_NEURON(fold, l, n)] += weights[IDX_WEIGHT(fold, l, n, j)] * in[j];
            }

            // aplica a funcao de ativacao sigmoide
            float output = sigmoid(activations[IDX_NEURON(fold, l, n)]);
            outputs[IDX_NEURON(fold, l, n)] = output;
            layer_buffer[IDX_NEURON(fold, l, n)] = output; // new_inputs
        }
        in = &layer_buffer[IDX_NEURON(fold, l, 0)]; // atualiza os inputs para a proxima camada
    }
}

__device__ void backward_propagate_error_device(int fold, int n_layers, int *n_neurons, float *weights, float *outputs, float *deltas, float *expected)
{
    for (int l = n_layers - 1; l >= 0; l--) // começa da ultima camada
    {
        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            float error = 0.0f;

            if (l == n_layers - 1) // se for a camada de saida, erro é a diferenca entre o esperado e o atual
            {
                error = expected[n] - outputs[IDX_NEURON(fold, l, n)];
            }
            else // se for uma camada escondida, soma os erros ponderados da proxima camada
            {
                int next_layer = l + 1;
                for (int k = 0; k < n_neurons[IDX_LAYER(fold, next_layer)]; k++)
                {
                    error += weights[IDX_WEIGHT(fold, next_layer, k, n)] * deltas[IDX_NEURON(fold, next_layer, k)];
                }
            }
            // delta = erro * derivada da funcao de ativacao
            deltas[IDX_NEURON(fold, l, n)] = error * sigmoid_derivative(outputs[IDX_NEURON(fold, l, n)]);
        }
    }
}

__device__ void update_weights_device(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *deltas, float *inputs, float l_rate)
{
    for (int l = 0; l < n_layers; l++)
    {
        float *current_in;
        int n_in;
        if (l == 0)
        {
            current_in = inputs;
            n_in = COLS - 1;
        }
        else
        {
            current_in = &outputs[IDX_NEURON(fold, l - 1, 0)];
            n_in = n_neurons[IDX_LAYER(fold, l - 1)];
        }

        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            for (int j = 0; j < n_in; j++) // atualiza os pesos com o gradiente descendente
            {
                weights[IDX_WEIGHT(fold, l, n, j)] += l_rate * deltas[IDX_NEURON(fold, l, n)] * current_in[j];
            }
            weights[IDX_WEIGHT(fold, l, n, n_in)] += l_rate * deltas[IDX_NEURON(fold, l, n)]; // atualiza o peso do bias
        }
    }
}

void forward_propagate_host(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
{
    float *in = inputs;
    for (int l = 0; l < n_layers; l++)
    {
        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            int nw = n_weights[IDX_NEURON(fold, l, n)];
            activations[IDX_NEURON(fold, l, n)] = weights[IDX_WEIGHT(fold, l, n, nw - 1)];
            for (int j = 0; j < nw - 1; j++)
            {
                activations[IDX_NEURON(fold, l, n)] += weights[IDX_WEIGHT(fold, l, n, j)] * in[j];
            }
            outputs[IDX_NEURON(fold, l, n)] = 1.0f / (1.0f + expf(-activations[IDX_NEURON(fold, l, n)]));
            layer_buffer[IDX_NEURON(fold, l, n)] = outputs[IDX_NEURON(fold, l, n)];
        }
        in = &layer_buffer[IDX_NEURON(fold, l, 0)];
    }
}

void initialize_network(int fold_id, int n_inputs, int n_hidden, int n_outputs, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *activations, float *deltas)
{
    // REDE NEURAL COM 2 CAMADAS
    n_neurons[IDX_LAYER(fold_id, 0)] = n_hidden; // primeira camada tem n_hidden neurônios
    for (int n = 0; n < n_hidden; n++)
    {
        n_weights[IDX_NEURON(fold_id, 0, n)] = n_inputs + 1;
        for (int w = 0; w < n_inputs + 1; w++)
        {
            weights[IDX_WEIGHT(fold_id, 0, n, w)] = rand_weight();
        }
    }

    n_neurons[IDX_LAYER(fold_id, 1)] = n_outputs; // segunda camada tem n_outputs neurônios
    for (int n = 0; n < n_outputs; n++)
    {
        n_weights[IDX_NEURON(fold_id, 1, n)] = n_hidden + 1;
        for (int w = 0; w < n_hidden + 1; w++)
        {
            weights[IDX_WEIGHT(fold_id, 1, n, w)] = rand_weight();
        }
    }
}

int predict(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
{
    forward_propagate_host(fold, n_layers, n_neurons, n_weights, weights, activations, outputs, inputs, layer_buffer);

    int n_outputs = n_neurons[IDX_LAYER(fold, n_layers - 1)];
    int max_i = 0;
    float max_v = outputs[IDX_NEURON(fold, n_layers - 1, 0)];

    for (int i = 1; i < n_outputs; i++)
    {
        float val = outputs[IDX_NEURON(fold, n_layers - 1, i)];
        if (val > max_v)
        {
            max_v = val;
            max_i = i;
        }
    }
    return max_i;
}

int main(int argc, char **argv)
{
    printf("Inicio\n");
    if (argc < 7)
    {
        printf("Uso: %s <num_blocks> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>\n", argv[0]);
        return 1;
    }

    int num_blocks = atoi(argv[1]);
    char *dataset_file = argv[2];
    int n_folds = atoi(argv[3]);
    float l_rate = atof(argv[4]);
    int n_epoch = atoi(argv[5]);
    int n_hidden = atoi(argv[6]);

    assert(num_blocks > 0);
    assert(n_hidden <= MAX_NEURONS);
    assert(n_folds > 0 && n_folds <= MAX_ROWS);
    assert(n_epoch > 0);
    assert(l_rate > 0.0f && l_rate <= 1.0f);

    srand(time(NULL));

    int rows, cols;
    float *dataset = load_csv_data(dataset_file, &rows, &cols);
    normalize_data(dataset, rows, cols);

    if (VERBOSE)
        printf("Dataset carregado com %d linhas e %d colunas.\n", rows, cols);

    float mean_accuracy = evaluate_network(dataset, rows, cols, n_folds, l_rate, n_epoch, n_hidden, num_blocks);
    printf("Acuracia media: %.3f\n", mean_accuracy);

    free(dataset);
    return 0;
}