#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_ROWS 40000
#define MAX_COLS 100
#define MAX_NEURONS 100

#ifndef VERBOSE
#define VERBOSE 0
#endif

typedef struct {
    float *weights;
    float output, activation, delta;
    int n_weights;
} Neuron;

typedef struct {
    Neuron *neurons;
    int n_neurons;
} Layer;

typedef struct {
    Layer *layers;
    int n_layers;
} Network;

__device__ float rand_weight(curandState *state) {
    return curand_uniform(state);
}

__global__ void evaluate_kernel(float **dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden, float *accuracies);
__device__ void initialize_network(Network *net, int n_inputs, int n_hidden, int n_outputs);
__device__ void train(Network *net, float **train_data, int train_rows, float l_rate, int n_epoch, int n_outputs, int id);
__device__ int predict(Network *net, float *input);
__device__ void free_network(Network *net);
__device__ float accuracy_metric(int *expected, int *predicted, int size);
__device__ void forward_propagate(Network *net, float *inputs);
__device__ void backward_propagate_error(Network *net, float *expected);
__device__ void update_weights(Network *net, float *inputs, float l_rate);
float evaluate_network(float **dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden);

float **load_csv_data(const char *filename, int *rows, int *cols);
void normalize_data(float **data, int rows, int cols);

int main(int argc, char **argv)
{
    if (argc < 7) {
        printf("Uso: %s <num_threads> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    char *dataset_file = argv[2];
    int n_folds = atoi(argv[3]);
    float l_rate = atof(argv[4]);
    int n_epoch = atoi(argv[5]);
    int n_hidden = atoi(argv[6]);

    srand(time(NULL));

    int rows, cols;
    float **dataset = load_csv_data(dataset_file, &rows, &cols);
    normalize_data(dataset, rows, cols);

    float mean_accuracy = evaluate_network(dataset, rows, cols, n_folds, l_rate, n_epoch, n_hidden);
    printf("Acurácia média: %.3f\n", mean_accuracy);
    return 0;
}

float evaluate_network(float **dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden)
{
    float sum_accuracy = 0.0f;
    float *d_accuracies, *h_accuracies = (float *)malloc(n_folds * sizeof(float));
    cudaMalloc(&d_accuracies, n_folds * sizeof(float));

    int threads = 256;
    int blocks = (n_folds + threads - 1) / threads;

    evaluate_kernel<<<blocks, threads>>>(dataset, rows, cols, n_folds, l_rate, n_epoch, n_hidden, d_accuracies);
    cudaDeviceSynchronize();

    cudaMemcpy(h_accuracies, d_accuracies, n_folds * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_folds; i++)
        sum_accuracy += h_accuracies[i];

    cudaFree(d_accuracies);
    free(h_accuracies);

    return sum_accuracy / (float)n_folds;
}
// --------------------------------------------

__global__ void evaluate_kernel(
    float **dataset, int rows, int cols, int n_folds, float l_rate,
    int n_epoch, int n_hidden, float *accuracies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_folds) return;

    int fold_size = rows / n_folds;
    int start = i * fold_size;
    int end = start + fold_size;
    const int n_outputs = 2;

    float **train_set = (float **)malloc((rows - fold_size) * sizeof(float *));
    float **test_set = (float **)malloc(fold_size * sizeof(float *));
    int *expected = (int *)malloc(fold_size * sizeof(int));
    int train_idx = 0, test_idx = 0;

    for (int r = 0; r < rows; r++) {
        if (r >= start && r < end) {
            test_set[test_idx] = (float *)malloc(cols * sizeof(float));
            memcpy(test_set[test_idx], dataset[r], cols * sizeof(float));
            expected[test_idx] = (int)test_set[test_idx][cols - 1];
            test_set[test_idx][cols - 1] = -1;
            test_idx++;
        } else {
            train_set[train_idx++] = dataset[r];
        }
    }

    Network net;
    initialize_network(&net, cols - 1, n_hidden, n_outputs);
    train(&net, train_set, rows - fold_size, l_rate, n_epoch, n_outputs, i);

    int *predicted = (int *)malloc(fold_size * sizeof(int));
    for (int j = 0; j < fold_size; j++) {
        predicted[j] = predict(&net, test_set[j]);
    }

    float acc = accuracy_metric(expected, predicted, fold_size);
    accuracies[i] = acc;

    for (int j = 0; j < fold_size; j++) free(test_set[j]);
    free(test_set);
    free(train_set);
    free(predicted);
    free(expected);
    free_network(&net);
}
// carrega os dados do CSV como uma matriz de floats
float **load_csv_data(const char *filename, int *rows, int *cols)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Erro ao abrir CSV");
        exit(1);
    }

    float **data = (float **)malloc(MAX_ROWS * sizeof(float *));
    char line[4096];
    int r = 0, c = 0;

    while (fgets(line, sizeof(line), fp) && r < MAX_ROWS)
    {
        data[r] = (float *)malloc(MAX_COLS * sizeof(float));
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
    float *mins = (float *)malloc(cols * sizeof(float));
    float *maxs = (float *)malloc(cols * sizeof(float));
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

// --------------------------------------------

__device__ void initialize_network(Network *net, int n_inputs, int n_hidden, int n_outputs)
{
    curandState state;
    // REDE NEURAL COM 2 CAMADAS
    net->n_layers = 2;
    net->layers = (Layer *)malloc(2 * sizeof(Layer));

    net->layers[0].n_neurons = n_hidden;
    net->layers[0].neurons = (Neuron *)malloc(n_hidden * sizeof(Neuron));
    for (int i = 0; i < n_hidden; i++)
    {
        net->layers[0].neurons[i].n_weights = n_inputs + 1;
        net->layers[0].neurons[i].weights = (float *)malloc((n_inputs + 1) * sizeof(float));
        for (int j = 0; j < n_inputs + 1; j++)
        {
            curand_init(1234, j, 0, &state);
            net->layers[0].neurons[i].weights[j] = rand_weight(&state);
        }
    }

    net->layers[1].n_neurons = n_outputs;
    net->layers[1].neurons = (Neuron *)malloc(n_outputs * sizeof(Neuron));
    for (int i = 0; i < n_outputs; i++)
    {
        net->layers[1].neurons[i].n_weights = n_hidden + 1;
        net->layers[1].neurons[i].weights = (float *)malloc((n_hidden + 1) * sizeof(float));
        for (int j = 0; j < n_hidden + 1; j++)
        {
            curand_init(1234, j, 0, &state);
            net->layers[1].neurons[i].weights[j] = rand_weight(&state);
        }
    }
}

__device__ void forward_propagate(Network *net, float *inputs)
{
    float *in = inputs; // inicia com os inputs

    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *layer = &net->layers[l];
        float *new_inputs = (float *)malloc(layer->n_neurons * sizeof(float));

        for (int n = 0; n < layer->n_neurons; n++)
        {
            Neuron *neuron = &layer->neurons[n];

             // começa a ativacao com o bias (ultimo peso)
            neuron->activation = neuron->weights[neuron->n_weights - 1];
            for (int j = 0; j < neuron->n_weights - 1; j++) 
            {
                neuron->activation += neuron->weights[j] * in[j]; // soma ponderada dos inputs
            }
            // aplica a funcao de ativacao sigmoide
            neuron->output = 1.0f / (1.0f + expf(-neuron->activation));
            new_inputs[n] = neuron->output;
        }

        in = new_inputs; // atualiza os inputs para a proxima camada
    }
}

__device__ void backward_propagate_error(Network *net, float *expected)
{
    for (int l = net->n_layers - 1; l >= 0; l--) // começa da ultima camada
    {
        Layer *layer = &net->layers[l];
        for (int n = 0; n < layer->n_neurons; n++)
        {
            Neuron *neuron = &layer->neurons[n];
            float error = 0.0f;

            if (l == net->n_layers - 1) // se for a camada de saida, erro é a diferenca entre o esperado e o atual
            {
                error = expected[n] - neuron->output; 
            }
            else // se for uma camada escondida, soma os erros ponderados da proxima camada
            {
                Layer *next_layer = &net->layers[l + 1];
                for (int k = 0; k < next_layer->n_neurons; k++)
                {
                    error += next_layer->neurons[k].weights[n] * next_layer->neurons[k].delta;
                }
            }
            // delta = erro * derivada da funcao de ativacao
            neuron->delta = error * neuron->output * (1.0f - neuron->output);
        }
    }
}

__device__ void update_weights(Network *net, float *inputs, float l_rate)
{
    float *in = inputs; // inicia com os inputs

    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *layer = &net->layers[l];
        for (int n = 0; n < layer->n_neurons; n++)
        {
            Neuron *neuron = &layer->neurons[n];
            for (int j = 0; j < neuron->n_weights - 1; j++) // atualiza os pesos com o gradiente descendente
            {
                neuron->weights[j] += l_rate * neuron->delta * in[j];
            }
            neuron->weights[neuron->n_weights - 1] += l_rate * neuron->delta; // atualiza o peso do bias
        }

        // prepara os inputs para a proxima camada
        float *new_in = (float *)malloc(layer->n_neurons * sizeof(float));
        for (int i = 0; i < layer->n_neurons; i++)
        {
            new_in[i] = layer->neurons[i].output;
        }

        in = new_in; // atualiza os inputs para a proxima camada
    }
}

__device__ void train(Network *net, float **train_data, int train_rows, float l_rate, int n_epoch, int n_outputs, int id)
{
    int cols = net->layers[0].neurons[0].n_weights - 1; 
    float *expected = (float *)malloc(n_outputs * sizeof(float));

    for (int epoch = 0; epoch < n_epoch; epoch++)
    {
        float sum_error = 0.0f;

        for (int r = 0; r < train_rows; r++)
        {
            float *row = train_data[r];
            forward_propagate(net, row);

            for (int i = 0; i < n_outputs; i++)
                expected[i] = 0.0f;
            expected[(int)row[cols]] = 1.0f;

            Layer *output_layer = &net->layers[net->n_layers - 1];
            for (int i = 0; i < n_outputs; i++)
            {
                float diff = expected[i] - output_layer->neurons[i].output;
                sum_error += diff * diff;
            }

            backward_propagate_error(net, expected);
            update_weights(net, row, l_rate);
        }

#if VERBOSE > 0
        int thread = omp_get_thread_num();
        printf("[%d>%d] epoch=%d, l_rate=%.3f, error=%.6f\n", thread, id, epoch, l_rate, sum_error);
        fflush(stdout);
#endif
    }

    free(expected);
}

__device__ int predict(Network *net, float *input)
{
    forward_propagate(net, input);
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

__device__ float accuracy_metric(int *expected, int *predicted, int size)
{
    int correct = 0;
    for (int i = 0; i < size; i++)
    {
        if (expected[i] == predicted[i])
            correct++;
    }
    return 100.0f * correct / size;
}

__device__ void free_network(Network *net)
{
    for (int l = 0; l < net->n_layers; l++)
    {
        for (int n = 0; n < net->layers[l].n_neurons; n++)
        {
            free(net->layers[l].neurons[n].weights);
        }
        free(net->layers[l].neurons);
    }
    free(net->layers);
}
