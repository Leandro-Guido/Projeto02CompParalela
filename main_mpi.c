
// main_mpi.c: versão MPI de main_gpu.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <omp.h>
#include <mpi.h>

#define MAX_ROWS 40000
#define COLS 15
#define MAX_NEURONS COLS

#ifndef VERBOSE
#define VERBOSE 0
#endif

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
float evaluate_network(float *dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden, int rank, int size);
float rand_weight() { return (float)rand() / (float)RAND_MAX; }

// aproximação de expf(x) válida para -10 <= x <= 10
float fast_exp(float x)
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
float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }
float sigmoid_derivative(float output) { return output * (1.0f - output); }

// OPERAÇÕES NETWORK
void forward_propagate(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer);
void backward_propagate_error(int fold, int n_layers, int *n_neurons, float *weights, float *outputs, float *deltas, float *expected);
void update_weights(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *deltas, float *inputs, float *layer_buffer, float l_rate);
int predict(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer);

void print_arr_int(int *arr, int size)
{
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

void print_arr_float(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (arr[i] == -FLT_MAX)
            printf(" -inf ");
        else if (arr[i] == FLT_MAX)
            printf(" inf ");
        else
            printf("%.4f ", arr[i]);
    }
    printf("\n");
}

void print_data(float *data, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", data[i * COLS + j]);
        }
        printf("\n");
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
    float *data = malloc(MAX_ROWS * COLS * sizeof(float));
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
    float *mins = malloc(cols * sizeof(float));
    float *maxs = malloc(cols * sizeof(float));
    for (int j = 0; j < cols - 1; j++)
    {
        mins[j] = maxs[j] = data[j];
        for (int i = 1; i < rows; i++)
        {
            float v = data[i * cols + j];
            if (v < mins[j])
                mins[j] = v;
            if (v > maxs[j])
                maxs[j] = v;
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

// Avalia rede distribuindo folds entre processos MPI
float evaluate_network(float *dataset, int rows, int cols, int n_folds,
                       float l_rate, int n_epoch, int n_hidden,
                       int rank, int size)
{
    /*
        INICIALIZAÇÃO DOS DADOS E NETWORK
    */
    float sum_accuracy = 0.0f;
    int fold_size = rows / n_folds;

    int n_layers = 2;
    const int n_outputs = 2; // no nosso caso so tem as classes renda anual <=50K e >50K

    int *n_neurons = malloc(n_folds * n_layers * sizeof(int));
    int *n_weights = malloc(n_folds * n_layers * MAX_NEURONS * sizeof(int));
    float *weights = malloc(n_folds * n_layers * MAX_NEURONS * MAX_NEURONS * sizeof(float));
    float *outputs = malloc(n_folds * n_layers * MAX_NEURONS * sizeof(float));
    float *activations = malloc(n_folds * n_layers * MAX_NEURONS * sizeof(float));
    float *deltas = malloc(n_folds * n_layers * MAX_NEURONS * sizeof(float));
    float *layer_buffer = malloc(n_folds * n_layers * MAX_NEURONS * sizeof(float)); // para auxiliar no forward propagation e update_weights

    // inicialize_network
    for (int f = 0; f < n_folds; f++)
    {
        for (int l = 0; l < n_layers; l++)
        {
            int neurons = (l == 0 ? n_hidden : n_outputs);
            n_neurons[IDX_LAYER(f, l)] = neurons;
            for (int n = 0; n < neurons; n++)
            {
                n_weights[IDX_NEURON(f, l, n)] = (l == 0 ? cols - 1 : n_hidden) + 1;
                for (int w = 0; w < n_weights[IDX_NEURON(f, l, n)]; w++)
                {
                    weights[IDX_WEIGHT(f, l, n, w)] = rand_weight();
                }
            }
        }
    }

    float *train_sets = malloc(n_folds * (rows - fold_size) * cols * sizeof(float));
    float *test_sets = malloc(n_folds * fold_size * cols * sizeof(float));
    int *expected_labels = malloc(n_folds * fold_size * sizeof(int));
    for (int i = 0; i < n_folds; i++)
    {
        int start = i * fold_size, train_idx = 0, test_idx = 0;
        for (int r = 0; r < rows; r++)
        {
            if (r >= start && r < start + fold_size)
            {
                memcpy(&test_sets[(i * fold_size + test_idx) * cols], &dataset[r * cols], cols * sizeof(float));
                expected_labels[i * fold_size + test_idx] = (int)test_sets[(i * fold_size + test_idx) * cols + (cols - 1)];
                test_sets[(i * fold_size + test_idx) * cols + (cols - 1)] = -1;
                test_idx++;
            }
            else
            {
                memcpy(&train_sets[(i * (rows - fold_size) + train_idx) * cols], &dataset[r * cols], cols * sizeof(float));
                train_idx++;
            }
        }
    }

    int local_count = 0;
    int *predicted_labels = malloc(fold_size * sizeof(int));
    float *expected_output = malloc(n_outputs * sizeof(float));

/*
TREINO DA NETWORK
    paralelismo OpenMP, mesmas explicações que no main.c e main_gpu.c
*/
#pragma omp parallel for
    for (int f = rank; f < n_folds; f += size)
    {

        for (int epoch = 0; epoch < n_epoch; epoch++)
        {
            float sum_error = 0.0f;
            for (int r = 0; r < rows - fold_size; r++)
            {
                float *row = &train_sets[(f * (rows - fold_size) + r) * cols];
                forward_propagate(f, n_layers, n_neurons, n_weights, weights, activations, outputs, row, layer_buffer);

                for (int k = 0; k < n_outputs; k++)
                    expected_output[k] = 0.0f;
                expected_output[(int)row[cols - 1]] = 1.0f;

                int last_layer = n_layers - 1;
                for (int j = 0; j < n_outputs; j++)
                {
                    float diff = expected_output[j] - outputs[IDX_NEURON(f, last_layer, j)];
                    sum_error += diff * diff;
                }

                backward_propagate_error(f, n_layers, n_neurons, weights, outputs, deltas, expected_output);
                update_weights(f, n_layers, n_neurons, n_weights, weights, outputs, deltas, row, layer_buffer, l_rate);
            }
#if VERBOSE > 0
            int thread = omp_get_thread_num();
            printf("[Proc:%.3d|Fold:%.3d|Thread:%.3d]\tepoch=%.3d, l_rate=%.3f, error=%.6f\n", rank, f, thread, epoch, l_rate, sum_error);
            fflush(stdout);
#endif
        }
        // Teste
        for (int j = 0; j < fold_size; j++)
        {
            predicted_labels[j] = predict(f, n_layers, n_neurons, n_weights, weights, activations, outputs, &test_sets[(f * fold_size + j) * cols], layer_buffer);
        }
        sum_accuracy += accuracy_metric(&expected_labels[f * fold_size], predicted_labels, fold_size);
        local_count++;
    }

    free(train_sets);
    free(test_sets);
    free(expected_labels);
    free(predicted_labels);
    free(expected_output);
    free(n_neurons);
    free(n_weights);
    free(outputs);
    free(activations);
    free(deltas);
    free(layer_buffer);
    free(weights);

    return sum_accuracy / (local_count > 0 ? local_count : 1);
}

void forward_propagate(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
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
            float out = sigmoid(activations[IDX_NEURON(fold, l, n)]);
            outputs[IDX_NEURON(fold, l, n)] = out;
            layer_buffer[IDX_NEURON(fold, l, n)] = out;
        }
        in = &layer_buffer[IDX_NEURON(fold, l, 0)];
    }
}

void backward_propagate_error(int fold, int n_layers, int *n_neurons, float *weights, float *outputs, float *deltas, float *expected)
{
    for (int l = n_layers - 1; l >= 0; l--)
    {
        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            float err = 0.0f;
            if (l == n_layers - 1)
            {
                err = expected[n] - outputs[IDX_NEURON(fold, l, n)];
            }
            else
            {
                int nl = l + 1;
                for (int k = 0; k < n_neurons[IDX_LAYER(fold, nl)]; k++)
                {
                    err += weights[IDX_WEIGHT(fold, nl, k, n)] * deltas[IDX_NEURON(fold, nl, k)];
                }
            }
            deltas[IDX_NEURON(fold, l, n)] = err * sigmoid_derivative(outputs[IDX_NEURON(fold, l, n)]);
        }
    }
}

void update_weights(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *deltas, float *inputs, float *layer_buffer, float l_rate)
{
    float *in = inputs;
    for (int l = 0; l < n_layers; l++)
    {
        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            int nw = n_weights[IDX_NEURON(fold, l, n)];
            for (int j = 0; j < nw - 1; j++)
            {
                weights[IDX_WEIGHT(fold, l, n, j)] += l_rate * deltas[IDX_NEURON(fold, l, n)] * in[j];
            }
            weights[IDX_WEIGHT(fold, l, n, nw - 1)] += l_rate * deltas[IDX_NEURON(fold, l, n)];
        }
        for (int i = 0; i < n_neurons[IDX_LAYER(fold, l)]; i++)
        {
            layer_buffer[i] = outputs[IDX_NEURON(fold, l, i)];
        }
        in = &layer_buffer[IDX_NEURON(fold, l, 0)];
    }
}

int predict(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
{
    forward_propagate(fold, n_layers, n_neurons, n_weights, weights, activations, outputs, inputs, layer_buffer);
    int n_out = n_neurons[IDX_LAYER(fold, n_layers - 1)];
    int max_i = 0;
    float max_v = outputs[IDX_NEURON(fold, n_layers - 1, 0)];
    for (int i = 1; i < n_out; i++)
    {
        float v = outputs[IDX_NEURON(fold, n_layers - 1, i)];
        if (v > max_v)
        {
            max_v = v;
            max_i = i;
        }
    }
    return max_i;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 7)
    {
        if (rank == 0)
            printf("Uso: %s <num_thread> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int num_threads = atoi(argv[1]);
    char *dataset_file = argv[2];
    int n_folds = atoi(argv[3]);
    float l_rate = atof(argv[4]);
    int n_epoch = atoi(argv[5]);
    int n_hidden = atoi(argv[6]);

    omp_set_num_threads(num_threads);
    srand(time(NULL));

    float *dataset = NULL;
    int rows = 0, cols = 0;
    if (rank == 0)
    {
        dataset = load_csv_data(dataset_file, &rows, &cols);
        normalize_data(dataset, rows, cols);
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        dataset = malloc(rows * cols * sizeof(float));
    MPI_Bcast(dataset, rows * cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float local_acc = evaluate_network(dataset, rows, cols, n_folds, l_rate, n_epoch, n_hidden, rank, size);
    float global_acc = 0.0f;
    MPI_Reduce(&local_acc, &global_acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Acuracia media MPI: %.3f\n", global_acc / size);

    free(dataset);
    MPI_Finalize();
    return 0;
}
