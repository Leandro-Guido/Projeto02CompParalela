
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
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
float evaluate_network(float *dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden, int num_threads);
float rand_weight() { return (float)rand() / (float)RAND_MAX; }

// aproximação de expf(x) válida para -10 <= x <= 10
#pragma omp declare target
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
#pragma omp end declare target

// OPERAÇÕES NETWORK
void initialize_network(int fold_id, int n_inputs, int n_hidden, int n_outputs, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *activations, float *deltas);
void forward_propagate(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer);
void backward_propagate_error(int fold, int n_layers, int *n_neurons, float *weights, float *outputs, float *deltas, float *expected);
void update_weights(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *deltas, float *inputs, float *layer_buffer, float l_rate);
int predict(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer);

void print_network(int net_id, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *activations, float *deltas)
{
    printf("===== NETWORK %d =====\n", net_id);

    for (int l = 0; l < n_layers; l++)
    {
        printf("Layer %d: %d neurons\n", l, n_neurons[IDX_LAYER(net_id, l)]);
        for (int n = 0; n < n_neurons[IDX_LAYER(net_id, l)]; n++)
        {
            int n_w = n_weights[IDX_NEURON(net_id, l, n)];
            printf("  Neuron %d: weights = [", n);
            for (int w = 0; w < n_w; w++)
            {
                printf("%.4f", weights[IDX_WEIGHT(net_id, l, n, w)]);
                if (w < n_weights[IDX_NEURON(net_id, l, n)] - 1)
                    printf(", ");
            }
            printf("], activation = %.4f, output = %.4f, delta=%.4f\n", activations[IDX_NEURON(net_id, l, n)], outputs[IDX_NEURON(net_id, l, n)], deltas[IDX_NEURON(net_id, l, n)]);
        }
    }
    printf("=======================\n");
}

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

void normalize_data(float *data, int rows, int cols)
{
    float *mins = malloc(cols * sizeof(float));
    float *maxs = malloc(cols * sizeof(float));
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

float evaluate_network(float *dataset, int rows, int cols, int n_folds, float l_rate, int n_epoch, int n_hidden, int num_threads)
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
    int *n_neurons;     // [net][layer]
    float *weights;     // [net][layer][neuron][weight]
    float *outputs;     // [net][layer][neuron]
    float *activations; // [net][layer][neuron]
    float *deltas;      // [net][layer][neuron]
    int *n_weights;     // [net][layer][neuron]

    n_neurons = malloc(TOTAL_LAYERS * sizeof(int));
    weights = malloc(TOTAL_WEIGHTS * sizeof(float));
    outputs = malloc(TOTAL_NEURONS * sizeof(float));
    activations = malloc(TOTAL_NEURONS * sizeof(float));
    deltas = malloc(TOTAL_NEURONS * sizeof(float));
    n_weights = malloc(TOTAL_NEURONS * sizeof(int));
    float *layer_buffer = malloc(TOTAL_NEURONS * sizeof(float)); // Para auxiliar no forward propagation e update_weights

    for (int i = 0; i < n_folds * n_layers; i++)
    {
        n_neurons[IDX_LAYER(i, 0)] = n_hidden;
        n_neurons[IDX_LAYER(i, 1)] = n_outputs;
    }
    for (int i = 0; i < TOTAL_WEIGHTS; i++)
        weights[i] = -FLT_MAX;
    for (int i = 0; i < TOTAL_NEURONS; i++)
    {
        outputs[i] = -FLT_MAX;
        activations[i] = -FLT_MAX;
        deltas[i] = -FLT_MAX;
        layer_buffer[i] = -FLT_MAX;
    }
    for (int i = 0; i < TOTAL_NEURONS; i++)
        n_weights[i] = -1;

    float *train_sets = malloc(n_folds * (rows - fold_size) * cols * sizeof(float));
    float *test_sets = malloc(n_folds * fold_size * cols * sizeof(float));
    int *expected_labels = malloc(n_folds * fold_size * sizeof(int));
    int *predicted_labels = malloc(n_folds * fold_size * sizeof(int));
    float *expected_output = malloc(n_outputs * sizeof(float));

    for (int i = 0; i < n_folds; i++)
    {
        int start = i * fold_size;
        int end = start + fold_size;
        int train_idx = 0, test_idx = 0;

        for (int r = 0; r < rows; r++)
        {
            if (r >= start && r < end)
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
        initialize_network(i, cols - 1, n_hidden, n_outputs, n_layers, n_neurons, n_weights, weights, outputs, activations, deltas);
    }

#if VERBOSE > 1
    // print_network(0, n_layers, n_neurons, n_weights, weights, outputs, activations, deltas);
    printf("===== NETWORK CONFIGURATION =====\n");
    printf("N_NEURONS arr:");
    print_arr_int(n_neurons, TOTAL_LAYERS);
    printf("N_WEIGHTS arr:");
    print_arr_int(n_weights, TOTAL_NEURONS);
    printf("WEIGHTS arr:");
    print_arr_float(weights, TOTAL_WEIGHTS);
    printf("OUTPUTS arr:");
    print_arr_float(outputs, TOTAL_NEURONS);
    printf("ACTIVATIONS arr:");
    print_arr_float(activations, TOTAL_NEURONS);
    printf("DELTAS arr:");
    print_arr_float(deltas, TOTAL_NEURONS);
    printf("LAYER_BUFFER arr:");
    print_arr_float(layer_buffer, TOTAL_NEURONS);
    printf("EXPECTED arr:");
    print_arr_float(expected_output, n_outputs);
    printf("=================================\n");
#endif

/*
    TREINO DA NETWORK
*/
#pragma omp target data                                           \
    map(to : train_sets[0 : n_folds * (rows - fold_size) * cols], \
            n_neurons[0 : TOTAL_LAYERS],                          \
            n_weights[0 : TOTAL_NEURONS])                         \
    map(tofrom : weights[0 : TOTAL_WEIGHTS],                      \
            outputs[0 : TOTAL_NEURONS],                           \
            activations[0 : TOTAL_NEURONS],                       \
            deltas[0 : TOTAL_NEURONS],                            \
            layer_buffer[0 : TOTAL_NEURONS],                      \
            expected_output[0 : n_outputs])
    {
#pragma omp target teams distribute parallel for thread_limit(num_threads)
        for (int i = 0; i < n_folds; i++)
        {
            // printf("nteams: %d, th lim: %d\n", omp_get_num_teams(), omp_get_thread_limit());
            for (int epoch = 0; epoch < n_epoch; epoch++)
            {
                float sum_error = 0.0f;
                for (int r = 0; r < rows - fold_size; r++)
                {

                    float *row = &train_sets[(i * (rows - fold_size) + r) * cols];

                    forward_propagate(i, n_layers, n_neurons, n_weights, weights, activations, outputs, row, layer_buffer);

                    for (int k = 0; k < n_outputs; k++)
                        expected_output[k] = 0.0f;
                    expected_output[(int)row[cols - 1]] = 1.0f;

                    int last_layer = n_layers - 1;
                    for (int j = 0; j < n_outputs; j++)
                    {
                        float diff = expected_output[j] - outputs[IDX_NEURON(i, last_layer, j)];
                        sum_error += diff * diff;
                    }

                    backward_propagate_error(i, n_layers, n_neurons, weights, outputs, deltas, expected_output);
                    update_weights(i, n_layers, n_neurons, n_weights, weights, outputs, deltas, row, layer_buffer, l_rate);
                }
#if SLOW > 0 // ESSE CODIGO E APENAS PARA MELHORAR VISUALIZAÇÃO, NÃO DEVE SER USADO PARA BENCHMARKS
    unsigned long delay_iters = 10000000UL;
    for (unsigned long k = 0; k < delay_iters; ++k) {
        asm volatile("" ::: "memory"); // evita que o compilador otimize o loop fora
    }
    for (unsigned long k = 0; k < delay_iters; ++k) {
        asm volatile("" ::: "memory");
    }
#endif
#if VERBOSE > 0
                int team = omp_get_team_num();
                printf("[%.3d>%.3d]\tepoch=%.3d, l_rate=%.3f, error=%.6f\n", team, i, epoch, l_rate, sum_error);
#endif
            }
        }
    }

    /*
        TESTE DAS NETWORKS E LIBERAÇÃO DOS RECURSOS
    */
    for (int i = 0; i < n_folds; i++)
    {
        for (int j = 0; j < fold_size; j++)
        {
            predicted_labels[i * fold_size + j] = predict(i, n_layers, n_neurons, n_weights, weights, activations, outputs, &test_sets[(i * fold_size + j) * cols], layer_buffer);
        }
        sum_accuracy += accuracy_metric(&expected_labels[i * fold_size], &predicted_labels[i * fold_size], fold_size);
    }

#if VERBOSE > 1
    // print_network(0, n_layers, n_neurons, n_weights, weights, outputs, activations, deltas);
    printf("===== NETWORK CONFIGURATION =====\n");
    printf("N_NEURONS arr:");
    print_arr_int(n_neurons, TOTAL_LAYERS);
    printf("N_WEIGHTS arr:");
    print_arr_int(n_weights, TOTAL_NEURONS);
    printf("WEIGHTS arr:");
    print_arr_float(weights, TOTAL_WEIGHTS);
    printf("OUTPUTS arr:");
    print_arr_float(outputs, TOTAL_NEURONS);
    printf("ACTIVATIONS arr:");
    print_arr_float(activations, TOTAL_NEURONS);
    printf("DELTAS arr:");
    print_arr_float(deltas, TOTAL_NEURONS);
    printf("LAYER_BUFFER arr:");
    print_arr_float(layer_buffer, TOTAL_NEURONS);
    printf("EXPECTED arr:");
    print_arr_float(expected_output, n_outputs);
    printf("=================================\n");
#endif

    free(train_sets);
    free(test_sets);
    free(expected_labels);
    free(predicted_labels);
    free(expected_output);
    free(layer_buffer);

    return sum_accuracy / (float)n_folds;
}

#pragma omp declare target
void forward_propagate(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
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
#pragma omp end declare target

#pragma omp declare target
void backward_propagate_error(int fold, int n_layers, int *n_neurons, float *weights, float *outputs, float *deltas, float *expected)
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
#pragma omp end declare target

#pragma omp declare target
void update_weights(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *deltas, float *inputs, float *layer_buffer, float l_rate)
{
    float *in = inputs; // inicia com os inputs

    for (int l = 0; l < n_layers; l++)
    {
        for (int n = 0; n < n_neurons[IDX_LAYER(fold, l)]; n++)
        {
            int nw = n_weights[IDX_NEURON(fold, l, n)];
            for (int j = 0; j < nw - 1; j++) // atualiza os pesos com o gradiente descendente
            {
                weights[IDX_WEIGHT(fold, l, n, j)] += l_rate * deltas[IDX_NEURON(fold, l, n)] * in[j];
            }
            weights[IDX_WEIGHT(fold, l, n, nw - 1)] += l_rate * deltas[IDX_NEURON(fold, l, n)]; // atualiza o peso do bias
        }

        // prepara os inputs para a proxima camada
        for (int i = 0; i < n_neurons[IDX_LAYER(fold, l)]; i++)
        {
            layer_buffer[i] = outputs[IDX_NEURON(fold, l, i)];
        }

        in = &layer_buffer[IDX_NEURON(fold, l, 0)]; // atualiza os inputs para a proxima camada
    }
}
#pragma omp end declare target

void initialize_network(int fold_id, int n_inputs, int n_hidden, int n_outputs, int n_layers, int *n_neurons, int *n_weights, float *weights, float *outputs, float *activations, float *deltas)
{
    // REDE NEURAL COM 2 CAMADAS
    // n_layers = 2

    n_neurons[IDX_LAYER(fold_id, 0)] = n_hidden; // primeira camada tem n_hidden neurônios
    for (int n = 0; n < n_neurons[IDX_LAYER(fold_id, 0)]; n++)
    {
        activations[IDX_NEURON(fold_id, 0, n)] = 0.0f;
        outputs[IDX_NEURON(fold_id, 0, n)] = 0.0f;
        deltas[IDX_NEURON(fold_id, 0, n)] = 0.0f;
        n_weights[IDX_NEURON(fold_id, 0, n)] = n_inputs + 1;
        for (int w = 0; w < n_inputs + 1; w++)
        {
            weights[IDX_WEIGHT(fold_id, 0, n, w)] = rand_weight();
        }
    }

    n_neurons[IDX_LAYER(fold_id, 1)] = n_outputs; // segunda camada tem n_outputs neurônios
    for (int n = 0; n < n_neurons[IDX_LAYER(fold_id, 1)]; n++)
    {
        activations[IDX_NEURON(fold_id, 1, n)] = 0.0f;
        outputs[IDX_NEURON(fold_id, 1, n)] = 0.0f;
        deltas[IDX_NEURON(fold_id, 1, n)] = 0.0f;
        n_weights[IDX_NEURON(fold_id, 1, n)] = n_hidden + 1;
        for (int w = 0; w < n_hidden + 1; w++)
        {
            weights[IDX_WEIGHT(fold_id, 1, n, w)] = rand_weight();
        }
    }
}

int predict(int fold, int n_layers, int *n_neurons, int *n_weights, float *weights, float *activations, float *outputs, float *inputs, float *layer_buffer)
{
    forward_propagate(fold, n_layers, n_neurons, n_weights, weights, activations, outputs, inputs, layer_buffer);

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
        printf("Uso: %s <num_threads> <dataset.csv> <n_folds> <l_rate> <n_epoch> <n_hidden>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    char *dataset_file = argv[2];
    int n_folds = atoi(argv[3]);
    float l_rate = atof(argv[4]);
    int n_epoch = atoi(argv[5]);
    int n_hidden = atoi(argv[6]);

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

    float mean_accuracy = evaluate_network(dataset, rows, cols, n_folds, l_rate, n_epoch, n_hidden, num_threads);
    printf("Acuracia media: %.3f\n", mean_accuracy);

    free(dataset);
    return 0;
}