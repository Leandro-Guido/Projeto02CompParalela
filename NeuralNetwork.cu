#include "NeuralNetwork.h"

#include <iostream>
#include <cstdlib> // Para rand, mas precisaria ser substituído por cuRAND no device
#include <cmath>
#include <ctime>   // Para srand, mas precisaria ser substituído por cuRAND no device
#include <algorithm>
// #include <omp.h> // Remover OpenMP

// Para alocação de memória na GPU (se necessário diretamente nas classes)
#include <cuda_runtime.h> 

/* NEURON */

/*
* Neuron Constructor
*/
CUDA_CALLABLE_MEMBER Neuron::Neuron(int n_weights) {
    m_nWeights = n_weights;
    // Alocação de pesos para a GPU (ex: no kernel)
    // Isso deve ser feito de forma mais controlada, talvez através de um alocador global
    // ou passando memória já alocada. Por enquanto, new/delete para demonstrar
    // (Atenção: new/delete em __device__ é restrito e deve ser evitado para performance.
    // Usar alocadores globais ou arrays pré-alocados é preferível)
    #ifdef __CUDA_ARCH__ // Se compilando para o device
        m_weights = new float[n_weights]; // Exemplo, mas é problemático em kernels
    #else // Se compilando para o host
        m_weights = new float[n_weights];
    #endif
    this->initWeights(n_weights); // Inicialização dos pesos
    m_activation = 0;
    m_output = 0;
    m_delta = 0;
}

/*
* Neuron Destructor
*/
CUDA_CALLABLE_MEMBER Neuron::~Neuron() {
    #ifdef __CUDA_ARCH__
        delete[] m_weights;
    #else
        delete[] m_weights;
    #endif
}

/*
* Initialize weights
*/
CUDA_CALLABLE_MEMBER void Neuron::initWeights(int n_weights) {
    // No ambiente CUDA, std::rand() não é seguro ou eficiente.
    // Para um kernel, você usaria cuRAND. Para esta simulação, vou manter a lógica,
    // mas esteja ciente que isso não é otimizado para GPU.
    for (int w = 0; w < n_weights; w++) {
        // Inicialização de pesos com valores aleatórios entre 0 e 1 (simulando)
        m_weights[w] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

/* * Calculate the activation of a neuron for a given input
*/
CUDA_CALLABLE_MEMBER void Neuron::activate(float* inputs, int num_inputs) {
    m_activation = m_weights[m_nWeights-1]; // O último peso é o bias

    for (int i = 0; i < num_inputs; i++) {
        m_activation += m_weights[i] * inputs[i];
    }
}

/* * Transfer the activation of the neuron to an actual output
*/
CUDA_CALLABLE_MEMBER void Neuron::transfer() {
    m_output = 1.0f / (1.0f + expf(-m_activation)); // expf para float
}

/* LAYER */

/*
* Layer Constructor
*/
CUDA_CALLABLE_MEMBER Layer::Layer(int n_neurons, int n_weights) {
    m_nNeurons = n_neurons; // Adicionar o número de neurônios
    #ifdef __CUDA_ARCH__
        m_neurons = new Neuron[n_neurons]; // Alocar array de Neurons
    #else
        m_neurons = new Neuron[n_neurons];
    #endif
    this->initNeurons(n_neurons, n_weights);
}

/*
* Layer Destructor
*/
CUDA_CALLABLE_MEMBER Layer::~Layer() {
    #ifdef __CUDA_ARCH__
        delete[] m_neurons;
    #else
        delete[] m_neurons;
    #endif
}

CUDA_CALLABLE_MEMBER void Layer::initNeurons(int n_neurons, int n_weights) {
    // Construir os neurônios dentro do array pré-alocado
    for (int n = 0; n < n_neurons; n++) {
        new (&m_neurons[n]) Neuron(n_weights); // Uso de placement new
    }
}


/* NETWORK */

/*
* Network Constructor
*/
CUDA_CALLABLE_MEMBER Network::Network() {
    // A inicialização do PRNG deve ser gerenciada pelo host ou cuRAND no device
    // std::srand(static_cast<unsigned int>(std::time(nullptr))); // Remover para device

    m_nLayers = 0;
    m_layers = nullptr; // Inicializar ponteiro para nulo
}

/* * Network Destructor
*/
CUDA_CALLABLE_MEMBER Network::~Network() {
    #ifdef __CUDA_ARCH__
        if (m_layers) { // Só liberar se alocado
            for(size_t i = 0; i < m_nLayers; ++i) {
                m_layers[i].~Layer(); // Chamar destrutor de cada Layer
            }
            delete[] m_layers;
        }
    #else
        if (m_layers) {
            delete[] m_layers;
        }
    #endif
}

/* * Initialize a network manually
*/
CUDA_CALLABLE_MEMBER void Network::initialize_network(int n_inputs, int n_hidden, int n_outputs) {
    m_nLayers = 2; // Duas camadas (oculta e saída)
    #ifdef __CUDA_ARCH__
        m_layers = new Layer[m_nLayers];
    #else
        m_layers = new Layer[m_nLayers];
    #endif

    // Adicionar camada oculta
    new (&m_layers[0]) Layer(n_hidden, n_inputs + 1); // Placement new
    
    // Adicionar camada de saída
    new (&m_layers[1]) Layer(n_outputs, n_hidden + 1);
}

/*
* Add another layer to the network
* (Não será usada na inicialização direta da GPU, mas mantida para consistência)
*/
CUDA_CALLABLE_MEMBER void Network::add_layer(int n_neurons, int n_weights) {
    // Isso seria complicado para alocar dinamicamente em um kernel
    // A inicialização é feita diretamente no initialize_network_kernel
}

/* * One forward propagation of an input
*/
CUDA_CALLABLE_MEMBER void Network::forward_propagate(float* inputs, int num_inputs, float* outputs_buffer) {
    float* current_inputs = inputs;
    int current_num_inputs = num_inputs;

    for (size_t i = 0; i < m_nLayers; i++) {
        Layer& current_layer = m_layers[i];
        Neuron* layer_neurons = current_layer.get_neurons();
        int num_neurons_in_layer = current_layer.m_nNeurons; // Acessar o número de neurônios

        // Buffer temporário para as saídas desta camada (se não for a última)
        float temp_outputs[num_neurons_in_layer]; 

        for (int n = 0; n < num_neurons_in_layer; n++) {
            layer_neurons[n].activate(current_inputs, current_num_inputs);
            layer_neurons[n].transfer();
            temp_outputs[n] = layer_neurons[n].get_output();
        }
        current_inputs = temp_outputs; // A próxima camada usa as saídas desta como entrada
        current_num_inputs = num_neurons_in_layer;

        if (i == m_nLayers - 1) { // Se for a última camada, copiar para o buffer final
            for(int k = 0; k < num_neurons_in_layer; ++k) {
                outputs_buffer[k] = temp_outputs[k];
            }
        }
    }
}


/* * Propagate the deviation from an expected output backwards through the network
*/
CUDA_CALLABLE_MEMBER void Network::backward_propagate_error(float* expected, int num_outputs) {
    for (size_t i = m_nLayers; i --> 0;) {
        Layer& current_layer = m_layers[i];
        Neuron* layer_neurons = current_layer.get_neurons();
        int num_neurons_in_layer = current_layer.m_nNeurons;

        for (int n = 0; n < num_neurons_in_layer; n++) {
            float error = 0.0;
            if (i == m_nLayers - 1) { // Camada de saída
                error = expected[n] - layer_neurons[n].get_output();
            } else { // Camadas ocultas
                Layer& next_layer = m_layers[i + 1];
                Neuron* next_layer_neurons = next_layer.get_neurons();
                int num_neurons_in_next_layer = next_layer.m_nNeurons;
                for (int neu_idx = 0; neu_idx < num_neurons_in_next_layer; ++neu_idx) {
                    error += (next_layer_neurons[neu_idx].get_weights()[n] * next_layer_neurons[neu_idx].get_delta());
                }
            }
            layer_neurons[n].set_delta(error * layer_neurons[n].transfer_derivative());
        }
    }
}

/*
* Update weights of a network after an error back propagation
*/
CUDA_CALLABLE_MEMBER void Network::update_weights(float* inputs, int num_original_inputs, float l_rate) {
    for (size_t i = 0; i < m_nLayers; i++) {
        float new_inputs[num_original_inputs]; // Buffer para as entradas da camada atual
        int current_input_size = 0;

        if (i != 0) {
            Layer& prev_layer = m_layers[i-1];
            Neuron* prev_layer_neurons = prev_layer.get_neurons();
            current_input_size = prev_layer.m_nNeurons;
            for (int k = 0; k < current_input_size; ++k) {
                new_inputs[k] = prev_layer_neurons[k].get_output();
            }
        } else {
            // Para a primeira camada, usa as entradas originais
            current_input_size = num_original_inputs;
            for (int k = 0; k < num_original_inputs; ++k) {
                new_inputs[k] = inputs[k];
            }
        }

        Layer& current_layer = m_layers[i];
        Neuron* layer_neurons = current_layer.get_neurons();
        int num_neurons_in_layer = current_layer.m_nNeurons;

        for (int n = 0; n < num_neurons_in_layer; n++) {
            float* weights = layer_neurons[n].get_weights();
            for (int j = 0; j < current_input_size; j++) {
                weights[j] += l_rate * layer_neurons[n].get_delta() * new_inputs[j];
            }
            // Update bias (último peso)
            weights[current_input_size] += l_rate * layer_neurons[n].get_delta();
        }
    }
}

/* * Train the network with trainings data (Este método completo seria chamado do kernel)
* O método `train` no `Network` original opera em `std::vector<std::vector<float>>`,
* que não é adequado para o device. A lógica de treinamento foi transferida para o kernel.
* Por isso, este método original de `train` não pode ser diretamente um `__device__` function.
* As partes internas que usam `Neuron` e `Layer` podem ser `__device__`.
*/
// CUDA_CALLABLE_MEMBER void Network::train(...) {...} // Este método não é diretamente usado


/* * Make a prediction for an input (one forward propagation)
*/
CUDA_CALLABLE_MEMBER int Network::predict(float* input, int num_inputs) {
    float outputs_buffer[m_layers[m_nLayers - 1].m_nNeurons]; // Buffer para as saídas finais
    this->forward_propagate(input, num_inputs, outputs_buffer);

    int max_idx = 0;
    float max_val = outputs_buffer[0];
    for (int i = 1; i < m_layers[m_nLayers - 1].m_nNeurons; ++i) {
        if (outputs_buffer[i] > max_val) {
            max_val = outputs_buffer[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/*
* Display the network in a human readable format
*/
void Network::display_human() {
    // Isso é uma função de debug para o host, não para o device.
    // std::cout << "[Network] (Layers: " << m_nLayers << ")" << std::endl;
    // ... (restante da implementação original)
}