#pragma once

#include <iostream>
#include <vector>

// Adicionando qualificador para funções que podem ser chamadas do host e do device
#define CUDA_CALLABLE_MEMBER __host__ __device__

class Neuron {
public:
	CUDA_CALLABLE_MEMBER Neuron(int n_weights);
	CUDA_CALLABLE_MEMBER ~Neuron();

	CUDA_CALLABLE_MEMBER void activate(float* inputs, int num_inputs); // Alterado para array C-style
	CUDA_CALLABLE_MEMBER void transfer();
	CUDA_CALLABLE_MEMBER float transfer_derivative() { return static_cast<float>(m_output * (1.0 - m_output));  };

	// return mutable reference to the neuron weights - em CUDA, acesso direto via ponteiro
	CUDA_CALLABLE_MEMBER float* get_weights(void) { return m_weights; }; // Alterado para ponteiro puro

	CUDA_CALLABLE_MEMBER float get_output(void) { return m_output; };
	CUDA_CALLABLE_MEMBER float get_activation(void) { return m_activation; };
	CUDA_CALLABLE_MEMBER float get_delta(void) { return m_delta; };
	
	CUDA_CALLABLE_MEMBER void set_output(float o) { m_output = o; }
	CUDA_CALLABLE_MEMBER void set_delta(float delta) { m_delta = delta; };

private:
	size_t m_nWeights;
	float* m_weights; // Alterado para ponteiro puro para alocação em GPU
	float m_activation;
	float m_output;
	float m_delta;

private:
	CUDA_CALLABLE_MEMBER void initWeights(int n_weights);
};

class Layer {
public:
	CUDA_CALLABLE_MEMBER Layer(int n_neurons, int n_weights);
	CUDA_CALLABLE_MEMBER ~Layer();

	// return mutable reference to the neurons - em CUDA, acesso direto via ponteiro
	CUDA_CALLABLE_MEMBER Neuron* get_neurons(void) { return m_neurons; }; // Alterado para ponteiro puro

private:
	CUDA_CALLABLE_MEMBER void initNeurons(int n_neurons, int n_weights);

	Neuron* m_neurons; // Alterado para ponteiro puro
	size_t m_nNeurons; // Adicionado para rastrear o número de neurônios
};

class Network {
public:
	CUDA_CALLABLE_MEMBER Network();
	CUDA_CALLABLE_MEMBER ~Network();

	CUDA_CALLABLE_MEMBER void initialize_network(int n_inputs, int n_hidden, int n_outputs);

	CUDA_CALLABLE_MEMBER void add_layer(int n_neurons, int n_weights);
	// Funções que operam em dados de GPU precisam de ponteiros
	CUDA_CALLABLE_MEMBER void forward_propagate(float* inputs, int num_inputs, float* outputs_buffer);
	CUDA_CALLABLE_MEMBER void backward_propagate_error(float* expected, int num_outputs);
	CUDA_CALLABLE_MEMBER void update_weights(float* inputs, int num_inputs, float l_rate);
	// O método train precisaria ser refeito para a GPU ou chamado de um kernel para cada rede
	// Por simplicidade, farei a chamada do host para cada rede em um loop CUDA
	// CUDA_CALLABLE_MEMBER void train(float* trainings_data, float l_rate, size_t n_epoch, size_t n_outputs, int id);
	CUDA_CALLABLE_MEMBER int predict(float* input, int num_inputs);

	// Display é uma função de debug, não é CUDA_CALLABLE_MEMBER
	void display_human();

private:
	size_t m_nLayers;
	Layer* m_layers; // Alterado para ponteiro puro
};