#include <iostream>
#include <vector>
#include <stdexcept>
#include <assert.h>

#include "functions.cpp"

using namespace std;

namespace DOD_FNN {
	struct Layer {
		size_t inputSize, outputSize;
		vector<double> biases;
		vector<double> weights;
	};

	struct FNN {
		size_t inputSize, outputSize;
		vector<Layer> layers;
	};


	class FNN_OP {
		static void initLayer(Layer& layer, size_t inputSize, size_t outputSize) {
			layer.inputSize = inputSize;
			layer.outputSize = outputSize;

			for (int i = 0; i < outputSize; i++) {
				double paramBias = random_double();
				layer.biases.push_back(paramBias);


				for (int j = 0; j < inputSize; j++) {
					double paramWeight = random_double();
					layer.weights.push_back(paramWeight);
				}

			}

			assert(layer.weights.size() == inputSize * outputSize);
			assert(layer.biases.size() == outputSize);
		}

		template <typename ActivationFn>
		static vector<double> Linear(const Layer& layer, const vector<double>& input, ActivationFn activate) {
			vector<double> output(layer.outputSize);

			for (int i = 0; i < layer.outputSize; i++) {
				double value = layer.biases[i];
				const int offset = i * layer.inputSize;

				for (int j = 0; j < layer.inputSize; j++) {
					value += layer.weights[offset + j] * input[j];
				}

				output[i] = activate(value);
			}

			return output;
		}
	public:

		static void Init(FNN& net, size_t inputSize, size_t outputSize, size_t depth, size_t width) {
			if (depth < 2) {
				throw invalid_argument("network depth must be more than 2");
			}

			if (width < 1) {
				throw invalid_argument("network width must be more than 1");
			}

			if (inputSize < 1) {
				throw invalid_argument("network input must be more than 1!");
			}

			if (outputSize < 1) {
				throw invalid_argument("network output must be more than 1!");
			}

			net.inputSize = inputSize;
			net.outputSize = outputSize;

			net.layers.resize(depth);

			//for the input layer
			initLayer(net.layers[0], inputSize, width);

			//for the hidden layers
			for (int d = 1; d < depth - 1; d++) {

				initLayer(net.layers[d], width, width);

			}

			//for the output layer
			initLayer(net.layers[depth - 1], width, outputSize);
		}

		template <typename ActivationFn>
		static vector<double> Forward(const FNN& net, const vector<double>& input, ActivationFn ActivationFunction) {
			assert(net.inputSize == input.size());
			vector<double> output = input;
			for (int i = 0; i < net.layers.size(); i++) {
				output = Linear(net.layers[i], output, ActivationFunction);
			}
			return output;
		}
	};
}