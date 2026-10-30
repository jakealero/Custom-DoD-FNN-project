#pragma once
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <algorithm>

#include "functions.h"

namespace DOD_FNN {

	// ---------------------------------------------------------------- parameters
	struct Layer {
		size_t inputSize = 0, outputSize = 0;
		std::vector<double> biases;   // [outputSize]
		std::vector<double> weights;  // [outputSize * inputSize], row i = neuron i
	};

	struct FNN {
		size_t inputSize = 0, outputSize = 0;
		std::vector<Layer> layers;
	};

	// ------------------------------------------------- per-sample intermediates
	// Slot 0 = network input, slot k+1 = output of layer k.
	// off[k] is where slot k starts inside act / delta.
	//   act[off[k] + j]   : activation j of slot k
	//   delta[off[k] + j] : dLoss/dz for neuron j of layer k-1 (unused for slot 0)
	struct Trace {
		std::vector<size_t> off;
		std::vector<double> act;
		std::vector<double> delta;
	};

	class FNN_OP {
		static void initLayer(Layer& layer, size_t inputSize, size_t outputSize) {
			layer.inputSize = inputSize;
			layer.outputSize = outputSize;
			layer.biases.clear();
			layer.weights.clear();

			for (size_t i = 0; i < outputSize; i++) {
				layer.biases.push_back(random_double());
				for (size_t j = 0; j < inputSize; j++)
					layer.weights.push_back(random_double());
			}
			assert(layer.weights.size() == inputSize * outputSize);
			assert(layer.biases.size() == outputSize);
		}

		template <typename ActivationFn>
		static std::vector<double> Linear(const Layer& layer, const std::vector<double>& input, ActivationFn activate) {
			std::vector<double> output(layer.outputSize);
			for (size_t i = 0; i < layer.outputSize; i++) {
				double value = layer.biases[i];
				const size_t offset = i * layer.inputSize;
				for (size_t j = 0; j < layer.inputSize; j++)
					value += layer.weights[offset + j] * input[j];
				output[i] = activate(value);
			}
			return output;
		}

	public:
		static void Init(FNN& net, size_t inputSize, size_t outputSize, size_t depth, size_t width) {
			if (depth < 2)      throw std::invalid_argument("network depth must be at least 2");
			if (width < 1)      throw std::invalid_argument("network width must be at least 1");
			if (inputSize < 1)  throw std::invalid_argument("network input must be at least 1");
			if (outputSize < 1) throw std::invalid_argument("network output must be at least 1");

			net.inputSize = inputSize;
			net.outputSize = outputSize;
			net.layers.resize(depth);

			initLayer(net.layers[0], inputSize, width);
			for (size_t d = 1; d < depth - 1; d++)
				initLayer(net.layers[d], width, width);
			initLayer(net.layers[depth - 1], width, outputSize);
		}

		// Allocate the Trace ONCE; reuse it for every sample.
		static void InitTrace(const FNN& net, Trace& t) {
			const size_t depth = net.layers.size();
			t.off.assign(depth + 1, 0);

			size_t total = net.inputSize;                // slot 0
			for (size_t l = 0; l < depth; l++) {
				t.off[l + 1] = total;                    // slot l+1 starts here
				total += net.layers[l].outputSize;
			}
			t.act.assign(total, 0.0);
			t.delta.assign(total, 0.0);
		}

		// Inference only (allocates, no trace).
		template <typename ActivationFn>
		static std::vector<double> Forward(const FNN& net, const std::vector<double>& input, ActivationFn activate) {
			assert(net.inputSize == input.size());
			std::vector<double> output = input;
			for (size_t i = 0; i < net.layers.size(); i++)
				output = Linear(net.layers[i], output, activate);
			return output;
		}

		// ---- Phase 1: forward, recording activations into the trace ------------
		template <typename ActivationFn>
		static void Forward(const FNN& net, const std::vector<double>& input, Trace& t, ActivationFn activate) {
			assert(input.size() == net.inputSize);
			std::copy(input.begin(), input.end(), t.act.begin());

			for (size_t l = 0; l < net.layers.size(); l++) {
				const Layer& L = net.layers[l];
				const double* in = &t.act[t.off[l]];
				double* out = &t.act[t.off[l + 1]];

				for (size_t i = 0; i < L.outputSize; i++) {
					const double* w = &L.weights[i * L.inputSize];
					double z = L.biases[i];
					for (size_t j = 0; j < L.inputSize; j++)
						z += w[j] * in[j];
					out[i] = activate(z);
				}
			}
		}

		// MSE of the last forward pass.
		static double Loss(const Trace& t, const std::vector<double>& target) {
			const double* a = &t.act[t.off.back()];
			double sum = 0.0;
			for (size_t j = 0; j < target.size(); j++) {
				const double d = a[j] - target[j];
				sum += d * d;
			}
			return sum / target.size();
		}

		// ---- Phase 2: backward, filling deltas (reads weights, writes only delta)
		template <typename DerivFromOutput>
		static void Backward(const FNN& net, const std::vector<double>& target, Trace& t, DerivFromOutput dact) {
			const size_t last = net.layers.size() - 1;

			// output layer: dL/dz = dL/da * da/dz
			{
				const size_t n = net.layers[last].outputSize;
				const double* a = &t.act[t.off[last + 1]];
				double* d = &t.delta[t.off[last + 1]];
				for (size_t j = 0; j < n; j++)
					d[j] = 2.0 * (a[j] - target[j]) / n * dact(a[j]);
			}

			// walk back: layer l's deltas -> layer l-1's deltas
			for (size_t l = last; l > 0; l--) {
				const Layer& L = net.layers[l];
				const double* dl = &t.delta[t.off[l + 1]];
				const double* aPrev = &t.act[t.off[l]];
				double* dPrev = &t.delta[t.off[l]];

				std::fill(dPrev, dPrev + L.inputSize, 0.0);
				for (size_t i = 0; i < L.outputSize; i++) {
					const double* w = &L.weights[i * L.inputSize];
					for (size_t j = 0; j < L.inputSize; j++)
						dPrev[j] += w[j] * dl[i];             // W^T * delta
				}
				for (size_t j = 0; j < L.inputSize; j++)
					dPrev[j] *= dact(aPrev[j]);
			}
		}

		// ---- Phase 3: update (reads act + delta, writes weights) ---------------
		static void Update(FNN& net, const Trace& t, double learningRate) {
			for (size_t l = 0; l < net.layers.size(); l++) {
				Layer& L = net.layers[l];
				const double* in = &t.act[t.off[l]];
				const double* d = &t.delta[t.off[l + 1]];

				for (size_t i = 0; i < L.outputSize; i++) {
					L.biases[i] -= learningRate * d[i];
					double* w = &L.weights[i * L.inputSize];
					for (size_t j = 0; j < L.inputSize; j++)
						w[j] -= learningRate * d[i] * in[j];
				}
			}
		}

		// Convenience: one SGD step on one sample, returns the loss.
		static double Train(FNN& net, Trace& t, const std::vector<double>& input,
			const std::vector<double>& target, double learningRate) {
			Forward(net, input, t, sigmoid);
			const double loss = Loss(t, target);
			Backward(net, target, t, sigmoid_derivative_from_output);
			Update(net, t, learningRate);
			return loss;
		}
	};

} // namespace DOD_FNN