#pragma once
#ifdef DOD_NO_FNN
#error "DOD_FNN is disabled in this build of dodnn (configure with -DDODNN_ENABLE_FNN=ON)."
#endif

#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <algorithm>

#include "functions.h"

namespace DOD_FNN {

using DOD::sigmoid;
using DOD::sigmoid_derivative_from_output;
using DOD::random_double;

// ---------------------------------------------------------------- parameters
// Where one layer lives inside FNN::params. Pure metadata: no weights here.
//   params[biasOff   + i]                 = bias of neuron i
//   params[weightOff + i * inputSize + j] = weight from input j to neuron i
struct LayerMeta {
	size_t inputSize = 0, outputSize = 0;
	size_t biasOff = 0, weightOff = 0;
};

// params layout:  [ b0 | W0 | b1 | W1 | ... | b(L-1) | W(L-1) ]
// One contiguous block of doubles; `meta` is a tiny table describing it.
struct FNN {
	size_t inputSize = 0, outputSize = 0;
	std::vector<LayerMeta> meta;     // [layerCount]
	std::vector<double> params;      // every bias and weight

	size_t depth() const { return meta.size(); }

	double*       biases (size_t l)       { return params.data() + meta[l].biasOff;   }
	const double* biases (size_t l) const { return params.data() + meta[l].biasOff;   }
	double*       weights(size_t l)       { return params.data() + meta[l].weightOff; }
	const double* weights(size_t l) const { return params.data() + meta[l].weightOff; }
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

// Inference-only workspace: two ping-pong buffers, each as wide as the widest
// layer output. No delta, no offsets, no history.
struct Scratch {
	std::vector<double> bufA, bufB;
};

class FNN_OP {
	template <typename ActivationFn>
	static std::vector<double> Linear(const FNN& net, size_t l, const std::vector<double>& input, ActivationFn activate) {
		const LayerMeta& M = net.meta[l];
		const double* b = net.biases(l);
		const double* W = net.weights(l);

		std::vector<double> output(M.outputSize);
		for (size_t i = 0; i < M.outputSize; i++) {
			const double* w = W + i * M.inputSize;
			double value = b[i];
			for (size_t j = 0; j < M.inputSize; j++)
				value += w[j] * input[j];
			output[i] = activate(value);
		}
		return output;
	}

public:
	// ---- compiled into the library (FNN.cpp) -------------------------------

	// sizes = { n0, n1, ..., nL }: input size followed by every layer's output size.
	// Builds the metadata table and allocates ONE zeroed params block.
	static void Layout(FNN& net, const std::vector<size_t>& sizes);

	static void Init(FNN& net, size_t inputSize, size_t outputSize, size_t depth, size_t width);

	static void InitTrace(const FNN& net, Trace& t);        // allocate once, reuse
	static void InitScratch(const FNN& net, Scratch& s);    // allocate once, reuse

	static double Loss(const Trace& t, const std::vector<double>& target);   // MSE of last Forward
	static void Update(FNN& net, const Trace& t, double learningRate);       // phase 3

	// One SGD step on one sample (sigmoid network), returns the loss.
	static double Train(FNN& net, Trace& t, const std::vector<double>& input,
	                    const std::vector<double>& target, double learningRate);

	// ---- templates (activation is chosen by the caller, so they live here) -

	// Inference using ONLY the parameters (+ two small buffers). No allocation.
	// Returns a pointer to the final layer's outputs (net.outputSize values),
	// which lives inside `s` and stays valid until the next Predict call.
	template <typename ActivationFn>
	static const double* Predict(const FNN& net, const double* input, Scratch& s, ActivationFn activate) {
		const double* in = input;
		double* out = s.bufA.data();

		for (size_t l = 0; l < net.depth(); l++) {
			const LayerMeta& M = net.meta[l];
			const double* b = net.biases(l);
			const double* W = net.weights(l);
			for (size_t i = 0; i < M.outputSize; i++) {
				const double* w = W + i * M.inputSize;
				double z = b[i];
				for (size_t j = 0; j < M.inputSize; j++)
					z += w[j] * in[j];
				out[i] = activate(z);
			}
			in = out;                                                        // this layer's output is the next input
			out = (out == s.bufA.data()) ? s.bufB.data() : s.bufA.data();   // swap buffers
		}
		return in;
	}

	template <typename ActivationFn>
	static const double* Predict(const FNN& net, const std::vector<double>& input, Scratch& s, ActivationFn activate) {
		assert(input.size() == net.inputSize);
		return Predict(net, input.data(), s, activate);
	}

	// Inference only (allocates, no trace).
	template <typename ActivationFn>
	static std::vector<double> Forward(const FNN& net, const std::vector<double>& input, ActivationFn activate) {
		assert(net.inputSize == input.size());
		std::vector<double> output = input;
		for (size_t l = 0; l < net.depth(); l++)
			output = Linear(net, l, output, activate);
		return output;
	}

	// ---- Phase 1: forward, recording activations into the trace ------------
	template <typename ActivationFn>
	static void Forward(const FNN& net, const std::vector<double>& input, Trace& t, ActivationFn activate) {
		assert(input.size() == net.inputSize);
		std::copy(input.begin(), input.end(), t.act.begin());

		for (size_t l = 0; l < net.depth(); l++) {
			const LayerMeta& M = net.meta[l];
			const double* b = net.biases(l);
			const double* W = net.weights(l);
			const double* in = &t.act[t.off[l]];
			double* out = &t.act[t.off[l + 1]];

			for (size_t i = 0; i < M.outputSize; i++) {
				const double* w = W + i * M.inputSize;
				double z = b[i];
				for (size_t j = 0; j < M.inputSize; j++)
					z += w[j] * in[j];
				out[i] = activate(z);
			}
		}
	}

	// ---- Phase 2: backward, filling deltas (reads weights, writes only delta)
	template <typename DerivFromOutput>
	static void Backward(const FNN& net, const std::vector<double>& target, Trace& t, DerivFromOutput dact) {
		const size_t last = net.depth() - 1;

		// output layer: dL/dz = dL/da * da/dz
		{
			const size_t n = net.meta[last].outputSize;
			const double* a = &t.act[t.off[last + 1]];
			double* d = &t.delta[t.off[last + 1]];
			for (size_t j = 0; j < n; j++)
				d[j] = 2.0 * (a[j] - target[j]) / n * dact(a[j]);
		}

		// walk back: layer l's deltas -> layer l-1's deltas
		for (size_t l = last; l > 0; l--) {
			const LayerMeta& M = net.meta[l];
			const double* W = net.weights(l);
			const double* dl = &t.delta[t.off[l + 1]];
			const double* aPrev = &t.act[t.off[l]];
			double* dPrev = &t.delta[t.off[l]];

			std::fill(dPrev, dPrev + M.inputSize, 0.0);
			for (size_t i = 0; i < M.outputSize; i++) {
				const double* w = W + i * M.inputSize;
				for (size_t j = 0; j < M.inputSize; j++)
					dPrev[j] += w[j] * dl[i];             // W^T * delta
			}
			for (size_t j = 0; j < M.inputSize; j++)
				dPrev[j] *= dact(aPrev[j]);
		}
	}
};

} // namespace DOD_FNN
