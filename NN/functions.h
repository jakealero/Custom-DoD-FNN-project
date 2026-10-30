#pragma once
#include <cmath>
#include <cstdlib>

namespace DOD_FNN {

	inline double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	// Derivative written in terms of the OUTPUT a = sigmoid(z).
	// This means we never need to store z, only the activations.
	inline double sigmoid_derivative_from_output(double a) {
		return a * (1.0 - a);
	}

	inline double random_double() {
		return (double)std::rand() / RAND_MAX * 2.0 - 1.0;
	}

} // namespace DOD_FNN