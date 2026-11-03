#pragma once
// Shared, header-only helpers. Used by BOTH models; depends on neither.
#include <cmath>
#include <cstdlib>

namespace DOD {

inline double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

// Derivative written in terms of the OUTPUT a = sigmoid(z),
// so the pre-activation z never has to be stored.
inline double sigmoid_derivative_from_output(double a) {
	return a * (1.0 - a);
}

inline double random_double() {            // uniform in [-1, 1]
	return (double)std::rand() / RAND_MAX * 2.0 - 1.0;
}

} // namespace DOD
