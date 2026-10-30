#include <iostream>
#include <vector>
#include <stdexcept>
#include <assert.h>

using namespace std;

namespace DOD_FNN {
	//sigmoid activation function

	inline double sigmoid(double x) {
		return 1 / (1 + exp(-x));
	};


	//sigmoid derivative function

	inline double sigmoid_derivative(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	};

	//random

	inline double random_double() {
		return (double)rand() / RAND_MAX * 2.f - 1.f;
	};
}