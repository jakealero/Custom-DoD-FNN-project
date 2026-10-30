#include <iostream>
#include <vector>
#include <stdexcept>
#include <assert.h>

#include "FNN.cpp"

using namespace std;

int main() {
	DOD_FNN::FNN net;

	DOD_FNN::FNN_OP::Init(net, 2, 1, 2, 1);
	vector<double> output = DOD_FNN::FNN_OP::Forward(net, {1,1 }, DOD_FNN::sigmoid);

	cout << output[0] << endl;
}