#include <iostream>
#include <vector>

#include "FNN.h"
#include "FNN_IO.h"

int main() {
	std::srand(0);

	DOD_FNN::FNN net;
	DOD_FNN::Trace trace;
	DOD_FNN::FNN_OP::Init(net, 2, 1, 3, 4);     // 2 in, 1 out, 3 layers, width 4
	DOD_FNN::FNN_OP::InitTrace(net, trace);     // allocated once

	const std::vector<std::vector<double>> x = { {0,0}, {0,1}, {1,0}, {1,1} };
	const std::vector<std::vector<double>> y = { {0},   {1},   {1},   {0} };   // XOR

	for (int epoch = 0; epoch < 20000; epoch++) {
		double total = 0;
		for (size_t s = 0; s < x.size(); s++)
			total += DOD_FNN::FNN_OP::Train(net, trace, x[s], y[s], 0.5);
		if (epoch % 2000 == 0)
			std::cout << "epoch " << epoch << "  loss " << total / x.size() << "\n";
	}

	for (size_t s = 0; s < x.size(); s++) {
		auto out = DOD_FNN::FNN_OP::Forward(net, x[s], DOD_FNN::sigmoid);
		std::cout << x[s][0] << "," << x[s][1] << " -> " << out[0] << "\n";
	}

	// ---- save, then load into a fresh network and check it behaves identically
	DOD_FNN::io::Save(net, "xor.dfnn");

	DOD_FNN::FNN loaded;
	DOD_FNN::io::Load(loaded, "xor.dfnn");

	bool same = true;
	for (size_t s = 0; s < x.size(); s++) {
		double a = DOD_FNN::FNN_OP::Forward(net, x[s], DOD_FNN::sigmoid)[0];
		double b = DOD_FNN::FNN_OP::Forward(loaded, x[s], DOD_FNN::sigmoid)[0];
		if (a != b) same = false;
	}
	std::cout << "save/load roundtrip " << (same ? "OK (bit-identical)" : "MISMATCH") << "\n";
}