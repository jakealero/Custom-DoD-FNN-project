// Memory task: read a sequence of +-1 values, then say whether the FIRST one was +1.
// The answer is only available by remembering it across the whole sequence.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "LSTM.h"
#include "LSTM_io.h"

using namespace DOD_LSTM;

static const size_t SEQ_LEN = 8;

static void makeSequence(std::vector<double>& x, double& target) {
	for (size_t t = 0; t < SEQ_LEN; t++) x[t] = (std::rand() & 1) ? 1.0 : -1.0;
	target = x[0] > 0 ? 1.0 : 0.0;
}

/*int main() {
	std::srand(1);

	LSTM net;
	Trace trace;
	State state;
	LSTM_OP::Init(net, 1, 1, 1, 8, HeadActivation::Sigmoid);
	LSTM_OP::InitTrace(net, trace, SEQ_LEN);     // allocated once
	LSTM_OP::InitState(net, state);              // allocated once

	std::printf("parameters: %zu doubles (%zu bytes)\n", net.params.size(), net.params.size() * sizeof(double));

	std::vector<double> x(SEQ_LEN);
	double target = 0;
	double running = 0.25;
	for (int step = 1; step <= 6000; step++) {
		makeSequence(x, target);
		const double loss = LSTM_OP::Train(net, trace, x.data(), SEQ_LEN, &target, LossMode::LastStep, 0.5);
		running = 0.99 * running + 0.01 * loss;
		if (step % 1000 == 0) std::printf("sequence %5d  running loss %.5f\n", step, running);
	}

	// evaluate on fresh sequences with the parameters-only inference path
	io::Save(net, "echo.dlsm");
	LSTM loaded;
	io::Load(loaded, "echo.dlsm");
	State s2;
	LSTM_OP::InitState(loaded, s2);

	int correct = 0;
	const int N = 1000;
	for (int i = 0; i < N; i++) {
		makeSequence(x, target);
		const double p = LSTM_OP::Predict(loaded, x.data(), SEQ_LEN, s2)[0];
		if ((p > 0.5) == (target > 0.5)) correct++;
	}
	std::printf("accuracy on %d fresh sequences (loaded model, Predict): %.1f%%\n", N, 100.0 * correct / N);
}*/
