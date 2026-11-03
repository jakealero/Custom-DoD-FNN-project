// LSTM correctness tests (no framework; exits non-zero on failure).
//  1. gradient check: analytic (Backward+Update) vs central finite differences, every parameter
//  2. streaming Step()/Predict() is bit-identical to the trace-recording Forward()
//  3. a Trace sized for T steps can be reused for shorter sequences
//  4. save/load roundtrip is exact; corrupt files are rejected
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "LSTM.h"
#include "LSTM_io.h"
#include "functions.h"

using namespace DOD_LSTM;

static int g_failed = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_failed++; std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static double lossOf(const LSTM& net, Trace& t, const std::vector<double>& x, size_t steps,
	const std::vector<double>& tgt, LossMode mode) {
	LSTM_OP::Forward(net, x.data(), steps, t);
	return LSTM_OP::Loss(net, t, tgt.data(), mode);
}

static void gradCheck(size_t in, size_t out, size_t layers, size_t hidden, size_t steps,
	HeadActivation ha, LossMode mode) {
	std::srand(7 + (unsigned)(layers * 13 + hidden * 7 + steps));
	LSTM net;
	LSTM_OP::Init(net, in, out, layers, hidden, ha);
	for (double& p : net.params) p += 0.4 * DOD::random_double();      // move away from the special init

	std::vector<double> x(steps * in), tgt((mode == LossMode::LastStep ? 1 : steps) * out);
	for (double& v : x) v = DOD::random_double();
	for (double& v : tgt) v = (ha == HeadActivation::Sigmoid) ? 0.5 + 0.5 * DOD::random_double() : DOD::random_double();

	Trace t;
	LSTM_OP::InitTrace(net, t, steps + 2);                              // capacity > steps on purpose

	// analytic gradient: run one update with lr = 1, so grad = before - after
	LSTM upd = net;
	Trace t2;
	LSTM_OP::InitTrace(upd, t2, steps + 2);
	LSTM_OP::Forward(upd, x.data(), steps, t2);
	LSTM_OP::Backward(upd, tgt.data(), t2, mode);
	LSTM_OP::Update(upd, x.data(), t2, 1.0);

	const double eps = 1e-6;
	double worst = 0.0;
	size_t worstIdx = 0, bad = 0;
	for (size_t i = 0; i < net.params.size(); i++) {
		const double orig = net.params[i];
		net.params[i] = orig + eps;  const double lp = lossOf(net, t, x, steps, tgt, mode);
		net.params[i] = orig - eps;  const double lm = lossOf(net, t, x, steps, tgt, mode);
		net.params[i] = orig;

		const double numeric = (lp - lm) / (2 * eps);
		const double analytic = net.params[i] - upd.params[i];
		const double err = std::fabs(numeric - analytic);
		const double tol = 1e-7 + 1e-5 * std::fmax(std::fabs(numeric), std::fabs(analytic));
		if (err > tol) bad++;
		if (err / tol > worst) { worst = err / tol; worstIdx = i; }
	}
	std::printf("  grad  in=%zu out=%zu layers=%zu H=%zu T=%zu %-7s %-8s  params=%4zu  worst err/tol=%.3f (idx %zu)\n",
		in, out, layers, hidden, steps,
		ha == HeadActivation::Sigmoid ? "sigmoid" : "linear",
		mode == LossMode::LastStep ? "LastStep" : "AllSteps",
		net.params.size(), worst, worstIdx);
	CHECK(bad == 0, "%zu parameters have a wrong gradient", bad);
}

static void streamingMatchesForward() {
	std::srand(11);
	bool ok = true;
	for (size_t layers = 1; layers <= 3; layers++) for (size_t H : {1, 4, 7}) for (size_t out : {1, 3}) {
		LSTM net; LSTM_OP::Init(net, 3, out, layers, H, HeadActivation::Sigmoid);
		for (double& p : net.params) p += 0.3 * DOD::random_double();
		const size_t T = 6;
		std::vector<double> x(T * 3);
		for (double& v : x) v = DOD::random_double();

		Trace t; LSTM_OP::InitTrace(net, t, T);
		LSTM_OP::Forward(net, x.data(), T, t);

		State s; LSTM_OP::InitState(net, s);
		LSTM_OP::Reset(s);
		for (size_t i = 0; i < T; i++) {
			const double* y = LSTM_OP::Step(net, x.data() + i * 3, s);
			const double* r = LSTM_OP::Output(net, t, i);
			for (size_t k = 0; k < out; k++) if (y[k] != r[k]) ok = false;          // every step, bit-exact
		}
		const double* p = LSTM_OP::Predict(net, x.data(), T, s);                      // Predict resets itself
		const double* r = LSTM_OP::Output(net, t, T - 1);
		for (size_t k = 0; k < out; k++) if (p[k] != r[k]) ok = false;
	}
	std::printf("  Step/Predict vs Forward over 18 shapes: %s\n", ok ? "bit-identical" : "MISMATCH");
	CHECK(ok, "streaming inference differs from Forward");
}

static void traceReuse() {
	std::srand(5);
	LSTM net; LSTM_OP::Init(net, 2, 1, 2, 5, HeadActivation::Linear);
	std::vector<double> x(10 * 2);
	for (double& v : x) v = DOD::random_double();

	Trace big; LSTM_OP::InitTrace(net, big, 10);
	Trace exact; LSTM_OP::InitTrace(net, exact, 4);
	LSTM_OP::Forward(net, x.data(), 10, big);       // dirty the big trace with a long sequence first
	LSTM_OP::Forward(net, x.data(), 4, big);        // then reuse it for a short one
	LSTM_OP::Forward(net, x.data(), 4, exact);
	bool ok = true;
	for (size_t s = 0; s < 4; s++) if (LSTM_OP::Output(net, big, s)[0] != LSTM_OP::Output(net, exact, s)[0]) ok = false;
	std::printf("  Trace reuse (10-step capacity run at 4 steps): %s\n", ok ? "identical to exact-size trace" : "MISMATCH");
	CHECK(ok, "reused trace gives different results");
}

static void fileIO() {
	std::srand(3);
	LSTM net; LSTM_OP::Init(net, 3, 2, 2, 6, HeadActivation::Sigmoid);
	for (double& p : net.params) p += 0.2 * DOD::random_double();
	io::Save(net, "t_lstm.dlsm");

	LSTM back; io::Load(back, "t_lstm.dlsm");
	const bool same = back.params == net.params && back.depth() == net.depth() &&
		back.inputSize == net.inputSize && back.outputSize == net.outputSize &&
		back.headAct == net.headAct && back.head.weightOff == net.head.weightOff;
	std::printf("  save/load roundtrip: %s\n", same ? "exact" : "MISMATCH");
	CHECK(same, "roundtrip differs");

	// corruption: every one of these must throw and leave `victim` untouched
	std::vector<char> good;
	{ std::ifstream f("t_lstm.dlsm", std::ios::binary); good.assign(std::istreambuf_iterator<char>(f), {}); }
	auto tryLoad = [&](const char* name, std::vector<char> bytes) {
		{ std::ofstream f("t_bad.dlsm", std::ios::binary); f.write(bytes.data(), (std::streamsize)bytes.size()); }
		LSTM victim = net; bool threw = false;
		try { io::Load(victim, "t_bad.dlsm"); }
		catch (const std::exception&) { threw = true; }
		const bool untouched = victim.params == net.params;
		std::printf("  corrupt file (%-18s): %s\n", name, threw && untouched ? "rejected, net untouched" : "NOT REJECTED");
		CHECK(threw && untouched, "%s was not rejected cleanly", name);
		};
	tryLoad("truncated", std::vector<char>(good.begin(), good.begin() + 100));
	tryLoad("junk", std::vector<char>{'h', 'e', 'l', 'l', 'o'});
	{ auto b = good; b[8 + 8] = 0x7f; b[8 + 15] = 0x7f; tryLoad("huge input size", b); }     // inputSize field
	{ auto b = good; b[8 + 8 * 3] = 9;  tryLoad("bad head activation", b); }
	{ auto b = good; b.push_back(0);    tryLoad("trailing byte", b); }
	{
		LSTM n; bool threw = false; try { io::Load(n, "does_not_exist.dlsm"); }
		catch (const std::exception&) { threw = true; }
		std::printf("  missing file: %s\n", threw ? "rejected" : "NOT REJECTED"); CHECK(threw, "missing file");
	}
}

/*int main() {
	std::printf("[gradient check]\n");
	gradCheck(2, 2, 1, 3, 5, HeadActivation::Linear, LossMode::LastStep);
	gradCheck(2, 2, 1, 3, 5, HeadActivation::Sigmoid, LossMode::LastStep);
	gradCheck(2, 3, 1, 4, 6, HeadActivation::Linear, LossMode::AllSteps);
	gradCheck(3, 2, 2, 3, 5, HeadActivation::Sigmoid, LossMode::AllSteps);
	gradCheck(2, 1, 3, 4, 7, HeadActivation::Linear, LossMode::LastStep);
	gradCheck(1, 1, 2, 1, 4, HeadActivation::Sigmoid, LossMode::AllSteps);     // hidden size 1
	gradCheck(4, 2, 1, 5, 1, HeadActivation::Linear, LossMode::LastStep);     // single timestep
	std::printf("[inference]\n");
	streamingMatchesForward();
	traceReuse();
	std::printf("[file I/O]\n");
	fileIO();

	std::printf("\n%s (%d failed checks)\n", g_failed ? "TESTS FAILED" : "ALL TESTS PASSED", g_failed);
	return g_failed ? 1 : 0;
}*/