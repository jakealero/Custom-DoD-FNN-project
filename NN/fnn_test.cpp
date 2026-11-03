// FNN tests (no framework; exits non-zero on failure).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "FNN.h"
#include "FNN_io.h"

using namespace DOD_FNN;

static int g_failed = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_failed++; std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
	std::srand(0);
	FNN net; Trace trace; Scratch scratch;
	FNN_OP::Init(net, 2, 1, 3, 4);
	FNN_OP::InitTrace(net, trace);
	FNN_OP::InitScratch(net, scratch);

	const std::vector<std::vector<double>> x = { {0,0}, {0,1}, {1,0}, {1,1} };
	const std::vector<std::vector<double>> y = { {0},   {1},   {1},   {0} };      // XOR
	for (int epoch = 0; epoch < 20000; epoch++)
		for (size_t s = 0; s < x.size(); s++) FNN_OP::Train(net, trace, x[s], y[s], 0.5);

	bool learned = true, same = true;
	for (size_t s = 0; s < x.size(); s++) {
		const double p = FNN_OP::Predict(net, x[s], scratch, sigmoid)[0];
		const double f = FNN_OP::Forward(net, x[s], sigmoid)[0];
		if (std::fabs(p - y[s][0]) > 0.05) learned = false;
		if (p != f) same = false;
	}
	std::printf("  XOR learned (all outputs within 0.05): %s\n", learned ? "yes" : "NO");
	std::printf("  Predict == Forward (bit-identical):   %s\n", same ? "yes" : "NO");
	CHECK(learned, "XOR not learned");
	CHECK(same, "Predict differs from Forward");

	io::Save(net, "t_fnn.dfnn");
	FNN back; io::Load(back, "t_fnn.dfnn");
	const bool exact = back.params == net.params && back.depth() == net.depth();
	std::printf("  save/load roundtrip exact:            %s\n", exact ? "yes" : "NO");
	CHECK(exact, "roundtrip differs");

	{ std::ofstream f("t_fnn.dfnn", std::ios::binary | std::ios::app); f.put('x'); }
	bool threw = false;
	FNN victim = net;
	try { io::Load(victim, "t_fnn.dfnn"); } catch (const std::exception&) { threw = true; }
	std::printf("  trailing byte rejected, net untouched: %s\n", (threw && victim.params == net.params) ? "yes" : "NO");
	CHECK(threw && victim.params == net.params, "trailing byte accepted");

	std::printf("\n%s (%d failed checks)\n", g_failed ? "TESTS FAILED" : "ALL TESTS PASSED", g_failed);
	return g_failed ? 1 : 0;
}
