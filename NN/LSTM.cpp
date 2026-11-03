#include "LSTM.h"
#include "LSTM_io.h"
#include "functions.h"
#include "io_common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace DOD_LSTM {

using DOD::sigmoid;
using DOD::random_double;

namespace {

// Gate pre-activations -> post-activations for ONE timestep of ONE layer.
// Summation order is fixed (bias, then Wx.x, then Wh.h) so the trace-recording
// forward pass and the streaming Step() produce bit-identical numbers.
inline void computeGates(const LayerMeta& M, const double* b, const double* Wx, const double* Wh,
                         const double* in, const double* hPrev, double* gate) {
	const size_t H = M.hiddenSize, I = M.inputSize;
	for (size_t r = 0; r < 4 * H; r++) {
		double z = b[r];
		const double* wx = Wx + r * I;
		for (size_t m = 0; m < I; m++) z += wx[m] * in[m];
		const double* wh = Wh + r * H;
		for (size_t k = 0; k < H; k++) z += wh[k] * hPrev[k];
		gate[r] = (r >= 2 * H && r < 3 * H) ? std::tanh(z) : sigmoid(z);
	}
}

// c_t = f*c_(t-1) + i*g ; h_t = o*tanh(c_t).   cPrev may alias c (in-place streaming).
inline void cellUpdate(size_t H, const double* gate, const double* cPrev, double* c, double* tanhcOut, double* h) {
	for (size_t j = 0; j < H; j++) {
		const double cn = gate[H + j] * cPrev[j] + gate[j] * gate[2 * H + j];
		const double tc = std::tanh(cn);
		c[j] = cn;
		if (tanhcOut) tanhcOut[j] = tc;
		h[j] = gate[3 * H + j] * tc;
	}
}

inline void headForward(const HeadMeta& HM, HeadActivation act, const double* bo, const double* Wo,
                        const double* h, double* y) {
	for (size_t k = 0; k < HM.outputSize; k++) {
		double z = bo[k];
		const double* w = Wo + k * HM.inputSize;
		for (size_t j = 0; j < HM.inputSize; j++) z += w[j] * h[j];
		y[k] = (act == HeadActivation::Sigmoid) ? sigmoid(z) : z;
	}
}

} // namespace

// =============================================================== setup

void LSTM_OP::Layout(LSTM& net, size_t inputSize, const std::vector<size_t>& hidden,
                     size_t outputSize, HeadActivation headAct) {
	if (hidden.empty()) throw std::invalid_argument("need at least one LSTM layer");
	if (inputSize < 1)  throw std::invalid_argument("LSTM input must be at least 1");
	if (outputSize < 1) throw std::invalid_argument("LSTM output must be at least 1");
	for (size_t H : hidden)
		if (H < 1) throw std::invalid_argument("LSTM hidden size must be at least 1");

	net.inputSize = inputSize;
	net.outputSize = outputSize;
	net.headAct = headAct;
	net.meta.assign(hidden.size(), LayerMeta{});

	size_t cursor = 0, in = inputSize;
	for (size_t l = 0; l < hidden.size(); l++) {
		const size_t H = hidden[l];
		LayerMeta& M = net.meta[l];
		M.inputSize = in;
		M.hiddenSize = H;
		M.biasOff = cursor;
		M.wxOff = M.biasOff + 4 * H;
		M.whOff = M.wxOff + 4 * H * in;
		cursor = M.whOff + 4 * H * H;
		in = H;
	}
	net.head.inputSize = in;
	net.head.outputSize = outputSize;
	net.head.biasOff = cursor;
	net.head.weightOff = cursor + outputSize;
	cursor = net.head.weightOff + outputSize * in;

	net.params.assign(cursor, 0.0);
}

void LSTM_OP::Init(LSTM& net, size_t inputSize, size_t outputSize, size_t layers, size_t hidden,
                   HeadActivation headAct) {
	if (layers < 1) throw std::invalid_argument("LSTM needs at least 1 layer");
	if (hidden < 1) throw std::invalid_argument("LSTM hidden size must be at least 1");

	Layout(net, inputSize, std::vector<size_t>(layers, hidden), outputSize, headAct);

	for (size_t l = 0; l < net.depth(); l++) {
		const LayerMeta& M = net.meta[l];
		const size_t H = M.hiddenSize;
		const double scale = 1.0 / std::sqrt((double)H);

		double* b = net.bias(l);
		for (size_t j = 0; j < H; j++) b[H + j] = 1.0;           // forget gate bias = 1
		double* Wx = net.wx(l);
		for (size_t i = 0; i < 4 * H * M.inputSize; i++) Wx[i] = random_double() * scale;
		double* Wh = net.wh(l);
		for (size_t i = 0; i < 4 * H * H; i++) Wh[i] = random_double() * scale;
	}
	const double hs = 1.0 / std::sqrt((double)net.head.inputSize);
	double* Wo = net.headWeights();
	for (size_t i = 0; i < net.head.outputSize * net.head.inputSize; i++) Wo[i] = random_double() * hs;
}

void LSTM_OP::InitTrace(const LSTM& net, Trace& t, size_t maxSteps) {
	if (maxSteps < 1) throw std::invalid_argument("maxSteps must be at least 1");
	const size_t T = maxSteps, O = net.outputSize;

	t.maxSteps = T;
	t.steps = 0;
	t.layer.assign(net.depth(), TraceLayer{});

	size_t a = 0, d = 0;
	for (size_t l = 0; l < net.depth(); l++) {
		const size_t H = net.meta[l].hiddenSize;
		TraceLayer& L = t.layer[l];
		L.gateOff  = a; a += T * 4 * H;
		L.tanhcOff = a; a += T * H;
		L.cellOff  = a; a += (T + 1) * H;
		L.hidOff   = a; a += (T + 1) * H;

		L.dgateOff = d; d += T * 4 * H;
		L.dhOff    = d; d += T * H;
		L.runOff   = d; d += 2 * H;
	}
	t.outOff = a;  a += T * O;
	t.doutOff = d; d += T * O;

	t.act.assign(a, 0.0);
	t.delta.assign(d, 0.0);
}

// =============================================================== phase 1: forward

void LSTM_OP::Forward(const LSTM& net, const double* x, size_t steps, Trace& t) {
	assert(steps >= 1 && steps <= t.maxSteps);
	t.steps = steps;
	double* act = t.act.data();

	for (size_t l = 0; l < net.depth(); l++) {
		const LayerMeta& M = net.meta[l];
		const TraceLayer& L = t.layer[l];
		const size_t H = M.hiddenSize, I = M.inputSize;
		const double* b = net.bias(l);
		const double* Wx = net.wx(l);
		const double* Wh = net.wh(l);

		// initial state of every sequence is zero
		std::fill(act + L.cellOff, act + L.cellOff + H, 0.0);
		std::fill(act + L.hidOff,  act + L.hidOff + H, 0.0);

		// input of this layer: the caller's x for layer 0, else the layer below's h
		for (size_t s = 0; s < steps; s++) {
			const double* in = (l == 0)
				? x + s * I
				: act + t.layer[l - 1].hidOff + (s + 1) * I;          // I == H of layer below
			const double* hPrev = act + L.hidOff + s * H;
			const double* cPrev = act + L.cellOff + s * H;
			double* gate = act + L.gateOff + s * 4 * H;

			computeGates(M, b, Wx, Wh, in, hPrev, gate);
			cellUpdate(H, gate, cPrev, act + L.cellOff + (s + 1) * H,
			           act + L.tanhcOff + s * H, act + L.hidOff + (s + 1) * H);
		}
	}

	const HeadMeta& HM = net.head;
	const TraceLayer& top = t.layer.back();
	for (size_t s = 0; s < steps; s++)
		headForward(HM, net.headAct, net.headBias(), net.headWeights(),
		            act + top.hidOff + (s + 1) * HM.inputSize, act + t.outOff + s * HM.outputSize);
}

const double* LSTM_OP::Output(const LSTM& net, const Trace& t, size_t step) {
	return t.act.data() + t.outOff + step * net.outputSize;
}

double LSTM_OP::Loss(const LSTM& net, const Trace& t, const double* target, LossMode mode) {
	const size_t O = net.outputSize;
	const size_t first = (mode == LossMode::LastStep) ? t.steps - 1 : 0;
	const size_t count = t.steps - first;

	double sum = 0.0;
	for (size_t s = first; s < t.steps; s++) {
		const double* y = Output(net, t, s);
		const double* tg = (mode == LossMode::LastStep) ? target : target + s * O;
		for (size_t k = 0; k < O; k++) {
			const double d = y[k] - tg[k];
			sum += d * d;
		}
	}
	return sum / (double)(count * O);
}

// =============================================================== phase 2: BPTT

void LSTM_OP::Backward(const LSTM& net, const double* target, Trace& t, LossMode mode) {
	const size_t T = t.steps, O = net.outputSize;
	const double* act = t.act.data();
	double* delta = t.delta.data();

	// ---- head: dLoss/d(head pre-activation), then gradient into the top layer's h_t
	const size_t first = (mode == LossMode::LastStep) ? T - 1 : 0;
	const double norm = 2.0 / (double)((T - first) * O);

	double* dout = delta + t.doutOff;
	std::fill(dout, dout + T * O, 0.0);
	for (size_t s = first; s < T; s++) {
		const double* y = act + t.outOff + s * O;
		const double* tg = (mode == LossMode::LastStep) ? target : target + s * O;
		for (size_t k = 0; k < O; k++) {
			const double dy = norm * (y[k] - tg[k]);
			dout[s * O + k] = (net.headAct == HeadActivation::Sigmoid) ? dy * y[k] * (1.0 - y[k]) : dy;
		}
	}

	const size_t Ht = net.head.inputSize;
	const double* Wo = net.headWeights();
	double* dhTop = delta + t.layer.back().dhOff;
	std::fill(dhTop, dhTop + T * Ht, 0.0);
	for (size_t s = first; s < T; s++)
		for (size_t k = 0; k < O; k++) {
			const double d = dout[s * O + k];
			const double* w = Wo + k * Ht;
			for (size_t j = 0; j < Ht; j++) dhTop[s * Ht + j] += w[j] * d;
		}

	// ---- LSTM layers, top to bottom; inside a layer, time runs backwards
	for (size_t l = net.depth(); l-- > 0;) {
		const LayerMeta& M = net.meta[l];
		const TraceLayer& L = t.layer[l];
		const size_t H = M.hiddenSize, I = M.inputSize;
		const double* Wx = net.wx(l);
		const double* Wh = net.wh(l);

		double* dhNext = delta + L.runOff;
		double* dcNext = dhNext + H;
		std::fill(dhNext, dhNext + 2 * H, 0.0);

		for (size_t s = T; s-- > 0;) {
			const double* gate = act + L.gateOff + s * 4 * H;
			const double* tanhc = act + L.tanhcOff + s * H;
			const double* cPrev = act + L.cellOff + s * H;
			const double* dhExt = delta + L.dhOff + s * H;
			double* dgate = delta + L.dgateOff + s * 4 * H;

			for (size_t j = 0; j < H; j++) {
				const double i = gate[j], f = gate[H + j], g = gate[2 * H + j], o = gate[3 * H + j];
				const double dh = dhExt[j] + dhNext[j];
				const double dc = dh * o * (1.0 - tanhc[j] * tanhc[j]) + dcNext[j];

				dgate[j]         = dc * g * i * (1.0 - i);            // input gate
				dgate[H + j]     = dc * cPrev[j] * f * (1.0 - f);     // forget gate
				dgate[2 * H + j] = dc * i * (1.0 - g * g);            // candidate
				dgate[3 * H + j] = dh * tanhc[j] * o * (1.0 - o);     // output gate
				dcNext[j] = dc * f;
			}

			// dh_(t-1) = Wh^T * dgate
			std::fill(dhNext, dhNext + H, 0.0);
			for (size_t r = 0; r < 4 * H; r++) {
				const double d = dgate[r];
				const double* wh = Wh + r * H;
				for (size_t k = 0; k < H; k++) dhNext[k] += wh[k] * d;
			}

			// gradient for the layer below: dx_t = Wx^T * dgate
			if (l > 0) {
				double* dLow = delta + t.layer[l - 1].dhOff + s * I;
				std::fill(dLow, dLow + I, 0.0);
				for (size_t r = 0; r < 4 * H; r++) {
					const double d = dgate[r];
					const double* wx = Wx + r * I;
					for (size_t m = 0; m < I; m++) dLow[m] += wx[m] * d;
				}
			}
		}
	}
}

// =============================================================== phase 3: update

void LSTM_OP::Update(LSTM& net, const double* x, const Trace& t, double lr) {
	const size_t T = t.steps, O = net.outputSize;
	const double* act = t.act.data();
	const double* delta = t.delta.data();

	for (size_t l = 0; l < net.depth(); l++) {
		const LayerMeta& M = net.meta[l];
		const TraceLayer& L = t.layer[l];
		const size_t H = M.hiddenSize, I = M.inputSize;
		double* b = net.bias(l);
		double* Wx = net.wx(l);
		double* Wh = net.wh(l);

		for (size_t s = 0; s < T; s++) {
			const double* in = (l == 0) ? x + s * I : act + t.layer[l - 1].hidOff + (s + 1) * I;
			const double* hPrev = act + L.hidOff + s * H;
			const double* dg = delta + L.dgateOff + s * 4 * H;

			for (size_t r = 0; r < 4 * H; r++) {
				const double d = lr * dg[r];
				b[r] -= d;
				double* wx = Wx + r * I;
				for (size_t m = 0; m < I; m++) wx[m] -= d * in[m];
				double* wh = Wh + r * H;
				for (size_t k = 0; k < H; k++) wh[k] -= d * hPrev[k];
			}
		}
	}

	const size_t Ht = net.head.inputSize;
	const double* hTop = act + t.layer.back().hidOff;
	double* bo = net.headBias();
	double* Wo = net.headWeights();
	for (size_t s = 0; s < T; s++)
		for (size_t k = 0; k < O; k++) {
			const double d = lr * delta[t.doutOff + s * O + k];
			bo[k] -= d;
			double* w = Wo + k * Ht;
			const double* h = hTop + (s + 1) * Ht;
			for (size_t j = 0; j < Ht; j++) w[j] -= d * h[j];
		}
}

double LSTM_OP::Train(LSTM& net, Trace& t, const double* x, size_t steps,
                      const double* target, LossMode mode, double learningRate) {
	Forward(net, x, steps, t);
	const double loss = Loss(net, t, target, mode);
	Backward(net, target, t, mode);
	Update(net, x, t, learningRate);
	return loss;
}

// =============================================================== inference

void LSTM_OP::InitState(const LSTM& net, State& s) {
	s.off.assign(net.depth(), 0);
	size_t total = 0, widest = 0;
	for (size_t l = 0; l < net.depth(); l++) {
		s.off[l] = total;
		total += net.meta[l].hiddenSize;
		widest = std::max(widest, net.meta[l].hiddenSize);
	}
	s.h.assign(total, 0.0);
	s.c.assign(total, 0.0);
	s.gates.assign(4 * widest, 0.0);
	s.out.assign(net.outputSize, 0.0);
}

void LSTM_OP::Reset(State& s) {
	std::fill(s.h.begin(), s.h.end(), 0.0);
	std::fill(s.c.begin(), s.c.end(), 0.0);
}

const double* LSTM_OP::Step(const LSTM& net, const double* x, State& s) {
	const double* in = x;
	for (size_t l = 0; l < net.depth(); l++) {
		const LayerMeta& M = net.meta[l];
		double* h = s.h.data() + s.off[l];
		double* c = s.c.data() + s.off[l];

		computeGates(M, net.bias(l), net.wx(l), net.wh(l), in, h, s.gates.data());   // reads old h
		cellUpdate(M.hiddenSize, s.gates.data(), c, c, nullptr, h);                  // then overwrites c, h
		in = h;
	}
	headForward(net.head, net.headAct, net.headBias(), net.headWeights(), in, s.out.data());
	return s.out.data();
}

const double* LSTM_OP::Predict(const LSTM& net, const double* x, size_t steps, State& s) {
	assert(steps >= 1);
	Reset(s);
	const double* y = s.out.data();
	for (size_t i = 0; i < steps; i++)
		y = Step(net, x + i * net.inputSize, s);
	return y;
}

// =============================================================== file I/O
namespace io {

namespace {
constexpr char          MAGIC[4] = { 'D', 'L', 'S', 'M' };
constexpr std::uint32_t VERSION  = 1;
}

void Save(const LSTM& net, const std::string& path) {
	using namespace dod_detail;
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) throw std::runtime_error("cannot open for writing: " + path);

	f.write(MAGIC, 4);
	put(f, VERSION);
	put(f, (std::uint64_t)net.depth());
	put(f, (std::uint64_t)net.inputSize);
	put(f, (std::uint64_t)net.outputSize);
	put(f, (std::uint64_t)net.headAct);
	for (const LayerMeta& M : net.meta)
		put(f, (std::uint64_t)M.hiddenSize);

	f.write(reinterpret_cast<const char*>(net.params.data()), net.params.size() * sizeof(double));
	if (!f) throw std::runtime_error("write failed: " + path);
}

void Load(LSTM& net, const std::string& path) {
	using namespace dod_detail;
	using std::uint64_t;

	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) throw std::runtime_error("cannot open for reading: " + path);
	const uint64_t fileSize = (uint64_t)f.tellg();
	f.seekg(0);

	char magic[4];
	std::uint32_t version = 0;
	uint64_t layerCount = 0, inputSize = 0, outputSize = 0, headAct = 0;

	f.read(magic, 4);
	get(f, version);
	get(f, layerCount);
	get(f, inputSize);
	get(f, outputSize);
	get(f, headAct);
	if (!f || magic[0] != 'D' || magic[1] != 'L' || magic[2] != 'S' || magic[3] != 'M')
		throw std::runtime_error("not a DLSM file: " + path);
	if (version != VERSION)
		throw std::runtime_error("unsupported DLSM version");
	if (layerCount == 0 || layerCount > fileSize / 8 || headAct > 1)
		throw std::runtime_error("corrupt header");

	const uint64_t headerBytes = 4 + 4 + 8 * 4 + 8 * layerCount;
	if (headerBytes > fileSize)
		throw std::runtime_error("corrupt header");

	std::vector<uint64_t> hidden(layerCount);
	for (uint64_t l = 0; l < layerCount; l++) get(f, hidden[l]);
	if (!f) throw std::runtime_error("corrupt header");

	// validate sizes against the file BEFORE allocating anything
	if ((fileSize - headerBytes) % 8 != 0)
		throw std::runtime_error("file size is not a whole number of doubles");
	const uint64_t maxDoubles = (fileSize - headerBytes) / 8;
	uint64_t params = 0, in = inputSize;
	for (uint64_t l = 0; l < layerCount; l++) {
		const uint64_t H = hidden[l];
		uint64_t layerParams = 0;
		if (in == 0 || H == 0 || in > maxDoubles || H > maxDoubles / 4 ||
		    !mulWithin(4 * H, 1 + in + H, maxDoubles, layerParams))
			throw std::runtime_error("corrupt layer sizes");
		params += layerParams;
		if (params > maxDoubles) throw std::runtime_error("file too small for declared sizes");
		in = H;
	}
	uint64_t headParams = 0;
	if (outputSize == 0 || outputSize > maxDoubles || !mulWithin(outputSize, 1 + in, maxDoubles, headParams))
		throw std::runtime_error("corrupt output size");
	params += headParams;
	if (params != maxDoubles)
		throw std::runtime_error("file size does not match declared sizes");

	LSTM tmp;                                  // only commit to `net` if everything succeeds
	LSTM_OP::Layout(tmp, (size_t)inputSize, std::vector<size_t>(hidden.begin(), hidden.end()),
	                (size_t)outputSize, static_cast<HeadActivation>(headAct));
	f.read(reinterpret_cast<char*>(tmp.params.data()), tmp.params.size() * sizeof(double));
	if (!f) throw std::runtime_error("read failed: " + path);
	net = std::move(tmp);
}

} // namespace io
} // namespace DOD_LSTM
