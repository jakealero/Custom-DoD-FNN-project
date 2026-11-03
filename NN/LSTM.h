#pragma once
#ifdef DOD_NO_LSTM
#error "DOD_LSTM is disabled in this build of dodnn (configure with -DDODNN_ENABLE_LSTM=ON)."
#endif

#include <cstddef>
#include <cstdint>
#include <vector>

// Stacked LSTM layers followed by one dense output head.
//
// Gate order inside every layer is  i (input), f (forget), g (candidate), o (output):
//   rows [0,H) = i,  [H,2H) = f,  [2H,3H) = g,  [3H,4H) = o        (H = hiddenSize)
//   z_r  = b[r] + Wx[r,:] . x_t + Wh[r,:] . h_(t-1)
//   i,f,o = sigmoid(z);  g = tanh(z)
//   c_t  = f * c_(t-1) + i * g
//   h_t  = o * tanh(c_t)
//
// Sequences are time-major and contiguous: x[t * inputSize + m].
namespace DOD_LSTM {

using std::size_t;

enum class HeadActivation : std::uint32_t { Linear = 0, Sigmoid = 1 };

// LastStep : loss only on the final timestep   (many-to-one; target = [outputSize])
// AllSteps : loss on every timestep            (many-to-many; target = [steps][outputSize])
enum class LossMode : std::uint32_t { LastStep = 0, AllSteps = 1 };

// ---------------------------------------------------------------- parameters
// Where one LSTM layer lives inside LSTM::params. Pure metadata.
//   params[biasOff + r]                    bias of gate-row r            (4H values)
//   params[wxOff   + r * inputSize  + m]   input -> gate weights         (4H x I)
//   params[whOff   + r * hiddenSize + k]   recurrent weights             (4H x H)
struct LayerMeta {
	size_t inputSize = 0, hiddenSize = 0;
	size_t biasOff = 0, wxOff = 0, whOff = 0;
};

//   params[biasOff   + k]                  output bias
//   params[weightOff + k * inputSize + j]  output weights                (O x H_last)
struct HeadMeta {
	size_t inputSize = 0, outputSize = 0;
	size_t biasOff = 0, weightOff = 0;
};

// params layout: [ b0|Wx0|Wh0 | b1|Wx1|Wh1 | ... | bOut|WOut ]  -- ONE contiguous block.
struct LSTM {
	size_t inputSize = 0, outputSize = 0;
	HeadActivation headAct = HeadActivation::Linear;
	std::vector<LayerMeta> meta;     // [layerCount]
	HeadMeta head;
	std::vector<double> params;

	size_t depth() const { return meta.size(); }

	double*       bias(size_t l)       { return params.data() + meta[l].biasOff; }
	const double* bias(size_t l) const { return params.data() + meta[l].biasOff; }
	double*       wx  (size_t l)       { return params.data() + meta[l].wxOff;   }
	const double* wx  (size_t l) const { return params.data() + meta[l].wxOff;   }
	double*       wh  (size_t l)       { return params.data() + meta[l].whOff;   }
	const double* wh  (size_t l) const { return params.data() + meta[l].whOff;   }

	double*       headBias()          { return params.data() + head.biasOff;   }
	const double* headBias()    const { return params.data() + head.biasOff;   }
	double*       headWeights()       { return params.data() + head.weightOff; }
	const double* headWeights() const { return params.data() + head.weightOff; }
};

// ------------------------------------------------- training intermediates (BPTT)
// Where one layer's slices start. Everything lives in two flat arrays plus an offset table:
//   act   : everything the forward pass computes
//   delta : everything the backward pass computes
// T = maxSteps, H = hiddenSize of that layer.
struct TraceLayer {
	// into Trace::act
	size_t gateOff = 0;    // [T   ][4H]  post-activation gates  i f g o
	size_t tanhcOff = 0;   // [T   ][H ]  tanh(c_t)
	size_t cellOff = 0;    // [T+1 ][H ]  c ; slot 0 = initial state (zeros), slot t+1 = after step t
	size_t hidOff = 0;     // [T+1 ][H ]  h ; same indexing
	// into Trace::delta
	size_t dgateOff = 0;   // [T   ][4H]  dLoss/d(pre-activation) of every gate row
	size_t dhOff = 0;      // [T   ][H ]  gradient arriving at h_t from the layer above / the head
	size_t runOff = 0;     // [2H]        running dh_next, dc_next while walking back in time
};

struct Trace {
	size_t maxSteps = 0;   // capacity chosen at InitTrace
	size_t steps = 0;      // length of the last Forward
	std::vector<TraceLayer> layer;
	size_t outOff = 0;     // act   : [T][O] network outputs
	size_t doutOff = 0;    // delta : [T][O] dLoss/d(head pre-activation)
	std::vector<double> act, delta;
};

// ------------------------------------------------- inference state (no history)
// Everything a streaming predictor needs: the current h and c of each layer.
struct State {
	std::vector<size_t> off;     // per layer: start inside h and c
	std::vector<double> h, c;    // all layers concatenated
	std::vector<double> gates;   // temp, 4 * widest hidden size
	std::vector<double> out;     // [outputSize]
};

class LSTM_OP {
public:
	// hidden = { H_0, H_1, ... } one entry per LSTM layer.
	static void Layout(LSTM& net, size_t inputSize, const std::vector<size_t>& hidden,
	                   size_t outputSize, HeadActivation headAct);

	// layers stacked LSTM layers, all `hidden` wide. Weights ~ U(-1,1)/sqrt(hidden),
	// forget-gate biases start at 1.0 (standard trick), all other biases at 0.
	static void Init(LSTM& net, size_t inputSize, size_t outputSize, size_t layers, size_t hidden,
	                 HeadActivation headAct = HeadActivation::Linear);

	// ---- training: allocate the Trace ONCE for the longest sequence you will use
	static void InitTrace(const LSTM& net, Trace& t, size_t maxSteps);

	// Phase 1: forward over x[steps][inputSize]; records everything in the Trace.
	static void Forward(const LSTM& net, const double* x, size_t steps, Trace& t);

	// Network output at timestep `step` (outputSize values) after Forward.
	static const double* Output(const LSTM& net, const Trace& t, size_t step);

	// MSE (mean over the counted steps and outputs) of the last Forward.
	// target: LastStep -> [O];  AllSteps -> [steps][O].
	static double Loss(const LSTM& net, const Trace& t, const double* target, LossMode mode);

	// Phase 2: back-propagation through time. Reads weights + act, writes only delta.
	static void Backward(const LSTM& net, const double* target, Trace& t, LossMode mode);

	// Phase 3: plain gradient descent. `x` must be the same sequence given to Forward.
	static void Update(LSTM& net, const double* x, const Trace& t, double learningRate);

	// One SGD step on one sequence; returns the loss.
	static double Train(LSTM& net, Trace& t, const double* x, size_t steps,
	                    const double* target, LossMode mode, double learningRate);

	// ---- inference: parameters + a tiny State only (no Trace, no deltas)
	static void InitState(const LSTM& net, State& s);          // allocate once
	static void Reset(State& s);                                // zero h and c (start of a new sequence)

	// Advance ONE timestep (streaming). No allocation. Returns outputSize values
	// stored in `s`, valid until the next Step/Predict call.
	static const double* Step(const LSTM& net, const double* x, State& s);

	// Reset, run `steps` timesteps, return the LAST output.
	static const double* Predict(const LSTM& net, const double* x, size_t steps, State& s);
};

} // namespace DOD_LSTM
