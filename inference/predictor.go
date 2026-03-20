package inference

import (
	"fmt"

	"llama.go/tensor"
)

// Predictor is a small low-rank MLP that predicts which FFN neurons will be
// activated (output > 0 after ReLU/GELU) for a given layer, based on the
// attention output of that same layer.
//
// Architecture (following the paper):
//   input  = attention output of current layer  [hidden_size]
//   hidden = W_down_pred @ relu(W_up_pred @ input)  [rank]
//   logits = W_out_pred @ hidden                    [ffn_hidden_size]
//   predict positive if logits[i] > threshold
//
// The predictor weights (W_up_pred, W_down_pred, W_out_pred) are small and
// always resident in DRAM. They can be trained offline and loaded from a file.
type Predictor struct {
	// W_up:  [rank, hidden_size]
	// W_out: [ffn_hidden_size, rank]
	WUp   *tensor.Tensor
	WOut  *tensor.Tensor
	Bias  *tensor.Tensor // optional bias [ffn_hidden_size]

	// Threshold: logit > Threshold → neuron predicted active.
	// Paper uses 0.0 (sign of pre-activation).
	Threshold float32

	// FalsePositiveRate is a soft upper bound on FP rate (for monitoring).
	FalsePositiveRate float32
}

// NewRandomPredictor creates a dummy predictor with random weights for testing.
// In production, weights would be loaded from a checkpoint trained on C4.
func NewRandomPredictor(hiddenSize, ffnHiddenSize, rank int) *Predictor {
	// Initialize weights to small values (not trained — for structural testing only).
	wUp := tensor.New(rank, hiddenSize)
	wOut := tensor.New(ffnHiddenSize, rank)
	return &Predictor{
		WUp:       wUp,
		WOut:      wOut,
		Threshold: 0.0,
	}
}

// Predict returns the set of neuron indices predicted to be active.
// attnOut is the attention output for the current token, shape [hidden_size].
func (p *Predictor) Predict(attnOut *tensor.Tensor) ([]int, error) {
	if p.WUp == nil || p.WOut == nil {
		return nil, fmt.Errorf("predictor: weights not loaded")
	}

	// hidden = relu(W_up @ attnOut)
	hidden, err := tensor.MatMulVec(p.WUp, attnOut)
	if err != nil {
		return nil, fmt.Errorf("predictor W_up: %w", err)
	}
	tensor.ReLU(hidden)

	// logits = W_out @ hidden
	logits, err := tensor.MatMulVec(p.WOut, hidden)
	if err != nil {
		return nil, fmt.Errorf("predictor W_out: %w", err)
	}

	if p.Bias != nil {
		if err := tensor.Add(logits, p.Bias); err != nil {
			return nil, err
		}
	}

	// Collect predicted-active neuron indices.
	active := make([]int, 0, len(logits.Data)/10) // pre-allocate ~10% active
	for i, v := range logits.Data {
		if v > p.Threshold {
			active = append(active, i)
		}
	}
	return active, nil
}

// PredictorSet holds one predictor per transformer layer.
type PredictorSet struct {
	Predictors []*Predictor // one per layer; nil means "load all" (dense)
}

// NewDensePredictorSet creates a set that always predicts all neurons active
// (equivalent to no prediction = load all FFN weights). Used as a baseline.
func NewDensePredictorSet(nLayers int) *PredictorSet {
	ps := &PredictorSet{Predictors: make([]*Predictor, nLayers)}
	// Leave all as nil → dense (load everything).
	return ps
}

// NewRandomPredictorSet creates a set of random (untrained) predictors.
// Use for structural testing; replace with loaded weights for real use.
func NewRandomPredictorSet(nLayers, hiddenSize, ffnHiddenSize, rank int) *PredictorSet {
	preds := make([]*Predictor, nLayers)
	for i := range preds {
		preds[i] = NewRandomPredictor(hiddenSize, ffnHiddenSize, rank)
	}
	return &PredictorSet{Predictors: preds}
}

// Get returns the predictor for layer i (nil = dense/load-all).
func (ps *PredictorSet) Get(layerIdx int) *Predictor {
	if ps == nil || layerIdx >= len(ps.Predictors) {
		return nil
	}
	return ps.Predictors[layerIdx]
}
