package conv1dncwpadded

import (
	"math"
	"testing"
)

func TestConv1DPadded(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv1d_ncw_padded__configure()

	arg0 := make([]float32, 16)
	for i := 0; i < 16; i++ {
		arg0[i] = float32(i) * 0.1
	}

	expected := []float32{
		0.0, 0.35, 0.25, 0.15, 0.05, -0.05, -0.15, 1.35,
		-1.4, -0.7, -0.75, -0.8, -0.85, -0.9, -0.95, -0.6,
		0.0, -0.7, -0.7, -0.7, -0.7, -0.7, -0.7, -1.5,
	}

	ct0 := Conv1d_ncw_padded__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv1d_ncw_padded(evaluator, params, ecd, ct0)
	result := Conv1d_ncw_padded__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
