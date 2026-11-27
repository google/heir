package avgpool1d

import (
	"math"
	"testing"
)

func TestAvgPool1d(t *testing.T) {
	evaluator, params, ecd, enc, dec := Avgpool1d__configure()

	// Input: 1x4x16, output: 1x4x8 with kernel_size=2, stride=2.
	const C = 4
	const W = 16
	const OW = 8
	arg0 := make([]float32, C*W)
	for i := range arg0 {
		arg0[i] = float32(i) / 8.0
	}

	expected := make([]float32, C*OW)
	for c := 0; c < C; c++ {
		for i := 0; i < OW; i++ {
			expected[c*OW+i] = (arg0[c*W+2*i] + arg0[c*W+2*i+1]) / 2
		}
	}

	ct0 := Avgpool1d__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Avgpool1d(evaluator, params, ecd, ct0)
	result := Avgpool1d__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
