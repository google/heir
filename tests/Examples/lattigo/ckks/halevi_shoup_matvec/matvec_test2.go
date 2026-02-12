package matvec

import (
	"math"
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := matvec__configure()
	arg0 := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
										  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

	firstRow := []float32{0.09922469,  0.18692751,  1.1132321,  -0.6618669,
		1.1475141,  -0.21186261, -1.2645711,   0.02269642, -2.8995476,   0.25620663,
		0.58892345, -0.66741663, 0.6077784,   0.26087236, -0.1372716,  -0.79403836}

	// expected is sum of firstRow + bias (-0.451)
	expected := float32(-0.45141533017158508)
	for i := 0; i < len(firstRow); i++ {
		expected += firstRow[i]
	}

	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := matvec(evaluator, params, ecd, ct0)
	result := matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)[0]

	errorThreshold := float64(0.0001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.8f != %.8f", result, expected)
	}
}

