package sigmoid

import (
	"math"
	"testing"
	"time"
)

func Sigmoid32(x float32) float32 {
	// math.Exp requires float64
	exp := math.Exp(-float64(x))
	return float32(1.0 / (1.0 + exp))
}

func TestSigmoid(t *testing.T) {
	evaluator, params, ecd, enc, dec := Sigmoid__configure()
	// Currently a degree chebyshev polynomial should not consume more than depth 4.
	if params.MaxLevelQ() > 4 {
		t.Errorf("Expected polynomial approx level to be <= 4, got: %d", params.MaxLevelQ())
	}

	// Input: 1x1x32x32 = 1024 elements in the -3 to 3 range
	arg0 := make([]float32, 1024)
	for i := range arg0 {
		arg0[i] = float32(-3.0 + 6.0*float32(i)/1024.0)
	}

	// Expected output: 1x1x32x32 = 1024 elements
	expected := make([]float32, 1024)
	for i := range expected {
		expected[i] = Sigmoid32(arg0[i])
	}

	ct0 := Sigmoid__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	// No preprocessing needed
	start := time.Now()
	resultCt := Sigmoid(evaluator, params, ecd, ct0)
	t.Logf("Sigmoid took %s", time.Since(start))

	result := Sigmoid__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
