package paddot

import (
	"math"
	"testing"
)

func TestPadDot(t *testing.T) {
	evaluator, params, ecd, enc, dec := Pad_dot__configure()

	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	arg1 := []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	expected := float32(1.00)

	ct0 := Pad_dot__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := Pad_dot__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := Pad_dot(evaluator, params, ecd, ct0, ct1)

	result := Pad_dot__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.0001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.4f != %.4f", result, expected)
	}
}
