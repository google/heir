package matvec

import (
	"math"
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := matvec__configure()
	arg0 := []float32{1.0, 0, 0, 0, 0, 0, 0, 0,
										  0, 0, 0, 0, 0, 0, 0, 0}
	expected := float32(-0.35219)

	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := matvec(evaluator, params, ecd, ct0)
	result := matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)[0]

	errorThreshold := float64(0.0001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.2f != %.2f", result, expected)
	}
}

