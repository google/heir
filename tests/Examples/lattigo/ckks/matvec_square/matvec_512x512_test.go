package matvec512x512

import (
	"math"
	"testing"
)

func TestMatvec(t *testing.T) {
	evaluator, params, ecd, enc, dec := matvec__configure()

	cols := 512
	rows := 512
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = 0.1
	}

	expected := float32(51.2)
	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	zero := matvec__encrypt__zero__279e92d4f32b6885(evaluator, params, ecd, enc)
	resultCt := matvec(evaluator, params, ecd, ct0, zero)
	result := matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.00001)
	for i := 0; i < rows; i++ {
		if math.Abs(float64(result[i]-expected)) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected)
		}
	}
}
