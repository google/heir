package matvectall

import (
	"math"
	"testing"
)

func TestMatvec(t *testing.T) {
	evaluator, params, ecd, enc, dec := matvec__configure()

	cols := 4
	rows := 8
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = float32(i + 1)
	}

	expected := []float32{20, 60, 100, 140, 180, 220, 260, 300}
	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := matvec(evaluator, params, ecd, ct0)
	result := matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.5)
	for i := 0; i < rows; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected[i])
		}
	}
}
