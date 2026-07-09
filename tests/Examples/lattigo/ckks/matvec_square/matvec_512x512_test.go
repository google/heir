package matvec512x512

import (
	"math"
	"testing"
)

func TestMatvec(t *testing.T) {
	evaluator, params, ecd, enc, dec := Matvec__configure()

	cols := 512
	rows := 512
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = 0.1
	}

	expected := float32(51.2)
	ct0 := Matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	zero := Matvec__encrypt__zero__0(evaluator, params, ecd, enc)
	resultCt := Matvec(evaluator, params, ecd, ct0, zero)
	result := Matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.00001)
	for i := 0; i < rows; i++ {
		if math.Abs(float64(result[i]-expected)) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected)
		}
	}
}
