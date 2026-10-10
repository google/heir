package matvec512x784

import (
	"math"
	"testing"
)

func TestMatvec(t *testing.T) {
	evaluator, params, ecd, enc, dec := Matvec__configure()

	cols := 784
	rows := 512
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = 0.1
	}

	expected := float32(78.4)
	ct0 := Matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Matvec(evaluator, params, ecd, ct0)
	result := Matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	// Error threshold increased to 4.0 due to fallback to Halevi-Shoup kernel
	// which has different noise characteristics.
	errorThreshold := float64(4.0)
	for i := 0; i < rows; i++ {
		if math.Abs(float64(result[i]-expected)) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected)
		}
	}
}
