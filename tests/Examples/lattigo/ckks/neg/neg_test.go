package neg

import (
	"math"
	"testing"
)

func TestNeg(t *testing.T) {
	evaluator, params, ecd, enc, dec := Neg__configure()

	const N = 64
	arg0 := make([]float32, N)
	for i := 0; i < N; i++ {
		arg0[i] = float32(math.Sin(float64(i) * 0.7))
	}

	ct0 := Neg__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Neg(evaluator, params, ecd, ct0)
	result := Neg__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(1e-4)
	for i := range arg0 {
		if math.Abs(float64(result[i]+arg0[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.6f != %.6f", i, result[i], -arg0[i])
		}
	}
}
