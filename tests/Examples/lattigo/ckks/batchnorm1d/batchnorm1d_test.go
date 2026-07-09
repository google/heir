package batchnorm1d

import (
	"testing"
)

func TestBatchNorm1d(t *testing.T) {
	evaluator, params, ecd, enc, dec := Batchnorm1d__configure()

	arg0 := []float32{0.1, 0.2, 0.3} // mean
	arg1 := []float32{0.4, 0.5, 0.6} // var
	arg2 := []int64{0}
	arg3 := make([]float32, 48) // data
	for i := 0; i < 48; i++ {
		arg3[i] = float32(i) * 0.1
	}

	ct3 := Batchnorm1d__encrypt__arg3(evaluator, params, ecd, enc, arg3)

	resultCt := Batchnorm1d(evaluator, params, ecd, arg0, arg1, arg2, ct3)

	result := Batchnorm1d__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	t.Logf("Result size: %d", len(result))
}
