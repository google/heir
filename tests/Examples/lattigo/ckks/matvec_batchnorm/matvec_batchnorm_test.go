package matvecbatchnorm

import (
	"fmt"
	"testing"
)

func TestMatvecBatchNorm(t *testing.T) {
	evaluator, params, ecd, enc, dec := Matvec_batchnorm__configure()
	arg0 := []float32{1, 2, 3, 4}
	arg1 := []float32{2, 2, 2, 2} // scale
	arg2 := []float32{1, 1, 1, 1} // shift

	ct0 := Matvec_batchnorm__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Matvec_batchnorm(evaluator, params, ecd, ct0, arg1, arg2)
	result := Matvec_batchnorm__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	fmt.Println(result)

	expected := []float32{61, 141, 221, 301}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("Expected %v, got %v at index %d", expected[i], v, i)
		}
	}
}
