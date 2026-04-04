package matvec512x784

import (
	"math"
	"testing"
)

func TestMatvec(t *testing.T) {
	bt_eval, evaluator, params, ecd, enc, dec := matvec__configure()

	cols := 784
	rows := 512
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = 0.1
	}

	expected := float32(78.4)
	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	zero := matvec__encrypt__zero__tensor_1x1024xf32__23mgmt2Emgmt_level_3D_02C_scale_3D_45_(evaluator, params, ecd, enc)
	resultCt := matvec(bt_eval, evaluator, params, ecd, ct0, zero)
	result := matvec__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(2.0)
	for i := 0; i < rows; i++ {
		if math.Abs(float64(result[i]-expected)) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected)
		}
	}
}
