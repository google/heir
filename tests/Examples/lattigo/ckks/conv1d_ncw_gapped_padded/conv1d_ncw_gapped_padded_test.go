package conv1dncwgappedpadded

import (
	"math"
	"testing"
)

// The second conv in this chain takes a data operand that is both gapped (the
// first conv has stride 2) and padded. LayoutPropagation folds that pad into the
// conv's own padding parameter and absorbs the gapped packing into the plaintext
// diagonal filter, so this checks that the absorbed matrix computes the same
// numbers a plain convolution does.
func TestConv1DGappedPadded(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv1d_ncw_gapped_padded__configure()

	arg0 := make([]float32, 16)
	for i := 0; i < 16; i++ {
		arg0[i] = float32(i) * 0.1
	}

	expected := []float32{
		-1.0125, -0.8375, -0.875, 0.0125,
		0.825, 0.1875, 0.5375, 0.6375,
	}

	ct0 := Conv1d_ncw_gapped_padded__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv1d_ncw_gapped_padded(evaluator, params, ecd, ct0)
	result := Conv1d_ncw_gapped_padded__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
