package conv2dnchwpadded

import (
	"math"
	"testing"
)

func TestConv2DPadded(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_nchw_padded__configure()

	// 1x4x4x4 input, row-major over (c, h, w).
	arg0 := []float32{
		0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
		0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.0, 0.1, 0.2, 0.3, 0.4,
		1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		1.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
	}

	expected := []float32{
		-0.35, -1.85, -0.20, 2.10, -0.20, 4.80, -0.10, -0.80, -1.20, -2.80, -2.90, 0.80, 0.80, 0.90, 2.40, -0.30,
		2.70, -0.40, -2.05, -2.45, -1.75, -3.05, -0.65, 2.05, 1.85, 2.55, -0.65, -3.15, -1.05, -1.35, -1.30, 0.20,
		-1.00, 0.80, 4.10, 1.00, 4.20, 1.10, -1.20, -1.10, -0.60, -0.10, 1.60, 3.90, 1.60, 0.90, -0.75, -1.55,
		-0.45, -1.75, -1.75, 0.95, -2.35, -0.25, 4.50, 0.00, 0.45, -1.25, -2.90, -1.80, -0.50, 0.90, 1.80, 2.70,
	}

	ct0 := Conv2d_nchw_padded__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ctZero := Conv2d_nchw_padded__encrypt__zero__0(evaluator, params, ecd, enc)
	resultCt := Conv2d_nchw_padded(evaluator, params, ecd, ct0, ctZero)
	result := Conv2d_nchw_padded__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
