package bugminrepro

import (
	"math"
	"testing"
)

func TestBugMinrepro(t *testing.T) {
	evaluator, params, ecd, enc, dec := Bug_minrepro__configure()

	const W = 5
	arg0 := make([]float32, W*W)
	for i := range arg0 {
		arg0[i] = float32(i*i + 1)
	}

	// Output: sum of each 3x3 patch (filter is all ones). 3x3 output spatial.
	expected := make([]float32, 9)
	for oh := 0; oh < 3; oh++ {
		for ow := 0; ow < 3; ow++ {
			var sum float32
			for kh := 0; kh < 3; kh++ {
				for kw := 0; kw < 3; kw++ {
					sum += arg0[(oh+kh)*W+(ow+kw)]
				}
			}
			expected[oh*3+ow] = sum
		}
	}

	ct0 := Bug_minrepro__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Bug_minrepro(evaluator, params, ecd, ct0)
	result := Bug_minrepro__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.5)
	for i := 0; i < 9; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f (diff %.4f)",
				i, result[i], expected[i], result[i]-expected[i])
		}
	}
}
