package dwpool

import (
	"math"
	"testing"
)

func TestDepthwiseConvPool(t *testing.T) {
	evaluator, params, ecd, enc, dec := Dwpool__configure()

	const channels, size = 2, 6
	filter := [channels][3][3]float64{
		{{0.1, 0.2, 0.1}, {0.2, 0.4, 0.2}, {0.1, 0.2, 0.1}},
		{{0.0, -0.25, 0.0}, {-0.25, 1.0, -0.25}, {0.0, -0.25, 0.0}},
	}
	weights := [channels][3]float64{
		{0.027777778, 0.013888889, -0.027777778},
		{0.013888889, -0.027777778, 0.027777778},
	}

	arg0 := make([]float32, channels*size*size)
	for i := range arg0 {
		arg0[i] = float32(i%7-3) * 0.1
	}
	at := func(c, h, w int) float64 {
		if h < 0 || h >= size || w < 0 || w >= size {
			return 0 // zero padding
		}
		return float64(arg0[(c*size+h)*size+w])
	}
	// Depthwise convolution, then the sum pool.
	var pooled [channels]float64
	for c := 0; c < channels; c++ {
		for h := 0; h < size; h++ {
			for w := 0; w < size; w++ {
				for kh := 0; kh < 3; kh++ {
					for kw := 0; kw < 3; kw++ {
						pooled[c] += at(c, h+kh-1, w+kw-1) * filter[c][kh][kw]
					}
				}
			}
		}
	}
	expected := make([]float64, 3)
	for j := range expected {
		for c := 0; c < channels; c++ {
			expected[j] += pooled[c] * weights[c][j]
		}
	}

	ct0 := Dwpool__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Dwpool(evaluator, params, ecd, ct0)
	result := Dwpool__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	for i, want := range expected {
		if got := float64(result[i]); math.IsNaN(got) || math.Abs(got-want) > 1e-3 {
			t.Errorf("result[%d] = %.5f, want %.5f", i, result[i], want)
		}
	}
}
