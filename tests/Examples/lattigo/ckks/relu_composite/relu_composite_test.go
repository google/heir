package relucomposite

import (
	"math"
	"testing"
	"time"
)

func TestReluComposite(t *testing.T) {
	evaluator, params, ecd, enc, dec := Relu_composite__configure()

	const n = 16
	arg0 := make([]float32, n)
	expected := make([]float32, n)
	for i := 0; i < n; i++ {
		x := float32(i)*0.25 - 2.0
		arg0[i] = x
		if x > 0 {
			expected[i] = x
		}
	}

	ct0 := Relu_composite__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	start := time.Now()
	resultCt := Relu_composite(evaluator, params, ecd, ct0)
	t.Logf("composite-sign ReLU took %s", time.Since(start))

	result := Relu_composite__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// The composite sign converges to the true sign away from zero but rounds
	// the kink. Because the result is x*step(x/B), the kink error is damped by
	// |x|, so the absolute error stays small even where step itself is poor.
	const errorThreshold = 0.01
	worst := 0.0
	for i := range expected {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > worst {
			worst = diff
		}
		if diff > errorThreshold {
			t.Errorf("index %d (x=%.4f): got %.4f want %.4f (err %.4f)",
				i, arg0[i], result[i], expected[i], diff)
		}
	}
	t.Logf("worst absolute error: %.6f", worst)

	for i := range arg0 {
		if arg0[i] < 0 && result[i] > errorThreshold {
			t.Errorf("index %d (x=%.4f): negative input produced %.4f, want ~0",
				i, arg0[i], result[i])
		}
	}
}
