package mean

import (
	"math"
	"testing"
)

func TestMean(t *testing.T) {
	evaluator, params, ecd, enc, dec := Mean__configure()

	const N = 64
	arg0 := make([]float32, N)
	sum := float64(0)
	for i := 0; i < N; i++ {
		arg0[i] = float32(math.Sin(float64(i)*0.7) + 0.3)
		sum += float64(arg0[i])
	}
	expected := sum / N

	ct0 := Mean__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Mean(evaluator, params, ecd, ct0)
	result := Mean__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if len(result) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(result))
	}
	errorThreshold := float64(1e-3)
	if math.Abs(float64(result[0])-expected) > errorThreshold {
		t.Errorf("Decryption error: %.6f != %.6f", result[0], expected)
	}
	t.Logf("Result: %.6f, expected: %.6f", result[0], expected)
}
