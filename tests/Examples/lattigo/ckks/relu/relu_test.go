package relu

import (
	"math"
	"testing"
)

func TestRelu(t *testing.T) {
	evaluator, params, ecd, enc, dec := just_relu__configure()
	arg0 := []float32{1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	expected := []float32{1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	ct0 := just_relu__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := just_relu(evaluator, params, ecd, ct0)
	result := just_relu__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		resultVal := result[i];
		expectedVal := expected[i]
		if math.Abs(float64(resultVal-expectedVal)) > errorThreshold {
			t.Errorf("Decryption error at %d: %.6f != %.6f", i, resultVal, expectedVal)
		}
	}
}

