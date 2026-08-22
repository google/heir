package hoistrotations

import (
	"math"
	"testing"
)

func TestHoistRotations(t *testing.T) {
	evaluator, params, encoder, encryptor, decryptor := Hoist_rotations__configure()

	// Vector of plaintext values
	arg0 := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

	// Expected: rotate(arg0, 1) + rotate(arg0, 3)
	// rotate(arg0, 1) = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1]
	// rotate(arg0, 3) = [0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3]
	// sum = [0.6, 0.8, 1.0, 1.2, 1.4, 0.8, 1.0, 0.4]
	expected := []float64{0.6, 0.8, 1.0, 1.2, 1.4, 0.8, 1.0, 0.4}

	ct0 := Hoist_rotations__encrypt__arg0(evaluator, params, encoder, encryptor, arg0)

	resultCt := Hoist_rotations(evaluator, params, encoder, ct0)

	result := Hoist_rotations__decrypt__result0(evaluator, params, encoder, decryptor, resultCt)

	errorThreshold := 0.001
	for i, val := range expected {
		if math.Abs(result[i]-val) > errorThreshold {
			t.Errorf("At index %d: expected %f, got %f", i, val, result[i])
		}
	}
}
