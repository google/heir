package bicyclicmatmul

import (
	"math"
	"testing"
)

func TestBicyclicMatmulRobust(t *testing.T) {
	evaluator, params, ecd, enc, dec := Bicyclic_matmul__configure()

	q := make([]float32, 16*17)
	for i := 0; i < 16; i++ {
		for j := 0; j < 17; j++ {
			q[i*17+j] = float32(i+j) / 100.0
		}
	}

	k := make([]float32, 17*19)
	for j := 0; j < 17; j++ {
		for l := 0; l < 19; l++ {
			k[j*19+l] = float32(j-l) / 100.0
		}
	}

	// Precompute expected C[i][l] directly from the actual q and k slice elements
	expected := make([]float32, 16*19)
	for i := 0; i < 16; i++ {
		for l := 0; l < 19; l++ {
			for j := 0; j < 17; j++ {
				expected[i*19+l] += q[i*17+j] * k[j*19+l]
			}
		}
	}

	qCt := Bicyclic_matmul__encrypt__arg0(evaluator, params, ecd, enc, q)
	kCt := Bicyclic_matmul__encrypt__arg1(evaluator, params, ecd, enc, k)

	resultCt := Bicyclic_matmul(evaluator, params, ecd, qCt, kCt)
	result := Bicyclic_matmul__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(1e-2)
	for i := 0; i < 16*19; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
