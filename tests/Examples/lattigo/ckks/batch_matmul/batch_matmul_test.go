package batchmatmul

import (
	"math"
	"testing"
)

func TestBatchMatmulRobust(t *testing.T) {
	evaluator, params, ecd, enc, dec := Batch_matmul__configure()

	q := make([]float32, 2*17*19)
	for b := 0; b < 2; b++ {
		for i := 0; i < 17; i++ {
			for j := 0; j < 19; j++ {
				q[b*17*19+i*19+j] = float32(b+i+j) / 100.0
			}
		}
	}

	k := make([]float32, 2*19*21)
	for b := 0; b < 2; b++ {
		for j := 0; j < 19; j++ {
			for l := 0; l < 21; l++ {
				k[b*19*21+j*21+l] = float32(b+j-l) / 100.0
			}
		}
	}

	expected := make([]float32, 2*17*21)
	for b := 0; b < 2; b++ {
		for i := 0; i < 17; i++ {
			for l := 0; l < 21; l++ {
				for j := 0; j < 19; j++ {
					expected[b*17*21+i*21+l] += q[b*17*19+i*19+j] * k[b*19*21+j*21+l]
				}
			}
		}
	}

	qCt := Batch_matmul__encrypt__arg0(evaluator, params, ecd, enc, q)
	kCt := Batch_matmul__encrypt__arg1(evaluator, params, ecd, enc, k)

	resultCt := Batch_matmul(evaluator, params, ecd, qCt, kCt)
	result := Batch_matmul__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(1e-2)
	for i := 0; i < 2*17*21; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
