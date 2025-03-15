package dotproduct8fdebug

import (
	"math"
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := dot_product__configure()

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	arg1 := []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	expected := float32(2.50)

	ct0 := dot_product__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := dot_product__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	resultCt := dot_product(evaluator, params, ecd, dec, ct0, ct1)

	result := dot_product__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.0001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.2f != %.2f", result, expected)
	}
}
