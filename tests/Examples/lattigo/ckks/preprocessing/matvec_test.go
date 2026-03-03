package matvec

import (
	"fmt"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/preprocessing/matvec_utils"
)

func TestMatvec(t *testing.T) {
	evaluator, params, ecd, enc, dec := matvec__configure()

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startTime := time.Now()
	resultCt := matvec(evaluator, params, ecd, ct0, arg0)
	duration := time.Since(startTime)
	fmt.Printf("matvec call took: %v\n", duration)

	result := matvec__decrypt__result0(nil, params, ecd, dec, resultCt)
	fmt.Printf("Result: %v\n", result)
}

func TestMatvecSplit(t *testing.T) {
	evaluator, params, ecd, enc, dec := matvec__configure()

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

	ct0 := matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	// Call preprocessing separately
	v2, v3 := matvec_utils.Matvec__preprocessing(params, ecd, arg0)

	startTime := time.Now()
	// Call preprocessed function separately
	resultCt := matvec__preprocessed(evaluator, params, ecd, ct0, v2, v3)
	duration := time.Since(startTime)
	fmt.Printf("matvec__preprocessed call took: %v\n", duration)

	result := matvec__decrypt__result0(nil, params, ecd, dec, resultCt)
	fmt.Printf("Split Result: %v\n", result)
}
