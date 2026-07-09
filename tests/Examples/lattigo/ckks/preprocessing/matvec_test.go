package matvec

import (
	"fmt"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/preprocessing/matvec_utils"
)

func TestMatvec(t *testing.T) {
	evaluator, params, ecd, enc, dec := Matvec__configure()

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

	ct0 := Matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startTime := time.Now()
	resultCt := Matvec(evaluator, params, ecd, ct0, arg0)
	duration := time.Since(startTime)
	fmt.Printf("matvec call took: %v\n", duration)

	result := Matvec__decrypt__result0(nil, params, ecd, dec, resultCt)
	fmt.Printf("Result: %v\n", result)
}

func TestMatvecSplit(t *testing.T) {
	evaluator, params, ecd, enc, dec := Matvec__configure()

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

	ct0 := Matvec__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	// Call preprocessing separately
	storage := matvec_utils.Matvec__preprocessing(params, ecd, arg0)

	startTime := time.Now()
	// Call preprocessed function separately
	resultCt := Matvec__preprocessed(evaluator, params, ecd, ct0, arg0, storage)
	duration := time.Since(startTime)
	fmt.Printf("Matvec__preprocessed call took: %v\n", duration)

	result := Matvec__decrypt__result0(nil, params, ecd, dec, resultCt)
	fmt.Printf("Split Result: %v\n", result)
}
