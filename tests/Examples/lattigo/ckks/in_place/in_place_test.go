package in_place_lib

import (
	"fmt"
	"math"
	"testing"
)

func TestInPlaceRotation(t *testing.T) {
	evaluator, params, encoder, encryptor, decryptor := in_place__configure()

	input := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	// Rotation left by 1: [2, 3, 4, 5, 6, 7, 8, 1]
	// Multiplication by 2: [4, 6, 8, 10, 12, 14, 16, 2]
	expected := []float64{4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 2.0}

	ct := in_place__encrypt__arg0(evaluator, params, encoder, encryptor, input)

	// Call the function
	resultCt := in_place(evaluator, params, encoder, decryptor, ct)

	// Verify in-place behavior using debug hooks
	expectedOps := []string{
		"input",
		"rotate",
		"mul",
	}

	if len(ObservedOps) < len(expectedOps) {
		t.Fatalf("Expected at least %d debug ops, got %d. Ops: %v", len(expectedOps), len(ObservedOps), ObservedOps)
	}

	for i, expectedOp := range expectedOps {
		if ObservedOps[i] != expectedOp {
			t.Errorf("At index %d: expected op %s, got %s", i, expectedOp, ObservedOps[i])
		}
	}

	if len(ObservedPointers) < len(expectedOps) {
		t.Fatalf("Expected at least %d debug pointers, got %d", len(expectedOps), len(ObservedPointers))
	}

	pInput := ObservedPointers[0]
	pRotate := ObservedPointers[1]
	pMul := ObservedPointers[2]
	pResult := fmt.Sprintf("%p", resultCt[0])

	if pInput == pRotate {
		t.Errorf("Expected rotate to be out-of-place (different pointers), but got same pointer: %s", pInput)
	}

	if pRotate != pMul {
		t.Errorf("Expected mul to be in-place (same pointer), but got different pointers: rotate=%s, mul=%s", pRotate, pMul)
	}

	if pMul != pResult {
		t.Errorf("Expected rescale to be in-place (same pointer), but got different pointers: mul=%s, result=%s", pMul, pResult)
	}

	// Verify that the result has the expected length
	if len(resultCt) != len(ct) {
		t.Fatalf("Expected result slice length %d, got %d", len(ct), len(resultCt))
	}

	// Decrypt and verify values
	output := in_place__decrypt__result0(evaluator, params, encoder, decryptor, resultCt)

	errorThreshold := 0.001
	for i, val := range expected {
		if math.Abs(output[i]-val) > errorThreshold {
			t.Errorf("At index %d: expected %f, got %f", i, val, output[i])
		}
	}
}
