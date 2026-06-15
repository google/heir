package main

import (
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
	resultCt := in_place(evaluator, params, encoder, ct)

	// Verify that it was indeed in-place (same pointer)
	if len(resultCt) != len(ct) {
		t.Fatalf("Expected result slice length %d, got %d", len(ct), len(resultCt))
	}
	if len(ct) == 0 {
		t.Fatalf("Expected non-empty ciphertext slice")
	}
	for i := range ct {
		if resultCt[i] != ct[i] {
			t.Errorf("Expected in-place rotation at index %d (same pointer), but got different pointers", i)
		}
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
