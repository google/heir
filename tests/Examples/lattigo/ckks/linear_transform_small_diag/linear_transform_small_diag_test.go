package lineartransformsmalldiag

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestLinearTransformSmallDiag(t *testing.T) {
	evaluator, params, encoder, encryptor, decryptor := Test_small_diag__configure()
	numSlots := 1 << 12 // 4096
	diagWidth := 4

	// Input vector x of length W = 4, cyclically tiled across all slots.
	// Use non-trivial, non-symmetric values.
	x := []float64{2.0, 3.0, 5.0, 7.0}
	inputClear := make([]float64, numSlots)
	for i := range inputClear {
		inputClear[i] = x[i%diagWidth]
	}

	// 4 diagonals of width 4 with indices [-1, 0, 1, 2]:
	// diagNeg1 (offset -1): [0.75, -1.25, 2.5, -0.5]
	// diag0    (offset  0): [1.5, -2.5, 3.5, -4.5]
	// diag1    (offset  1): [0.5, 1.25, -2.25, 4.0]
	// diag2    (offset  2): [-1.0, 2.0, -0.75, 1.5]
	diagNeg1 := []float64{0.75, -1.25, 2.5, -0.5}
	diag0 := []float64{1.5, -2.5, 3.5, -4.5}
	diag1 := []float64{0.5, 1.25, -2.25, 4.0}
	diag2 := []float64{-1.0, 2.0, -0.75, 1.5}
	matrix := make([]float64, 4*diagWidth)
	copy(matrix[0:4], diagNeg1)
	copy(matrix[4:8], diag0)
	copy(matrix[8:12], diag1)
	copy(matrix[12:16], diag2)

	posMod := func(val, m int) int {
		return (val%m + m) % m
	}

	// Independently computed plaintext reference:
	// y[j] = sum_k diag_k[j mod W] * x[(j + idx_k) mod W]
	// with proper positive modulo for all slots j in [0, numSlots).
	expectedClear := make([]float64, numSlots)
	for j := range expectedClear {
		expectedClear[j] = diagNeg1[j%diagWidth]*x[posMod(j-1, diagWidth)] +
			diag0[j%diagWidth]*x[posMod(j+0, diagWidth)] +
			diag1[j%diagWidth]*x[posMod(j+1, diagWidth)] +
			diag2[j%diagWidth]*x[posMod(j+2, diagWidth)]
	}

	pt := ckks.NewPlaintext(params, 3)
	pt.Scale = params.DefaultScale()
	if err := encoder.Encode(inputClear, pt); err != nil {
		t.Fatal(err)
	}
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}

	epsilon := 0.05

	// Test 1: Combined test_small_diag (prepare + apply)
	resultCt := Test_small_diag(evaluator, params, encoder, ct, matrix)
	resultPt := decryptor.DecryptNew(resultCt)
	resultFloat64 := make([]float64, numSlots)
	if err := encoder.Decode(resultPt, resultFloat64); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < numSlots; i++ {
		if diff := math.Abs(resultFloat64[i] - expectedClear[i]); diff > epsilon {
			t.Fatalf("test_small_diag slot %d: got %f, expected %f (diff %e)", i, resultFloat64[i], expectedClear[i], diff)
		}
	}

	// Test 2: Separate test_prepare and test_apply
	preparedLT := Test_prepare(params, encoder, matrix)
	resultCt2 := Test_apply(evaluator, ct, preparedLT)
	resultPt2 := decryptor.DecryptNew(resultCt2)
	resultFloat64_2 := make([]float64, numSlots)
	if err := encoder.Decode(resultPt2, resultFloat64_2); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < numSlots; i++ {
		if diff := math.Abs(resultFloat64_2[i] - expectedClear[i]); diff > epsilon {
			t.Fatalf("test_prepare/apply slot %d: got %f, expected %f (diff %e)", i, resultFloat64_2[i], expectedClear[i], diff)
		}
	}

	// Test 3: Full-width test_full_diag (W == slots, zero-copy sub-slice path)
	fullMatrix := make([]float64, 2*numSlots)
	for i := 0; i < numSlots; i++ {
		fullMatrix[i] = 1.0          // diag 0
		fullMatrix[numSlots+i] = 2.0 // diag 1
	}
	expectedFull := make([]float64, numSlots)
	for i := 0; i < numSlots; i++ {
		// diag0[i]*x[i] + diag1[i]*x[(i+1)%4096]
		expectedFull[i] = 1.0*inputClear[i] + 2.0*inputClear[(i+1)%numSlots]
	}
	preparedFull := Test_full_diag(params, encoder, fullMatrix)
	resultCtFull := Test_apply(evaluator, ct, preparedFull)
	resultPtFull := decryptor.DecryptNew(resultCtFull)
	resultFloat64Full := make([]float64, numSlots)
	if err := encoder.Decode(resultPtFull, resultFloat64Full); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < numSlots; i++ {
		if diff := math.Abs(resultFloat64Full[i] - expectedFull[i]); diff > epsilon {
			t.Fatalf("test_full_diag slot %d: got %f, expected %f (diff %e)", i, resultFloat64Full[i], expectedFull[i], diff)
		}
	}
}
