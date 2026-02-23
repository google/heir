package main

import (
	"fmt"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// MakeFlattenedOnes creates a slice of float64 filled with 1.0s.
// The size of the slice is determined by the product of the input 2D dimensions (rows * cols).
func MakeFlattenedOnes(rows, cols int) []float64 {
	size := rows * cols
	tensor := make([]float64, size)
	for i := range tensor {
		tensor[i] = 1.0
	}
	return tensor
}

func makeRange(n int) []int {
	a := make([]int, n)
	for i := range a {
		a[i] = i
	}
	return a
}

func generateGalEls(param ckks.Parameters, indices []int) []uint64 {
	var galEls []uint64
	for _, index := range indices {
		galEls = append(galEls, param.GaloisElement(index))
	}
	return galEls
}

func TestMLP(t *testing.T) {
	logN := 14
	numSlots := 1 << (logN - 1)

	// Input is arbitrary, doesn't matter since we're just testing
	// performance
	inputClear := make([]float64, numSlots)
	for i := range inputClear {
		inputClear[i] = 1.0
	}

	// Function args:
	//
	// %ct: encrypted input,

	// These parameters should match the mlir file, though due to the weird
	// nature of this test, this is the source of truth for what is used,
	// not the mlir file.
	logQ := make([]int, 7)
	for i := range logQ {
		logQ[i] = 60
	}
	param, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{60},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	encoder := ckks.NewEncoder(param)
	kgen := rlwe.NewKeyGenerator(param)
	sk, pk := kgen.GenKeyPairNew()
	encryptor := rlwe.NewEncryptor(param, pk)
	rk := kgen.GenRelinearizationKeyNew(sk)

	// We have to do this once for each distinct linear_transform op to
	// ensure we generate all the galois keys needed by lattigo
	var galEls []uint64
	// Manually add Galois key for extra rotation indices used in the
	// mlir file, outside of linear_transform
	//
	// For some reason I need to manually add rotation keys used in
	// linear_transform! That should have been handled by the above code...
	rotIndices := makeRange(10)
	galEls = append(galEls, generateGalEls(param, rotIndices)...)

	fmt.Printf("Final galEls: %v\n", galEls)

	evk := rlwe.NewMemEvaluationKeySet(rk, kgen.GenGaloisKeysNew(galEls, sk)...)
	evaluator := ckks.NewEvaluator(param, evk)

	pt := ckks.NewPlaintext(param, 2)
	encoder.Encode(inputClear, pt)
	ctInput, err25 := encryptor.EncryptNew(pt)
	if err25 != nil {
		panic(err25)
	}

	fmt.Printf("Starting call")
	startTime := time.Now()
	in_place(evaluator, param, encoder, ctInput)
	duration := time.Since(startTime)
	fmt.Printf("MLP call took: %v\n", duration)
}
