package mlp

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"testing"
	"time"
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

func makeRange(n uint64) []uint64 {
	a := make([]uint64, n)
	for i := range a {
		a[i] = uint64(i)
	}
	return a
}

func generateLintransGaloisKeys(param ckks.Parameters, diagonalIndices []int) []uint64 {
	lintransParams := lintrans.Parameters{
		DiagonalsIndexList:        diagonalIndices,
		LevelQ:                    5,
		LevelP:                    param.MaxLevelP(),
		Scale:                     rlwe.NewScale(param.Q()[5]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: 12}, // 1x4096
		LogBabyStepGiantStepRatio: 2,
	}
	lt := lintrans.NewTransformation(param, lintransParams)
	return lt.GaloisElements(param)
}

func TestMLP(t *testing.T) {
	logN := 17
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
	logQ := make([]int, 29)
	for i := range logQ {
		logQ[i] = 51
	}
	param, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{51},
		LogDefaultScale: 51,
	})
	if err != nil {
		panic(err)
	}

	encoder := ckks.NewEncoder(param)
	kgen := rlwe.NewKeyGenerator(param)
	sk, pk := kgen.GenKeyPairNew()
	encryptor := rlwe.NewEncryptor(param, pk)
	decryptor := rlwe.NewDecryptor(param, sk)
	rk := kgen.GenRelinearizationKeyNew(sk)

	// We have to do this once for each distinct linear_transform op to
	// ensure we generate all the galois keys needed by lattigo
	var galEls []uint64
	// all128Indices := make([]int, 0, 100)
	// fc3Extra := []int{4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095}
	// fc3Indices := append(all128Indices, fc3Extra...)
	// galEls = append(galEls, generateLintransGaloisKeys(param, all128Indices)...)
	// galEls = append(galEls, generateLintransGaloisKeys(param, all128Indices)...)
	// galEls = append(galEls, generateLintransGaloisKeys(param, fc3Indices)...)

	// Manually add Galois key for extra rotation indices used in the
	// mlir file, outside of linear_transform
	//
	// For some reason I need to manually add rotation keys used in
	// linear_transform! That should have been handled by the above code...
	rotIndices := []uint64{200, 400}
	rotIndices = append(rotIndices, makeRange(100)...)
	// fmt.Printf("Adding extra rot indices: %v\n", rotIndices)

	for _, rotIndex := range rotIndices {
		galoisElement := uint64(1)
		for j := uint64(0); j < rotIndex; j++ {
			galoisElement = (galoisElement * 5) % (1 << (logN + 1))
		}
		galEls = append(galEls, galoisElement)
	}
	fmt.Printf("Final galEls: %v\n", galEls)

	evk := rlwe.NewMemEvaluationKeySet(rk, kgen.GenGaloisKeysNew(galEls, sk)...)
	evaluator := ckks.NewEvaluator(param, evk)

	pt := ckks.NewPlaintext(param, param.MaxLevel())
	encoder.Encode(inputClear, pt)
	ctInput, err25 := encryptor.EncryptNew(pt)
	if err25 != nil {
		panic(err25)
	}

	startTime := time.Now()
	resultCt := _hecate_MLP(evaluator, param, encoder, ctInput)
	duration := time.Since(startTime)
	fmt.Printf("MLP call took: %v\n", duration)
	resultPt := decryptor.DecryptNew(resultCt)
	resultFloat64 := make([]float64, numSlots)
	encoder.Decode(resultPt, resultFloat64)
}
