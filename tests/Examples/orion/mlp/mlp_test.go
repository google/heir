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
	lintrans_params := lintrans.Parameters{
		DiagonalsIndexList:        diagonalIndices,
		LevelQ:                    5,
		LevelP:                    param.MaxLevelP(),
		Scale:                     rlwe.NewScale(param.Q()[5]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: 12}, // 1x4096
		LogBabyStepGiantStepRatio: 2,
	}
	lt := lintrans.NewTransformation(param, lintrans_params)
	return lt.GaloisElements(param)
}

func TestMLP(t *testing.T) {
	log_n := 13
	num_slots := 1 << (log_n - 1)

	// Input is arbitrary, doesn't matter since we're just testing
	// performance
	input_clear := make([]float64, num_slots)
	for i := range input_clear {
		input_clear[i] = 1.0
	}

	// Function args:
	//
	// %ct: encrypted input,
	// % arg0: tensor<128x4096xf64> fc1 weights,
	// % arg1: tensor<4096xf64> fc1 bias,
	// % arg2: tensor<128x4096xf64> fc2 weights,
	// % arg3: tensor<4096xf64> fc2 bias,
	// % arg4: tensor<137x4096xf64> fc3 weights,
	// % arg5: tensor<4096xf64> fc3 bias,
	//
	// Cleartext args are converted to row-major flattened 1D slices. Data
	// is arbitrary because we're just testing perf.
	arg0 := MakeFlattenedOnes(128, 4096)
	arg1 := MakeFlattenedOnes(1, 4096)
	arg2 := MakeFlattenedOnes(128, 4096)
	arg3 := MakeFlattenedOnes(1, 4096)
	arg4 := MakeFlattenedOnes(137, 4096)
	arg5 := MakeFlattenedOnes(1, 4096)

	// These parameters should match the mlir file, though due to the weird
	// nature of this test, this is the source of truth for what is used,
	// not the mlir file.
	param, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            log_n,
		Q:               []uint64{536903681, 67043329, 66994177, 67239937, 66961409, 66813953},
		P:               []uint64{536952833, 536690689},
		LogDefaultScale: 26,
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
	galEls := []uint64{}
	all128Indices := make([]int, 0, 128)
	fc3Extra := []int{4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095}
	fc3Indices := append(all128Indices, fc3Extra...)
	galEls = append(galEls, generateLintransGaloisKeys(param, all128Indices)...)
	galEls = append(galEls, generateLintransGaloisKeys(param, all128Indices)...)
	galEls = append(galEls, generateLintransGaloisKeys(param, fc3Indices)...)

	// Manually add Galois key for extra rotation indices used in the
	// mlir file, outside of linear_transform
	//
	// For some reason I need to manually add rotation keys used in
	// linear_transform! That should have been handled by the above code...
	rotIndices := []uint64{2048, 1024, 512, 256, 128}
	rotIndices = append(rotIndices, makeRange(128)...)
	rotIndices = append(rotIndices, 4080)
	// fmt.Printf("Adding extra rot indices: %v\n", rotIndices)

	for _, rotIndex := range rotIndices {
		galoisElement := uint64(1)
		for j := uint64(0); j < rotIndex; j++ {
			galoisElement = (galoisElement * 5) % (1 << (log_n + 1))
		}
		galEls = append(galEls, galoisElement)
	}
	fmt.Printf("Final galEls: %v\n", galEls)

	evk := rlwe.NewMemEvaluationKeySet(rk, kgen.GenGaloisKeysNew(galEls, sk)...)
	evaluator := ckks.NewEvaluator(param, evk)

	pt := ckks.NewPlaintext(param, param.MaxLevel())
	encoder.Encode(input_clear, pt)
	ct_input, err25 := encryptor.EncryptNew(pt)
	if err25 != nil {
		panic(err25)
	}

	startTime := time.Now()
	result_ct := mlp(evaluator, param, encoder, ct_input, arg0, arg1, arg2, arg3, arg4, arg5)
	duration := time.Since(startTime)
	fmt.Printf("MLP call took: %v\n", duration)
	result_pt := decryptor.DecryptNew(result_ct)
	result_float64 := make([]float64, num_slots)
	encoder.Decode(result_pt, result_float64)
}
