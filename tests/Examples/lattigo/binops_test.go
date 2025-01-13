package binops

import (
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

func TestBinops(t *testing.T) {
	var err error
	var params bgv.Parameters

	// 128-bit secure parameters enabling depth-7 circuits.
	// LogN:14, LogQP: 431.
	if params, err = bgv.NewParametersFromLiteral(
		bgv.ParametersLiteral{
			LogN:             14,                                    // log2(ring degree)
			LogQ:             []int{55, 45, 45, 45, 45, 45, 45, 45}, // log2(primes Q) (ciphertext modulus)
			LogP:             []int{61},                             // log2(primes P) (auxiliary modulus)
			PlaintextModulus: 0x10001,                               // log2(scale)
		}); err != nil {
		panic(err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	ecd := bgv.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)
	relinKeys := kgen.GenRelinearizationKeyNew(sk)
	galKey := kgen.GenGaloisKeyNew(5, sk)
	evalKeys := rlwe.NewMemEvaluationKeySet(relinKeys, galKey)
	evaluator := bgv.NewEvaluator(params, evalKeys /*scaleInvariant=*/, false)

	T := params.MaxSlots()
	// Vector of plaintext values
	// 0, 1, 2, 3
	arg0 := make([]uint64, T)
	// 1, 2, 3, 4
	arg1 := make([]uint64, T)

	expected := make([]uint64, T)
	result := make([]uint64, T)

	// Hack until we have packing system: replicate values cycling every 4
	dataSize := 4
	for i := range arg0 {
		arg0[i] = uint64(i % dataSize)
		arg1[i] = (uint64(i%dataSize) + 1)
		expected[i] = (arg0[i] + arg1[i]) * arg1[i]
	}

	// Rotate by 1
	tmp := expected[0]
	for i := range T - 1 {
		expected[i] = expected[i+1]
	}
	expected[T-1] = tmp

	// Allocates a plaintext at the max level.
	// Default rlwe.MetaData:
	// - IsBatched = true (slots encoding)
	// - Scale = params.DefaultScale()
	pt0 := bgv.NewPlaintext(params, params.MaxLevel())
	pt1 := bgv.NewPlaintext(params, params.MaxLevel())

	// Encodes the vector of plaintext values
	if err = ecd.Encode(arg0, pt0); err != nil {
		panic(err)
	}
	if err = ecd.Encode(arg1, pt1); err != nil {
		panic(err)
	}

	// Encrypts the vector of plaintext values
	var ct0 *rlwe.Ciphertext
	var ct1 *rlwe.Ciphertext
	if ct0, err = enc.EncryptNew(pt0); err != nil {
		panic(err)
	}
	if ct1, err = enc.EncryptNew(pt1); err != nil {
		panic(err)
	}

	resultCt := add(evaluator, ct0, ct1)
	resultEncoded := dec.DecryptNew(resultCt)
	err = ecd.Decode(resultEncoded, result)
	if err != nil {
		panic(err)
	}

	for i := range 4 {
		if result[i] != expected[i] {
			t.Errorf("Decryption error at index %d: %d != %d", i, result[i], expected[i])
		}
	}
}
