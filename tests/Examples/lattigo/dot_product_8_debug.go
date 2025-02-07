// Package dotproduct8debug is a debug handler callback for the compiled lattigo code.
package dotproduct8debug

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

func __heir_debug(evaluator *bgv.Evaluator, param bgv.Parameters, encoder *bgv.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext) {
	value := make([]int64, 8)
	pt := decryptor.DecryptNew(ct)
	encoder.Decode(pt, value)
	fmt.Println(value)

	// get a new pt with no noise
	// in Lattigo, Decrypt won't mod T
	// Decode will mod T so we Decode then Encode
	valueFull := make([]int64, ct.N())
	encoder.Decode(pt, valueFull)
	// set the level and scale of the plaintext
	ptNoNoise := bgv.NewPlaintext(param, ct.Level())
	ptNoNoise.Scale = ct.Scale
	encoder.Encode(valueFull, ptNoNoise)

	// subtract the message from the ciphertext
	vec, _ := evaluator.SubNew(ct, ptNoNoise)
	// get infty norm of the noise
	_, _, max := rlwe.Norm(vec, decryptor)
	// get logQi for current level
	total := 0
	for i := 0; i <= ct.LevelQ(); i++ {
		total += param.LogQi()[i]
	}
	// t * e for BGV
	fmt.Printf("Noise: %.2f Total: %d\n", max+param.LogT(), total)
}
