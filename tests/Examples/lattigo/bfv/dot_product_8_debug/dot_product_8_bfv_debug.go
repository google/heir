// Package dotproduct8bfvdebug is a debug handler callback for the compiled lattigo code.
package dotproduct8bfvdebug

import (
	"fmt"
	"strconv"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

func __heir_debug(evaluator *bgv.Evaluator, param bgv.Parameters, encoder *bgv.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext, debugAttrMap map[string]string) {
	// print op
	isBlockArgument := debugAttrMap["asm.is_block_arg"]
	if isBlockArgument == "1" {
		fmt.Println("Input")
	} else {
		fmt.Println(debugAttrMap["asm.op_name"])
	}

	// print the decryption result
	messageSizeStr := debugAttrMap["message.size"]
	messageSize, err := strconv.Atoi(messageSizeStr)
	if err != nil {
		panic(err)
	}
	value := make([]int64, messageSize)
	pt := decryptor.DecryptNew(ct)
	encoder.Decode(pt, value)
	fmt.Printf("  %v\n", value)

	// print the noise

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
	total -= int(param.LogT()) + 1
	fmt.Printf("  Noise: %.2f Total: %d\n", max, total)

	// print the predicted bound by analysis
	noiseBoundStr, ok := debugAttrMap["noise.bound"]
	if ok {
		noiseBound, err := strconv.ParseFloat(noiseBoundStr, 64)
		if err != nil {
			panic(err)
		}
		fmt.Printf("  Noise Bound: %.2f Gap: %.2f\n", noiseBound, noiseBound-max)
		if noiseBound < max {
			panic("Noise Bound Exceeded")
		}
	}
}
