// Debug func implementation for testing
package main

import (
	"fmt"
	"strconv"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

func __heir_debug(evaluator *bgv.Evaluator, param bgv.Parameters, encoder *bgv.Encoder, decryptor *rlwe.Decryptor, ctObj any, debugAttrMap map[string]string) {
	var ct *rlwe.Ciphertext
	switch v := ctObj.(type) {
	case *rlwe.Ciphertext:
		ct = v
	case []*rlwe.Ciphertext:
		if len(v) == 0 {
			fmt.Println("Empty ciphertext slice")
			return
		}
		fmt.Printf("Ciphertext slice of size %d (debugging first element)\n", len(v))
		ct = v[0]
	default:
		panic(fmt.Sprintf("unexpected type %T", ctObj))
	}
	// print op
	isBlockArgument := debugAttrMap["asm.is_block_arg"]
	if isBlockArgument == "1" {
		fmt.Println("Input")
	} else {
		fmt.Println(debugAttrMap["asm.op_name"])
	}

	// print the decryption result
	messageSizeStr, ok := debugAttrMap["message.size"]
	var messageSize int
	var err error
	if !ok || messageSizeStr == "" {
		fmt.Println("Warning: message.size missing, defaulting to 1")
		messageSize = 1
	} else {
		messageSize, err = strconv.Atoi(messageSizeStr)
		if err != nil {
			fmt.Printf("Warning: invalid message.size %s, defaulting to 1\n", messageSizeStr)
			messageSize = 1
		}
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
	// t * e for BGV
	fmt.Printf("  Noise: %.2f Total: %d\n", max+param.LogT(), total)

	// print the predicted bound by analysis
	noiseBoundStr, ok := debugAttrMap["noise.bound"]
	if ok {
		noiseBound, err := strconv.ParseFloat(noiseBoundStr, 64)
		if err != nil {
			panic(err)
		}
		fmt.Printf("  Noise Bound: %.2f Gap: %.2f\n", noiseBound, noiseBound-(max+param.LogT()))
		if noiseBound < max+param.LogT() {
			panic("Noise Bound Exceeded")
		}
	}
}
