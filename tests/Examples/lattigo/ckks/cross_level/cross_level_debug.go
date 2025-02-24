package main

import (
	"fmt"
	"math"
	"strconv"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func __heir_debug(evaluator *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext, debugAttrMap map[string]string) {
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
	value := make([]float64, messageSize)
	pt := decryptor.DecryptNew(ct)
	encoder.Decode(pt, value)
	fmt.Printf("  %v\n", value)

	// print the scale
	fmt.Printf("  Level: %v\n", ct.Level())
	fmt.Printf("  Scale: %v\n", ct.Scale.Log2())

	// Input scale should be double degree: 45 * 2 = 90
	if ct.Level() == 4 && isBlockArgument == "1" {
		if math.Abs(ct.Scale.Log2()-float64(90)) > 1e-3 {
			panic("unexpected scale")
		}
	}
	if ct.Level() == 0 {
		if math.Abs(ct.Scale.Log2()-float64(45)) > 1e-3 {
			panic("unexpected scale")
		}
	}
}
