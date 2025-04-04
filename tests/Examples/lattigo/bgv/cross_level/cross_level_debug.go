// Debug for cross-level test
package main

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

	// print the scale
	fmt.Printf("  Level: %v\n", ct.Level())
	fmt.Printf("  Scale: %v\n", ct.Scale.Uint64())

	if ct.Level() == 0 {
		if ct.Scale.Uint64() != uint64(59945) {
			panic("scale is not correct")
		}
	}
}
