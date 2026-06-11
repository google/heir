// Debug for cross-level test
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
		if ct == nil {
			fmt.Println("First ciphertext element is nil")
			return
		}
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

	// print the scale
	fmt.Printf("  Level: %v\n", ct.Level())
	fmt.Printf("  Scale: %v\n", ct.Scale.Uint64())

	if ct.Level() == 0 {
		if ct.Scale.Uint64() != uint64(59945) {
			panic("scale is not correct")
		}
	}
}
