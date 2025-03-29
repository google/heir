// Package dotproduct8fdebug is a debug handler callback for the compiled lattigo code.
package dotproduct8fdebug

import (
	"fmt"
	"math"
	"strconv"
	"strings"

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

	// calculate the precision
	if secretExecutionResult, ok := debugAttrMap["secret.execution_result"]; ok {
		// secretExecutionResult has the form "[1.0, 2.0, 3.0]", parse it into a slice of float64
		plaintextResultStr := strings.Trim(secretExecutionResult, "[]")
		plaintextResultStrs := strings.Split(plaintextResultStr, ",")
		plaintextResult := make([]float64, len(plaintextResultStrs))
		for i, s := range plaintextResultStrs {
			plaintextResult[i], err = strconv.ParseFloat(strings.TrimSpace(s), 64)
			if err != nil {
				panic(err)
			}
		}

		maxError := math.Inf(-1)
		for i := 0; i < len(value) && i < len(plaintextResult); i++ {
			err := math.Log2(math.Abs(value[i] - plaintextResult[i]))
			if err > maxError {
				maxError = err
			}
		}

		fmt.Printf("  Precision lost: 2^%3.1f\n", maxError)
	}
}
