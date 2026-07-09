// Debug func implementation for in_place test
package in_place_lib

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

var ObservedPointers []string
var ObservedOps []string

func __heir_debug(evaluator *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, decryptor *rlwe.Decryptor, ctObj any, debugAttrMap map[string]string) {
	var ct *rlwe.Ciphertext
	switch v := ctObj.(type) {
	case *rlwe.Ciphertext:
		ct = v
	case []*rlwe.Ciphertext:
		if len(v) > 0 {
			ct = v[0]
		}
	}
	if ct != nil {
		ObservedPointers = append(ObservedPointers, fmt.Sprintf("%p", ct))
		ObservedOps = append(ObservedOps, debugAttrMap["debug.name"])
	}
}
