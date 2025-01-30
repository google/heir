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
}
