package mlp

import (
	"fmt"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestMLP(t *testing.T) {
	evaluator, params, encoder, encryptor, decryptor := mlp__configure()
	slots := params.MaxSlots()

	inputClear := make([]float64, slots)
	for i := range inputClear {
		inputClear[i] = 1.0
	}

	// Cleartext args: row-major flattened weights/biases (arbitrary values for perf testing).
	arg0 := make([]float64, 128*slots) // fc1 weights
	arg1 := make([]float64, slots)     // fc1 bias
	arg2 := make([]float64, 128*slots) // fc2 weights
	arg3 := make([]float64, slots)     // fc2 bias
	arg4 := make([]float64, 137*slots) // fc3 weights
	arg5 := make([]float64, slots)     // fc3 bias
	for i := range arg0 {
		arg0[i] = 1.0
	}
	for i := range arg2 {
		arg2[i] = 1.0
	}
	for i := range arg4 {
		arg4[i] = 1.0
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	if err := encoder.Encode(inputClear, pt); err != nil {
		t.Fatal(err)
	}
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}

	startTime := time.Now()
	resultCt := mlp(evaluator, params, encoder, ct, arg0, arg1, arg2, arg3, arg4, arg5)
	duration := time.Since(startTime)
	fmt.Printf("MLP call took: %v\n", duration)

	resultPt := decryptor.DecryptNew(resultCt)
	resultFloat64 := make([]float64, slots)
	if err := encoder.Decode(resultPt, resultFloat64); err != nil {
		t.Fatal(err)
	}
}
