package pooling

import (
	"math"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/pooling/pooling_utils"
)

func TestPooling(t *testing.T) {
	evaluator, params, ecd, enc, dec := pooling__configure()

	// Input: 1x4x28x28 = 3136 elements
	arg0 := make([]float32, 3136)
	for i := range arg0 {
		arg0[i] = float32(1.0)
	}

	// Filter: 4x4x2x2 = 64 elements, all 0.25 for average pooling
	arg1 := make([]float32, 64)
	for f := 0; f < 4; f++ {
		for c := 0; c < 4; c++ {
			for hi := 0; hi < 2; hi++ {
				for wi := 0; wi < 2; wi++ {
					idx := f*4*2*2 + c*2*2 + hi*2 + wi
					if f == c {
						arg1[idx] = 0.25
					} else {
						arg1[idx] = 0
					}
				}
			}
		}
	}

	// Expected output: 1x4x14x14 = 784 elements
	expected := make([]float32, 784)
	// Compute expected average pooling 2x2 with stride 2
	// N=1, C=4, H=28, W=28
	// Output: N=1, F=4, Ho=14, Wo=14
	// pooling[n, f, ho, wo] = sum_{c, hi, wi} input[n, c, ho*2+hi, wo*2+wi] * filter[f, c, hi, wi]
	for f := 0; f < 4; f++ {
		for ho := 0; ho < 14; ho++ {
			for wo := 0; wo < 14; wo++ {
				var sum float32
				for c := 0; c < 4; c++ {
					for hi := 0; hi < 2; hi++ {
						for wi := 0; wi < 2; wi++ {
							inIdx := c*28*28 + (ho*2+hi)*28 + (wo*2 + wi)
							filterIdx := f*4*2*2 + c*2*2 + hi*2 + wi
							sum += arg0[inIdx] * arg1[filterIdx]
						}
					}
				}
				expected[f*14*14+ho*14+wo] = sum
			}
		}
	}

	ct0 := pooling__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startPre := time.Now()
	filterPlains := pooling_utils.Pooling__preprocessing(params, ecd)
	t.Logf("Preprocessing took %s", time.Since(startPre))

	start := time.Now()
	resultCt := pooling__preprocessed(evaluator, params, ecd, ct0, filterPlains)
	t.Logf("Pooling (preprocessed) took %s", time.Since(start))

	result := pooling__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
