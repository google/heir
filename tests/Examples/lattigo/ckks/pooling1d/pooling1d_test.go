package pooling1d

import (
	"math"
	"testing"
	"time"

	"google3/third_party/heir/tests/Examples/lattigo/ckks/pooling1d/pooling1d_utils"
)

func TestPooling(t *testing.T) {
	evaluator, params, ecd, enc, dec := pooling1d__configure()

	// Input: 1x4x28 = 112 elements
	arg0 := make([]float32, 112)
	for i := range arg0 {
		arg0[i] = float32(1.0)
	}

	// Filter: 4x4x2 = 32 elements, all 0.5 for average pooling
	arg1 := make([]float32, 32)
	for f := 0; f < 4; f++ {
		for c := 0; c < 4; c++ {
			for wi := 0; wi < 2; wi++ {
				idx := f*4*2 + c*2 + wi
				if f == c {
					arg1[idx] = 0.5
				} else {
					arg1[idx] = 0
				}
			}
		}
	}

	// Expected output: 1x4x14 = 56 elements
	expected := make([]float32, 56)
	// Compute expected average pooling kernel-size=2 with stride 2
	// N=1, C=4, W=28
	// Output: N=1, F=4, Wo=14
	// pooling[n, f, wo] = sum_{c, wi} input[n, c, wo*2+wi] * filter[f, c, wi]
	for f := 0; f < 4; f++ {
		for wo := 0; wo < 14; wo++ {
			var sum float32
			for c := 0; c < 4; c++ {
				for wi := 0; wi < 2; wi++ {
					inIdx := c*28 + wo*2 + wi
					filterIdx := f*4*2 + c*2 + wi
					sum += arg0[inIdx] * arg1[filterIdx]
				}
			}
			expected[f*14+wo] = sum
		}
	}

	ct0 := pooling1d__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startPre := time.Now()
	filterPlains := pooling1d_utils.Pooling1d__preprocessing(params, ecd)
	t.Logf("Preprocessing took %s", time.Since(startPre))

	start := time.Now()
	resultCt := pooling1d__preprocessed(evaluator, params, ecd, ct0, filterPlains)
	t.Logf("Pooling1d (preprocessed) took %s", time.Since(start))

	result := pooling1d__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
