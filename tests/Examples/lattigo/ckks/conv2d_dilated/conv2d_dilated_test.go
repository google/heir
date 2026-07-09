package conv2d_dilated

import (
	"math"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/conv2d_dilated/conv2d_dilated_utils"
)

func TestConv2DDilated(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_dilated__configure()

	const H = 10
	const W = 10
	arg0 := make([]float32, H*W)
	for i := range arg0 {
		arg0[i] = float32(i)
	}

	// Filter is dense<1.0> : tensor<4x1x3x3xf32>, dilation=2, stride=1.
	// New filter size after undilation is d*(K-1)+1 = 5x5 with zeros inserted
	// between the original taps.
	// Output[f, ho, wo] = sum_{kh, kw} arg0[ho+2*kh, wo+2*kw] for every output
	// channel f, since the filter is all-ones.
	const F = 4
	const K = 3
	const D = 2
	const OH = H - D*(K-1) // = 6 with stride 1
	const OW = W - D*(K-1) // = 6 with stride 1

	expected := make([]float32, F*OH*OW)
	for f := 0; f < F; f++ {
		for ho := 0; ho < OH; ho++ {
			for wo := 0; wo < OW; wo++ {
				var sum float32
				for kh := 0; kh < K; kh++ {
					for kw := 0; kw < K; kw++ {
						sum += arg0[(ho+D*kh)*W+(wo+D*kw)]
					}
				}
				expected[(f*OH+ho)*OW+wo] = sum
			}
		}
	}

	ct0 := Conv2d_dilated__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startPre := time.Now()
	filterPlains := conv2d_dilated_utils.Conv2d_dilated__preprocessing(params, ecd)
	t.Logf("Preprocessing took %s", time.Since(startPre))

	start := time.Now()
	resultCt := Conv2d_dilated__preprocessed(evaluator, params, ecd, ct0, filterPlains)
	t.Logf("Conv2d_dilated (preprocessed) took %s", time.Since(start))

	result := Conv2d_dilated__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.5)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
