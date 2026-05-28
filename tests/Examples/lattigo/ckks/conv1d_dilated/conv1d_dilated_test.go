package conv1d_dilated

import (
	"math"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/conv1d_dilated/conv1d_dilated_utils"
)

func TestConv1DDilated(t *testing.T) {
	evaluator, params, ecd, enc, dec := conv1d_dilated__configure()

	// Input: 1x1x28
	const W = 28
	arg0 := make([]float32, W)
	for i := range arg0 {
		arg0[i] = float32(i)
	}

	// Filter is dense<1.0> : tensor<4x1x3xf32>, dilation=2, stride=1.
	// New filter width after undilation is d*(KW-1)+1 = 5: [1, 0, 1, 0, 1].
	// Output[f, wo] = arg0[wo] + arg0[wo+2] + arg0[wo+4] for every output
	// channel f, since the filter is all-ones.
	const F = 4
	const KW = 3
	const D = 2
	const OW = W - D*(KW-1) // = 24 with stride 1

	expected := make([]float32, F*OW)
	for f := 0; f < F; f++ {
		for wo := 0; wo < OW; wo++ {
			var sum float32
			for ki := 0; ki < KW; ki++ {
				sum += arg0[wo+D*ki]
			}
			expected[f*OW+wo] = sum
		}
	}

	ct0 := conv1d_dilated__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startPre := time.Now()
	filterPlains := conv1d_dilated_utils.Conv1d_dilated__preprocessing(params, ecd)
	t.Logf("Preprocessing took %s", time.Since(startPre))

	start := time.Now()
	resultCt := conv1d_dilated__preprocessed(evaluator, params, ecd, ct0, filterPlains)
	t.Logf("Conv1d_dilated (preprocessed) took %s", time.Since(start))

	result := conv1d_dilated__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.5)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
