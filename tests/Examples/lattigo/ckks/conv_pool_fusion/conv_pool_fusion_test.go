package conv_pool_fusion

import (
	"math"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/conv_pool_fusion/conv_pool_fusion_utils"
)

func TestConvPoolFusion(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv_pool__configure()

	// Input: 1x1x8x8 = 64 elements, all 1.0
	arg0 := make([]float32, 64)
	for i := range arg0 {
		arg0[i] = float32(1.0)
	}

	// Expected output: 1x4x3x3 = 36 elements
	// Channel 0: all 9.5, Channel 1: all 18.5, Channel 2: all 5.0, Channel 3: all -8.5
	expected := make([]float32, 36)
	for c := 0; c < 4; c++ {
		var val float32
		switch c {
		case 0:
			val = 9.5
		case 1:
			val = 18.5
		case 2:
			val = 5.0
		case 3:
			val = -8.5
		}
		for h := 0; h < 3; h++ {
			for w := 0; w < 3; w++ {
				expected[c*3*3+h*3+w] = val
			}
		}
	}

	ct0 := Conv_pool__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	zeroCt := Conv_pool__encrypt__zero__0(evaluator, params, ecd, enc)

	startPre := time.Now()
	filterPlains := conv_pool_fusion_utils.Conv_pool__preprocessing(params, ecd)
	t.Logf("Preprocessing took %s", time.Since(startPre))

	start := time.Now()
	resultCt := Conv_pool__preprocessed(evaluator, params, ecd, ct0, zeroCt, filterPlains)
	t.Logf("ConvPool (preprocessed) took %s", time.Since(start))

	result := Conv_pool__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.05) // CKKS noise threshold
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d (Channel %d): %.4f != %.4f", i, i/(3*3), result[i], expected[i])
		}
	}
}
