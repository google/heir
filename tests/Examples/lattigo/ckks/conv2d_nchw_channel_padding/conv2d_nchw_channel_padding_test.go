package conv2dnchwchannelpadding

import (
	"math"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/conv2d_nchw_channel_padding/conv2dnchwchannelpadding_utils"
)

// conv2dNchwFchw computes a 2D convolution over a row-major NCHW input tensor
// with an FCHW filter tensor.
func conv2dNchwFchw(
	input []float32, filter []float32,
	N, Cin, Hin, Win int,
	Cout, Kh, Kw int,
	strideH, strideW int,
) []float32 {
	Hout := (Hin-Kh)/strideH + 1
	Wout := (Win-Kw)/strideW + 1

	output := make([]float32, N*Cout*Hout*Wout)

	for n := 0; n < N; n++ {
		for f := 0; f < Cout; f++ {
			for ho := 0; ho < Hout; ho++ {
				for wo := 0; wo < Wout; wo++ {
					var sum float32
					for c := 0; c < Cin; c++ {
						for kh := 0; kh < Kh; kh++ {
							for kw := 0; kw < Kw; kw++ {
								hi := ho*strideH + kh
								wi := wo*strideW + kw
								inIdx := n*(Cin*Hin*Win) + c*(Hin*Win) + hi*Win + wi
								filterIdx := f*(Cin*Kh*Kw) + c*(Kh*Kw) + kh*Kw + kw
								sum += input[inIdx] * filter[filterIdx]
							}
						}
					}
					outIdx := n*(Cout*Hout*Wout) + f*(Hout*Wout) + ho*Wout + wo
					output[outIdx] = sum
				}
			}
		}
	}
	return output
}

// The conv has 3 output channels at stride 2, so the pixel-shuffled layout has
// to reserve a full block of gap^2 = 4 channels. The fourth channel is empty
// and must not disturb the three real ones.
func TestConv2dChannelPadding(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_channel_pad__configure()

	// Input: 1x1x4x4 = 16 elements (values 1..16)
	arg0 := make([]float32, 16)
	for i := range arg0 {
		arg0[i] = float32(i + 1)
	}
	filter := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	expected := conv2dNchwFchw(arg0, filter, 1, 1, 4, 4, 3, 2, 2, 2, 2)

	ct0 := Conv2d_channel_pad__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	startPre := time.Now()
	filterPlains := conv2dnchwchannelpadding_utils.Conv2d_channel_pad__preprocessing(params, ecd)
	t.Logf("Preprocessing took %s", time.Since(startPre))
	start := time.Now()
	resultCt := Conv2d_channel_pad__preprocessed(evaluator, params, ecd, ct0, filterPlains)
	t.Logf("Conv2d (preprocessed) took %s", time.Since(start))
	result := Conv2d_channel_pad__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: got %.4f, expected %.4f", i, result[i], expected[i])
		}
	}
}
