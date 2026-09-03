package conv1dncwchannelpadding

import (
	"math"
	"testing"
)

// conv1dNcwFcw computes a 1D convolution over a row-major NCW input tensor with
// an FCW filter tensor.
func conv1dNcwFcw(input []float32, filter []float32, Cin, Win, Cout, Kw, stride int) []float32 {
	Wout := (Win-Kw)/stride + 1
	output := make([]float32, Cout*Wout)
	for f := 0; f < Cout; f++ {
		for wo := 0; wo < Wout; wo++ {
			var sum float32
			for c := 0; c < Cin; c++ {
				for kw := 0; kw < Kw; kw++ {
					sum += input[c*Win+wo*stride+kw] * filter[f*(Cin*Kw)+c*Kw+kw]
				}
			}
			output[f*Wout+wo] = sum
		}
	}
	return output
}

// The conv has 3 output channels at stride 2, so the shuffled layout reserves a
// whole pair of channels. The empty one must not disturb the three real ones.
func TestConv1dChannelPadding(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv1d_channel_pad__configure()

	// Input: 1x2x8 = 16 elements (values 1..16)
	arg0 := make([]float32, 16)
	for i := range arg0 {
		arg0[i] = float32(i + 1)
	}
	filter := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	expected := conv1dNcwFcw(arg0, filter, 2, 8, 3, 2, 2)

	ct0 := Conv1d_channel_pad__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ctZero := Conv1d_channel_pad__encrypt__zero__0(evaluator, params, ecd, enc)
	resultCt := Conv1d_channel_pad(evaluator, params, ecd, ct0, ctZero)
	result := Conv1d_channel_pad__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: got %.4f, expected %.4f", i, result[i], expected[i])
		}
	}
}
