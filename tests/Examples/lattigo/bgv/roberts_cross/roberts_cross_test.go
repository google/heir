package robertscross

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := roberts_cross__configure()

	input := make([]int16, 256)
	expected := make([]int16, 256)

	for i := 0; i < 256; i++ {
		input[i] = int16(i)
	}

	for row := 0; row < 16; row++ {
		for col := 0; col < 16; col++ {
			xY := (row*16 + col) % 256
			xYm1 := (row*16 + col - 1) % 256
			xm1Y := ((row-1)*16 + col) % 256
			xm1Ym1 := ((row-1)*16 + col - 1) % 256

			if xYm1 < 0 {
				xYm1 += 256
			}
			if xm1Y < 0 {
				xm1Y += 256
			}
			if xm1Ym1 < 0 {
				xm1Ym1 += 256
			}

			v1 := input[xm1Ym1] - input[xY]
			v2 := input[xm1Y] - input[xYm1]
			sum := v1*v1 + v2*v2
			expected[row*16+col] = sum
		}
	}

	ct0 := roberts_cross__encrypt__arg0(evaluator, params, ecd, enc, input)

	resultCt := roberts_cross(evaluator, params, ecd, ct0)

	result := roberts_cross__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	for i := 0; i < 256; i++ {
		if result[i] != expected[i] {
			t.Errorf("Decryption error at %d: %d != %d", i, result[i], expected[i])
		}
	}
}
