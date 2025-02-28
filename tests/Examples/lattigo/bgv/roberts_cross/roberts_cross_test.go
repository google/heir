package robertscross

import (
	"testing"
)

func TestBinops(t *testing.T) {
	evaluator, params, ecd, enc, dec := roberts_cross__configure()

	input := make([]int16, 4096)
	expected := make([]int16, 4096)

	for i := 0; i < 4096; i++ {
		input[i] = int16(i)
	}

	for row := 0; row < 64; row++ {
		for col := 0; col < 64; col++ {
			xY := (row*64 + col) % 4096
			xYm1 := (row*64 + col - 1) % 4096
			xm1Y := ((row-1)*64 + col) % 4096
			xm1Ym1 := ((row-1)*64 + col - 1) % 4096

			if xYm1 < 0 {
				xYm1 += 4096
			}
			if xm1Y < 0 {
				xm1Y += 4096
			}
			if xm1Ym1 < 0 {
				xm1Ym1 += 4096
			}

			v1 := input[xm1Ym1] - input[xY]
			v2 := input[xm1Y] - input[xYm1]
			sum := v1*v1 + v2*v2
			expected[row*64+col] = sum
		}
	}

	ct0 := roberts_cross__encrypt__arg0(evaluator, params, ecd, enc, input)

	resultCt := roberts_cross(evaluator, params, ecd, ct0)

	result := roberts_cross__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	for i := 0; i < 4096; i++ {
		if result[i] != expected[i] {
			t.Errorf("Decryption error at %d: %d != %d", i, result[i], expected[i])
		}
	}
}
