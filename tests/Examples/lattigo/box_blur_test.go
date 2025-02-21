package boxblur

import (
	"testing"
)

func TestBinops(t *testing.T) {
	// TODO(#1186): reenable this test
	// Disabled now for logN = 12 in parameter generation
	// which does not satisfy the requirement of logN >= 13
	// for 4096 input size
	t.Skip("Disabled because the parameters generated are too small.")
	evaluator, params, ecd, enc, dec := box_blur__configure()

	input := make([]int16, 4096)
	expected := make([]int16, 4096)

	for i := 0; i < 4096; i++ {
		input[i] = int16(i)
	}

	for row := 0; row < 64; row++ {
		for col := 0; col < 64; col++ {
			sum := int16(0)
			for di := -1; di < 2; di++ {
				for dj := -1; dj < 2; dj++ {
					index := (row*64 + col + di*64 + dj) % 4096
					if index < 0 {
						index += 4096
					}
					sum += input[index]
				}
			}
			expected[row*64+col] = sum
		}
	}

	ct0 := box_blur__encrypt__arg0(evaluator, params, ecd, enc, input)

	resultCt := box_blur(evaluator, params, ecd, ct0)

	result := box_blur__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	for i := 0; i < 4096; i++ {
		if result[i] != expected[i] {
			t.Errorf("Decryption error at %d: %d != %d", i, result[i], expected[i])
		}
	}
}
