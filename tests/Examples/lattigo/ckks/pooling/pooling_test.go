package pooling

import (
	"fmt"
	"testing"
	"time"
)

func TestPooling(t *testing.T) {
	evaluator, params, ecd, enc, dec := pooling__configure()

	// Input is a tensor of shape 1x2x6x6
	arg0 := [1][2][6][6]float64{
		{
			{
				{0.0, 0.1, 0.2, 0.3, 0.4, 0.5},
				{0.6, 0.7, 0.8, 0.9, 1.0, 1.1},
				{1.2, 1.3, 1.4, 1.5, 1.6, 1.7},
				{1.8, 1.9, 2.0, 2.1, 2.2, 2.3},
				{2.4, 2.5, 2.6, 2.7, 2.8, 2.9},
				{3.0, 3.1, 3.2, 3.3, 3.4, 3.5},
			},
			{
				{3.6, 3.7, 3.8, 3.9, 4.0, 4.1},
				{4.2, 4.3, 4.4, 4.5, 4.6, 4.7},
				{4.8, 4.9, 5.0, 5.1, 5.2, 5.3},
				{5.4, 5.5, 5.6, 5.7, 5.8, 5.9},
				{6.0, 6.1, 6.2, 6.3, 6.4, 6.5},
				{6.6, 6.7, 6.8, 6.9, 7.0, 7.1},
			},
		},
	}
	expected := [1][2][3][3]float64{
		{
			{
				{0.35, 0.55, 0.75},
				{1.55, 1.75, 1.95},
				{2.75, 2.95, 3.15},
			},
			{
				{3.95, 4.15, 4.35},
				{5.15, 5.35, 5.55},
				{6.35, 6.55, 6.75},
			},
		},
	}

	ct0 := pooling__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	startTime := time.Now()
	resultCt := pooling(evaluator, params, ecd, ct0)
	duration := time.Since(startTime)
	fmt.Printf("pooling call took: %v\n", duration)

	result := pooling__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// errorThreshold := float64(0.0001)
	// for i := 0; i < len(result); i++ {
	// 	if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
	// 		t.Errorf("Decryption error %.2f != %.2f", result[i], expected[i])
	// 	}
	// }
}
