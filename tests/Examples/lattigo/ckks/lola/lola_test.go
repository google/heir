package lola

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/lola/lola_utils"
)

const imagesPath = "../../../common/mnist/data/t10k-images-idx3-ubyte"

func loadFirstMNISTImage(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [16]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, err
	}
	if magic := binary.BigEndian.Uint32(header[0:4]); magic != 2051 {
		return nil, fmt.Errorf("invalid MNIST image magic: got %d, want 2051", magic)
	}
	rows := binary.BigEndian.Uint32(header[8:12])
	cols := binary.BigEndian.Uint32(header[12:16])
	if rows != 28 || cols != 28 {
		return nil, fmt.Errorf("invalid MNIST image shape: got %dx%d, want 28x28", rows, cols)
	}

	raw := make([]byte, rows*cols)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, err
	}
	image := make([]float32, len(raw))
	for i, pixel := range raw {
		// Orion trains LoLA with torchvision's MNIST normalization.
		image[i] = (float32(pixel)/255.0 - 0.1307) / 0.3081
	}
	return image, nil
}

// Run the LoLA MNIST CNN end-to-end under CKKS on Lattigo.
// The test input is MNIST test image 0, the expected logits are from
// running Orion's LoLA model (checkpoints/mnist/lola.pth) on that input.
func TestLoLA(t *testing.T) {
	evaluator, params, ecd, enc, dec := Lola__configure()

	arg0, err := loadFirstMNISTImage(imagesPath)
	if err != nil {
		t.Fatalf("load first MNIST test image: %v", err)
	}

	ct0 := Lola__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	// The embedded constant weights are encoded into plaintexts once, up front.
	startPre := time.Now()
	weightPlains := lola_utils.Lola__preprocessing(params, ecd)
	t.Logf("Lola__preprocessing took %s", time.Since(startPre))

	start := time.Now()
	resultCt := Lola__preprocessed(evaluator, params, ecd, ct0, weightPlains)
	t.Logf("Lola__preprocessed took %s", time.Since(start))

	result := Lola__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	expected := [...]float64{
		-8.300922393798828,
		-5.588466644287109,
		-3.6352932453155518,
		-7.802109241485596,
		-3.84030818939209,
		-5.856731414794922,
		-5.982525825500488,
		11.231959342956543,
		-3.0315234661102295,
		-3.8313705921173096,
	}
	const tolerance = 0.1
	const expectedClass = 7
	const numLogits = len(expected)
	if len(result) < numLogits {
		t.Fatalf("expected at least %d output logits, got %d", numLogits, len(result))
	}
	predictedClass := 0
	for i := 0; i < numLogits; i++ {
		got := float64(result[i])
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("output logit %d is not finite: %v", i, result[i])
		}
		if delta := math.Abs(got - expected[i]); delta > tolerance {
			t.Errorf("logit %d: got %.6f, want %.6f (+/- %.2f), delta %.6f", i, got, expected[i], tolerance, delta)
		}
		if result[i] > result[predictedClass] {
			predictedClass = i
		}
		t.Logf("logit %d: got %.6f, plaintext %.6f", i, got, expected[i])
	}
	if predictedClass != expectedClass {
		t.Errorf("predicted class %d, want %d", predictedClass, expectedClass)
	}
}
