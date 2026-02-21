package mnist_layer1

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
	"time"
)

// Unlike other languages where the cwd of a test is the runfiles root, in
// golang it is the current package directory. Maybe it would make sense to
// ultimately have a helper to find the root (by looking for, say,
// MODULE.bazel) but for now we can just hardcode the relative path to the
// data.
const (
	modelPath  = "../../../common/mnist/data/traced_model.pt"
	imagesPath = "../../../common/mnist/data/t10k-images-idx3-ubyte"
	labelsPath = "../../../common/mnist/data/t10k-labels-idx1-ubyte"
)

func loadWeights(path string) ([][]float32, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	weights := make([][]float32, 4)
	for i := 0; i < 4; i++ {
		f, err := r.Open(fmt.Sprintf("traced_model/data/%d", i))
		if err != nil {
			return nil, err
		}
		defer f.Close()

		data, err := io.ReadAll(f)
		if err != nil {
			return nil, err
		}

		numFloats := len(data) / 4
		weights[i] = make([]float32, numFloats)
		for j := 0; j < numFloats; j++ {
			bits := binary.LittleEndian.Uint32(data[j*4 : (j+1)*4])
			weights[i][j] = math.Float32frombits(bits)
		}
	}
	return weights, nil
}

func loadMNISTImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 16)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numImages := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	pixelsPerImage := rows * cols
	images := make([][]float64, numImages)
	for i := 0; i < numImages; i++ {
		imgData := make([]byte, pixelsPerImage)
		if _, err := f.Read(imgData); err != nil {
			return nil, err
		}
		images[i] = make([]float64, pixelsPerImage)
		for j := 0; j < pixelsPerImage; j++ {
			// Normalize: (X/255.0 - 0.1307) / 0.3081
			val := float64(imgData[j]) / 255.0
			images[i][j] = (val - 0.1307) / 0.3081
		}
	}
	return images, nil
}

func loadMNISTLabels(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 8)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numLabels := int(binary.BigEndian.Uint32(header[4:8]))
	labels := make([]int, numLabels)
	labelData := make([]byte, numLabels)
	if _, err := f.Read(labelData); err != nil {
		return nil, err
	}
	for i := 0; i < numLabels; i++ {
		labels[i] = int(labelData[i])
	}
	return labels, nil
}

func TestMNIST(t *testing.T) {
	weights, err := loadWeights(modelPath)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	images, err := loadMNISTImages(imagesPath)
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}

	evaluator, params, encoder, encryptor, decryptor := mnist__configure()
	input := images[0]

	// Convert float64 input to float32 for encryption helper
	// FIXME: revert from constant 0.1's to real data once bug is fixed
	inputFloat32 := make([]float32, len(input))
	for j := 0; j < len(input); j++ {
		inputFloat32[j] = float32(0.1)
	}

	// Dump the input for debugging/comparison with openfhe
	// fmt.Printf("Input:\n")
	// for i := 0; i < len(inputFloat32); i++ {
	// 	fmt.Printf("%d, %.6f\n", i, inputFloat32[i])
	// }

	ctInput := mnist__encrypt__arg4(evaluator, params, encoder, encryptor, inputFloat32)
	startTime := time.Now()
	// Calling the generated mnist function
	// Signature: func mnist(evaluator *ckks.Evaluator, params ckks.Parameters, encoder *ckks.Encoder,
	//                     v0 []float32, v1 []float32, v2 []float32, v3 []float32, v4 []*rlwe.Ciphertext) []*rlwe.Ciphertext
	resCt := mnist(evaluator, params, encoder, weights[0], weights[1], weights[2], weights[3], ctInput)
	duration := time.Since(startTime)
	t.Logf("Sample %d took %v", 0, duration)

	// Use generated decryption helper
	resValues := mnist__decrypt__result0(evaluator, params, encoder, decryptor, resCt)

	// These ten values are taken from the openfhe analogue, which is treated
	// as a source of truth for the sake of this debugging hell.
	// expectedFirstTen := []float32{0.338506, -0.138258, 0.103439, -0.327988, -0.398066, -0.411804, 0.144839, -0.398289, -0.360264, 0.438655, -0.225345}

	fmt.Printf("output:\n")
	for j := 0; j < 512; j++ {
		fmt.Printf("%d, %.6f\n", j, resValues[j])
	}
	t.Errorf("xFail to dump logs")

	// for j := 0; j < 10; j++ {
	// 	if math.Abs(float64(resValues[j]-expectedFirstTen[j])) > 0.0001 {
	// 		t.Errorf("Decryption error at index %d: %.6f != %.6f", j, resValues[j], expectedFirstTen[j])
	// 	}
	// }
}
