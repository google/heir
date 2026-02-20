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

func mnist__configure__override() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor) {
  param, err1073 := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
    LogN: 13,
    Q: []uint64{36028797018652673, 35184372121601},
    P: []uint64{1152921504606994433},
    LogDefaultScale: 45,
  })
  if err1073 != nil {
    panic(err1073)
  }
  encoder := ckks.NewEncoder(param)
  kgen := rlwe.NewKeyGenerator(param)
  sk, pk := kgen.GenKeyPairNew()
  encryptor := rlwe.NewEncryptor(param, pk)
  decryptor := rlwe.NewDecryptor(param, sk)
  gk := kgen.GenGaloisKeyNew(5, sk)
  gk1 := kgen.GenGaloisKeyNew(25, sk)
  gk2 := kgen.GenGaloisKeyNew(125, sk)
  gk3 := kgen.GenGaloisKeyNew(625, sk)
  gk4 := kgen.GenGaloisKeyNew(3125, sk)
  gk5 := kgen.GenGaloisKeyNew(15625, sk)
  gk6 := kgen.GenGaloisKeyNew(12589, sk)
  gk7 := kgen.GenGaloisKeyNew(13793, sk)
  gk8 := kgen.GenGaloisKeyNew(3429, sk)
  gk9 := kgen.GenGaloisKeyNew(761, sk)
  gk10 := kgen.GenGaloisKeyNew(3805, sk)
  gk11 := kgen.GenGaloisKeyNew(2641, sk)
  gk12 := kgen.GenGaloisKeyNew(13205, sk)
  gk13 := kgen.GenGaloisKeyNew(489, sk)
  gk14 := kgen.GenGaloisKeyNew(2445, sk)
  gk15 := kgen.GenGaloisKeyNew(12225, sk)
  gk16 := kgen.GenGaloisKeyNew(11973, sk)
  gk17 := kgen.GenGaloisKeyNew(10713, sk)
  gk18 := kgen.GenGaloisKeyNew(4413, sk)
  gk19 := kgen.GenGaloisKeyNew(5681, sk)
  gk20 := kgen.GenGaloisKeyNew(12021, sk)
  gk21 := kgen.GenGaloisKeyNew(10953, sk)
  gk22 := kgen.GenGaloisKeyNew(5613, sk)
  gk23 := kgen.GenGaloisKeyNew(15721, sk)
  gk24 := kgen.GenGaloisKeyNew(14133, sk)
  gk25 := kgen.GenGaloisKeyNew(13585, sk)
  gk26 := kgen.GenGaloisKeyNew(1469, sk)
  gk27 := kgen.GenGaloisKeyNew(4345, sk)
  gk28 := kgen.GenGaloisKeyNew(9093, sk)
  gk29 := kgen.GenGaloisKeyNew(2849, sk)
  gk30 := kgen.GenGaloisKeyNew(653, sk)
  gk31 := kgen.GenGaloisKeyNew(11657, sk)
  gk32 := kgen.GenGaloisKeyNew(9429, sk)
  gk33 := kgen.GenGaloisKeyNew(4657, sk)
  gk34 := kgen.GenGaloisKeyNew(7261, sk)
  gk35 := kgen.GenGaloisKeyNew(8985, sk)
  gk36 := kgen.GenGaloisKeyNew(2853, sk)
  gk37 := kgen.GenGaloisKeyNew(6721, sk)
  gk38 := kgen.GenGaloisKeyNew(9005, sk)
  gk39 := kgen.GenGaloisKeyNew(425, sk)
  gk40 := kgen.GenGaloisKeyNew(9845, sk)
  gk41 := kgen.GenGaloisKeyNew(13137, sk)
  gk42 := kgen.GenGaloisKeyNew(9981, sk)
  gk43 := kgen.GenGaloisKeyNew(6457, sk)
  gk44 := kgen.GenGaloisKeyNew(14337, sk)
  ekset := rlwe.NewMemEvaluationKeySet(nil, gk, gk1, gk2, gk3, gk4, gk5, gk6, gk7, gk8, gk9, gk10, gk11, gk12, gk13, gk14, gk15, gk16, gk17, gk18, gk19, gk20, gk21, gk22, gk23, gk24, gk25, gk26, gk27, gk28, gk29, gk30, gk31, gk32, gk33, gk34, gk35, gk36, gk37, gk38, gk39, gk40, gk41, gk42, gk43, gk44)
  evaluator := ckks.NewEvaluator(param, ekset)
  return evaluator, param, encoder, encryptor, decryptor
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

	evaluator, params, encoder, encryptor, decryptor := mnist__configure__override()
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
