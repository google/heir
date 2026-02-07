package mnist

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	modelPath  = "tests/Examples/common/mnist/data/traced_model.pt"
	imagesPath = "tests/Examples/common/mnist/data/t10k-images-idx3-ubyte"
	labelsPath = "tests/Examples/common/mnist/data/t10k-labels-idx1-ubyte"
)

func loadWeights(path string) ([][]float64, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	weights := make([][]float64, 4)
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
		weights[i] = make([]float64, numFloats)
		for j := 0; j < numFloats; j++ {
			bits := binary.LittleEndian.Uint32(data[j*4 : (j+1)*4])
			weights[i][j] = float64(math.Float32frombits(bits))
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

	labels, err := loadMNISTLabels(labelsPath)
	if err != nil {
		t.Fatalf("Failed to load labels: %v", err)
	}

	// These parameters should match what heir-opt uses
	// LogN=10 (ciphertext-degree=1024)
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            10,
		Q:               []uint64{0x10000000001, 0x20000001, 0x20000001},
		P:               []uint64{0x40000001},
		LogDefaultScale: 25,
	})
	if err != nil {
		t.Fatalf("Failed to create parameters: %v", err)
	}

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)

	// In a real scenario we'd need rotation keys for linear transforms
	// but here we assume the generated 'mnist' function handles it or
	// we'll need to add them if it fails.
	evk := rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk))
	evaluator := ckks.NewEvaluator(params, evk)

	total := 4
	correct := 0

	for i := 0; i < total; i++ {
		input := images[i]
		label := labels[i]

		pt := ckks.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(input, pt)
		ctInput, err := encryptor.EncryptNew(pt)
		if err != nil {
			t.Fatalf("Failed to encrypt: %v", err)
		}

		startTime := time.Now()
		// Calling the generated mnist function
		// Signature: func mnist(evaluator *ckks.Evaluator, params ckks.Parameters, encoder *ckks.Encoder,
		//                     v0 []float64, v1 []float64, v2 []float64, v3 []float64, v4 *rlwe.Ciphertext) *rlwe.Ciphertext
		resCt := mnist(evaluator, params, encoder, weights[0], weights[1], weights[2], weights[3], ctInput)
		duration := time.Since(startTime)
		t.Logf("Sample %d took %v", i, duration)

		resPt := decryptor.DecryptNew(resCt)
		resValues := make([]float64, params.MaxSlots())
		encoder.Decode(resPt, resValues)

		maxVal := -math.MaxFloat64
		maxIdx := -1
		for j := 0; j < 10; j++ {
			if resValues[j] > maxVal {
				maxVal = resValues[j]
				maxIdx = j
			}
		}

		if maxIdx == label {
			correct++
		}
		t.Logf("Sample %d: predicted %d, actual %d", i, maxIdx, label)
	}

	accuracy := float64(correct) / float64(total)
	t.Logf("Accuracy: %.2f", accuracy)
	if accuracy < 0.75 {
		t.Errorf("Accuracy too low: %.2f", accuracy)
	}
}
