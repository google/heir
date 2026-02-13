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
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
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

func encrypt_override(_ *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor, v0 []float32) ([]*rlwe.Ciphertext) {
  v1 := int64(0)
  v2 := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  v3 := int32(0)
  v4 := int32(1)
  v5 := int32(784)
  v6 := v2
  for v7 := v3; v7 < v5; v7 += v4 {
    v9 := int64(v7)
    v10 := v0[v9 + 784 * (v1)]
    v6 = append(make([]float32, 0, len(v6)), v6...)
    v6[v9 + 1024 * (v1)] = v10
  }
  v12_array := [1024]float32{}
  for v12_i0 := 0; v12_i0 < 1; v12_i0 += 1 {
    for v12_i1 := 0; v12_i1 < 1024; v12_i1 += 1 {
      v12_array[v12_i1 + 1024 * (v12_i0)] = v6[0 + v12_i1 * 1 + 1024 * (0 + v12_i0 * 1)];
    }
  }
  v12 := v12_array[:]
  pt := ckks.NewPlaintext(param, param.MaxLevel())
  v12_pt_packed := make([]float64, len(v12))
  for i := range v12_pt_packed {
    v12_pt_packed[i] = float64(v12[i])
  }
  pt.Scale = param.NewScale(math.Pow(2, 50))
  encoder.Encode(v12_pt_packed, pt)
  ct, err1166 := encryptor.EncryptNew(pt)
  if err1166 != nil {
    panic(err1166)
  }
  v13 := []*rlwe.Ciphertext{ct}
  return v13
}

func configure_override() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor) {
  param, err1168 := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
    LogN: 15,
    LogQ: []int{55, 55, 55, 55, 55, 55, 55, 55, 55},
    LogP: []int{60, 60, 60},
    LogDefaultScale: 50,
  })
  if err1168 != nil {
    panic(err1168)
  }
  encoder := ckks.NewEncoder(param)
  kgen := rlwe.NewKeyGenerator(param)
  sk, pk := kgen.GenKeyPairNew()
  encryptor := rlwe.NewEncryptor(param, pk)
  decryptor := rlwe.NewDecryptor(param, sk)
  rk := kgen.GenRelinearizationKeyNew(sk)
  gk := kgen.GenGaloisKeyNew(5, sk)
  gk1 := kgen.GenGaloisKeyNew(25, sk)
  gk2 := kgen.GenGaloisKeyNew(125, sk)
  gk3 := kgen.GenGaloisKeyNew(625, sk)
  gk4 := kgen.GenGaloisKeyNew(3125, sk)
  gk5 := kgen.GenGaloisKeyNew(15625, sk)
  gk6 := kgen.GenGaloisKeyNew(12589, sk)
  gk7 := kgen.GenGaloisKeyNew(62945, sk)
  gk8 := kgen.GenGaloisKeyNew(52581, sk)
  gk9 := kgen.GenGaloisKeyNew(761, sk)
  gk10 := kgen.GenGaloisKeyNew(3805, sk)
  gk11 := kgen.GenGaloisKeyNew(19025, sk)
  gk12 := kgen.GenGaloisKeyNew(29589, sk)
  gk13 := kgen.GenGaloisKeyNew(16873, sk)
  gk14 := kgen.GenGaloisKeyNew(18829, sk)
  gk15 := kgen.GenGaloisKeyNew(28609, sk)
  gk16 := kgen.GenGaloisKeyNew(11973, sk)
  gk17 := kgen.GenGaloisKeyNew(59865, sk)
  gk18 := kgen.GenGaloisKeyNew(37181, sk)
  gk19 := kgen.GenGaloisKeyNew(54833, sk)
  gk20 := kgen.GenGaloisKeyNew(12021, sk)
  gk21 := kgen.GenGaloisKeyNew(60105, sk)
  gk22 := kgen.GenGaloisKeyNew(38381, sk)
  gk23 := kgen.GenGaloisKeyNew(61313, sk)
  gk24 := kgen.GenGaloisKeyNew(48489, sk)
  gk25 := kgen.GenGaloisKeyNew(7937, sk)
  gk26 := kgen.GenGaloisKeyNew(30517, sk)
  gk27 := kgen.GenGaloisKeyNew(13585, sk)
  gk28 := kgen.GenGaloisKeyNew(1469, sk)
  gk29 := kgen.GenGaloisKeyNew(15873, sk)
  gk30 := kgen.GenGaloisKeyNew(20729, sk)
  gk31 := kgen.GenGaloisKeyNew(58245, sk)
  gk32 := kgen.GenGaloisKeyNew(2849, sk)
  gk33 := kgen.GenGaloisKeyNew(33421, sk)
  gk34 := kgen.GenGaloisKeyNew(60809, sk)
  gk35 := kgen.GenGaloisKeyNew(42197, sk)
  gk36 := kgen.GenGaloisKeyNew(31745, sk)
  gk37 := kgen.GenGaloisKeyNew(37425, sk)
  gk38 := kgen.GenGaloisKeyNew(56413, sk)
  gk39 := kgen.GenGaloisKeyNew(8985, sk)
  gk40 := kgen.GenGaloisKeyNew(2853, sk)
  gk41 := kgen.GenGaloisKeyNew(55873, sk)
  gk42 := kgen.GenGaloisKeyNew(58157, sk)
  gk43 := kgen.GenGaloisKeyNew(33193, sk)
  gk44 := kgen.GenGaloisKeyNew(26229, sk)
  gk45 := kgen.GenGaloisKeyNew(62289, sk)
  gk46 := kgen.GenGaloisKeyNew(26365, sk)
  gk47 := kgen.GenGaloisKeyNew(39225, sk)
  gk48 := kgen.GenGaloisKeyNew(63489, sk)
  ekset := rlwe.NewMemEvaluationKeySet(rk, gk, gk1, gk2, gk3, gk4, gk5, gk6, gk7, gk8, gk9, gk10, gk11, gk12, gk13, gk14, gk15, gk16, gk17, gk18, gk19, gk20, gk21, gk22, gk23, gk24, gk25, gk26, gk27, gk28, gk29, gk30, gk31, gk32, gk33, gk34, gk35, gk36, gk37, gk38, gk39, gk40, gk41, gk42, gk43, gk44, gk45, gk46, gk47, gk48)
  evaluator := ckks.NewEvaluator(param, ekset)
  return evaluator, param, encoder, encryptor, decryptor
}

func TestMNIST(t *testing.T) {
	weights, err := loadWeights(modelPath)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Dump the weights for debugging/comparison with openfhe
	// fmt.Printf("Weights:\n")
	// for i := 0; i < len(weights); i++ {
	// 	for j := 0; j < len(weights[i]); j++ {
	// 		fmt.Printf("%d, %d, %.6f\n", i, j, weights[i][j])
	// 	}
	// }
	// t.Errorf("intended failure")
	// return;

	images, err := loadMNISTImages(imagesPath)
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}

	// Use the autogenerated configuration which has the correct parameters
	// determined by the HEIR compiler
	evaluator, params, encoder, encryptor, decryptor := configure_override()

	// Test with 3 samples since the computation is slow with the
	// large security parameters (LogN=15)
	total := 3

	for i := 0; i < total; i++ {
		input := images[i]

		// Convert float64 input to float32 for encryption helper
		inputFloat32 := make([]float32, len(input))
		for j := 0; j < len(input); j++ {
			inputFloat32[j] = float32(input[j])
		}

		// Use generated encryption helper
		ctInput := encrypt_override(evaluator, params, encoder, encryptor, inputFloat32)

		startTime := time.Now()
		// Calling the generated mnist function
		// Signature: func mnist(evaluator *ckks.Evaluator, params ckks.Parameters, encoder *ckks.Encoder,
		//                     v0 []float32, v1 []float32, v2 []float32, v3 []float32, v4 []*rlwe.Ciphertext) []*rlwe.Ciphertext
		resCt := mnist(evaluator, params, encoder, weights[0], weights[1], weights[2], weights[3], ctInput)
		duration := time.Since(startTime)
		t.Logf("Sample %d took %v", i, duration)

		// Use generated decryption helper
		resValues := mnist__decrypt__result0(evaluator, params, encoder, decryptor, resCt)

		fmt.Printf("output:\n")
		for j := 0; j < 512; j++ {
			fmt.Printf("%d, %.6f\n", j, resValues[j])
		}
	}

	t.Errorf("xFail to dump logs")
}
