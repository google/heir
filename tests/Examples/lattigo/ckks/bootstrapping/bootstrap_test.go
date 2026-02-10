package bootstrap

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func encryptArg(_ *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor, v0 []float32) *rlwe.Ciphertext {
	v7 := make([]float64, 2048)
	for v8 := 0; v8 < 2048; v8++ {
		v7[v8] = float64(v0[v8%8])
	}
	pt := ckks.NewPlaintext(param, param.MaxLevel())
	pt.Scale = param.NewScale(math.Pow(2, 45))
	encoder.Encode(v7, pt)
	ct, err26 := encryptor.EncryptNew(pt)
	if err26 != nil {
		panic(err26)
	}
	return ct
}

func decryptResult(_ *ckks.Evaluator, _ ckks.Parameters, encoder *ckks.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext) []float32 {
	pt := decryptor.DecryptNew(ct)
	float64Decode := make([]float64, 2048)
	encoder.Decode(pt, float64Decode)
	float32Result := make([]float32, 2048)
	for i := range 2048 {
		float32Result[i] = float32(float64Decode[i])
	}
	return float32Result[0:7]
}

func TestBootstrap(t *testing.T) {
	btEvaluator, evaluator, params, ecd, enc, dec := bootstrap__configure()

	fmt.Printf("Residual parameters: logN=%d, logSlots=%d, H=%d, sigma=%f, logQ=%f, logQP=%f, levels=%d, scale=2^%d\n",
		btEvaluator.Parameters.ResidualParameters.LogN(),
		btEvaluator.Parameters.ResidualParameters.LogMaxSlots(),
		btEvaluator.Parameters.ResidualParameters.XsHammingWeight(),
		btEvaluator.Parameters.ResidualParameters.Xe(), params.LogQ(), params.LogQP(),
		btEvaluator.Parameters.ResidualParameters.MaxLevel(),
		btEvaluator.Parameters.ResidualParameters.LogDefaultScale())

	fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%f, logQ=%f, logQP=%f, levels=%d, scale=2^%d, maxLevel=%d\n",
		btEvaluator.Parameters.BootstrappingParameters.LogN(),
		btEvaluator.Parameters.BootstrappingParameters.LogMaxSlots(),
		btEvaluator.Parameters.BootstrappingParameters.XsHammingWeight(),
		btEvaluator.Parameters.EphemeralSecretWeight,
		btEvaluator.Parameters.BootstrappingParameters.Xe(),
		btEvaluator.Parameters.BootstrappingParameters.LogQ(),
		btEvaluator.Parameters.BootstrappingParameters.LogQP(),
		btEvaluator.Parameters.BootstrappingParameters.QCount(),
		btEvaluator.Parameters.BootstrappingParameters.LogDefaultScale(),
		btEvaluator.OutputLevel())

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	expected := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

	ct0 := encryptArg(evaluator, params, ecd, enc, arg0)

	startTime := time.Now()
	resultCt := bootstrap(btEvaluator, evaluator, params, ecd, ct0)
	duration := time.Since(startTime)
	fmt.Printf("bootstrap call took: %v\n", duration)

	result := decryptResult(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.0001)
	for i := 0; i < len(result); i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error %.2f != %.2f", result[i], expected[i])
		}
	}
}
