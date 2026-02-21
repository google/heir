package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"google3/third_party/golang/github_com/tuneinsight/lattigo/v/v6/circuits/ckks/bootstrapping/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"google3/third_party/golang/github_com/tuneinsight/lattigo/v/v6/utils/sampling/sampling"
	"google3/third_party/golang/github_com/tuneinsight/lattigo/v/v6/utils/utils"
)

type BenchmarkResult struct {
	LogN                  int
	EphemeralSecretWeight int
	LogQP                 float64
	SetupLatencySeconds   string
	EvalLatencySeconds    string
}

func runBenchmark(logN int, h int) BenchmarkResult {
	result := BenchmarkResult{
		LogN:                  logN,
		EphemeralSecretWeight: h,
		SetupLatencySeconds:   "N/A",
		EvalLatencySeconds:    "N/A",
	}

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		fmt.Printf("Error creating parameters for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}

	result.LogQP = params.LogQP()

	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN:                  utils.Pointy(logN),
		LogP:                  []int{61, 61, 61, 61},
		Xs:                    params.Xs(),
		EphemeralSecretWeight: utils.Pointy(h),
	}

	startTime := time.Now()
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		fmt.Printf("Error creating bootstrapping parameters for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	evk, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		fmt.Printf("Error generating evaluation keys for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}

	eval, err := bootstrapping.NewEvaluator(btpParams, evk)
	if err != nil {
		fmt.Printf("Error creating evaluator for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}
	result.SetupLatencySeconds = fmt.Sprintf("%f", time.Since(startTime).Seconds())

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, sk)

	values := make([]complex128, params.MaxSlots())
	for i := range values {
		values[i] = sampling.RandComplex128(-1, 1)
	}

	plaintext := ckks.NewPlaintext(params, 0)
	if err := encoder.Encode(values, plaintext); err != nil {
		fmt.Printf("Error encoding for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}

	ciphertext, err := encryptor.EncryptNew(plaintext)
	if err != nil {
		fmt.Printf("Error encrypting for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}

	startTime = time.Now()
	_, err = eval.Bootstrap(ciphertext)
	if err != nil {
		fmt.Printf("Error bootstrapping for LogN=%d H=%d: %v\n", logN, h, err)
		return result
	}
	result.EvalLatencySeconds = fmt.Sprintf("%f", time.Since(startTime).Seconds())

	return result
}

func main() {
	outPath := flag.String("out", "lattigo_results.toml", "output path for results")
	flag.Parse()

	// Open file once and write results as they come
	f, err := os.Create(*outPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	for _, logN := range []int{15, 16} {
		for _, h := range []int{0, 32, 64} {
			fmt.Printf("Running Lattigo benchmark for LogN=%d, EphemeralSecretWeight=%d...\n", logN, h)
			res := runBenchmark(logN, h)

			fmt.Fprintf(f, "[[results]]\n")
			fmt.Fprintf(f, "log_n = %d\n", res.LogN)
			fmt.Fprintf(f, "ephemeral_secret_weight = %d\n", res.EphemeralSecretWeight)
			fmt.Fprintf(f, "log_qp = %f\n", res.LogQP)
			if res.SetupLatencySeconds == "N/A" {
				fmt.Fprintf(f, "setup_latency_seconds = \"N/A\"\n")
			} else {
				fmt.Fprintf(f, "setup_latency_seconds = %s\n", res.SetupLatencySeconds)
			}
			if res.EvalLatencySeconds == "N/A" {
				fmt.Fprintf(f, "eval_latency_seconds = \"N/A\"\n")
			} else {
				fmt.Fprintf(f, "eval_latency_seconds = %s\n", res.EvalLatencySeconds)
			}
			f.Sync() // Ensure results are written even if long-running
		}
	}
}
