import params
import evaluator
from perf_counter import PerfCounter
from experiment import run_mutiple, print_table, Target


def polynomial_eval(ct, ct1, scheme_params: params.SchemeParams):
  stats = PerfCounter()
  stats += evaluator.multiply(ct, scheme_params.arch_param)
  ct2 = ct
  stats += evaluator.key_switch(
      ct2, scheme_params.fresh_ctxt, scheme_params.arch_param
  )
  ct3 = ct2
  stats += evaluator.multiply(ct, scheme_params.arch_param)
  ct4 = ct
  stats += evaluator.key_switch(
      ct4, scheme_params.fresh_ctxt, scheme_params.arch_param
  )
  ct5 = ct4
  stats += evaluator.add(ct5, scheme_params.arch_param)
  ct6 = ct5
  stats += evaluator.add(ct3, scheme_params.arch_param)
  ct7 = ct3
  stats += evaluator.add(ct7, scheme_params.arch_param)
  ct8 = ct7
  stats += evaluator.rotate(ct8, scheme_params.arch_param)
  ct9 = ct8
  stats += evaluator.add(ct8, scheme_params.arch_param)
  ct10 = ct8
  return stats


if __name__ == "__main__":
  targets = []

  # Note: Ideally, the parameters here would depend on the specific workload,
  # but currently, the emitter just uses the same parameters for all workloads.
  for scheme_params in [
      params.Alg_benchmark_baseline,
      params.Alg_benchmark_mod_down_merge,
      params.Alg_benchmark_mod_down_hoist,
      params.BEST_PARAMS,
  ]:
    print(scheme_params)

    targets.append(
        Target(
            "generated.polynomial_eval",
            1,
            [scheme_params.fresh_ctxt, scheme_params.fresh_ctxt, scheme_params],
        )
    )

  # Run and print
  headers, data = run_mutiple(targets)
  print_table(headers, data)
  print()
