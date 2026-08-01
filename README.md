# ActiveDGSM

Active learning for Gaussian process regression, driven by gradient-based acquisition
strategies aimed at global sensitivity analysis (Sobol' indices, DGSM). Given a black-box
function, the loop iteratively picks the next design point by maximizing an acquisition
function computed from the GP's exact posterior gradient, then refits and repeats.

No gradient observation are needed.

Method and benchmarks are described in the paper cited below.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

```bash
python run_loop.py --function Ishigami2 --method GlobalGradVarRed --n-init 20 --num-al-iter 280
```

`--function` selects the test problem (see `utils/util.py:available_problems()`):
`ishigami`, `ishigami2`, `lim`, `anisotropicvalley`, `valley`, `morris`, `hartmann4`, `hartmann6`.
`--method` selects the acquisition strategy:

| method | math | description |
| --- | --- | --- |
| `Sobol` | — | Space-filling baseline: candidates drawn from a Sobol' sequence, no acquisition function. |
| `GradMaxVar` | $\alpha_{\mathrm{GMV}}$ | Maximizes the variance of the posterior gradient norm at the candidate point. |
| `GradVarRed` | $\alpha_{\mathrm{GVR}}$ | One-step-lookahead reduction of that gradient variance, evaluated at the candidate point only. |
| `GlobalGradVarRed` | $\alpha_{\mathrm{GlobGVR}}$ | Global `GradVarRed` over a grid $X_s$ that achieves the best perfomances. |
| `GlobalGradVarRedKmeans` | $\alpha_{\mathrm{GlobGVR}}$ | `GlobalGradVarRed` with $X_s$ chunked via k-means for memory-bounded evaluation on larger point sets. |

For `GlobalGradVarRedKmeans`, `--kmeans-config` selects how the reference set is chunked:
`default` uses regular k-means (uneven cluster sizes), `balanced` uses constrained k-means
(equal-size clusters). Ignored by every other method.

For `GlobalGradVarRed` and `GlobalGradVarRedKmeans`, `--global-points-factor` (default `50`)
sets the number of global points $X_s\in(\mathcal{X})^N$ per input dimension: `points = min(dim *
factor, 2000 if dim <= 15 else 1000)`. Ignored by the local and Sobol' methods.

`--num-fantasies` (default `8`) sets the number of fantasy samples used by the
look-ahead reduction in `GradVarRed`, `GlobalGradVarRed`, and
`GlobalGradVarRedKmeans`. Ignored by `GradMaxVar` and `Sobol`.

Run `python run_loop.py --help` for the full list of options (retraining strategy, GP
normalization, metric logging frequency, state-dict checkpointing, ...).

## Layout

```
run_loop.py          main active-learning loop
run.py                minimal positional-args command line wrapper around run_loop
utils/acquisition.py  acquisition functions (the four methods above)
utils/posterior.py    exact GP gradient posterior helpers  (for ARD-Matern 5/2 kernel)
utils/model.py        GP wrapper (normalization, fit/predict/fantasize, state dict I/O, gradient posterior)
utils/util.py         test-problem registry, Q2/DGSM/Sobol' metric evaluation
utils/functions.py    synthetic test functions (Ishigami, Lim, AnisotropicValley, Morris, Hartmann, Gsobol)
```

## Citation

```bibtex
@misc{lambert2026gvr,
      title={Gradient-based Active Learning with Gaussian Processes for Global Sensitivity Analysis},
      author={Guerlain Lambert and C\'eline Helbert and Claire Lauvernet},
      year={2026},
}
```

Currently under review.

## Acknowledgments

Parts of `utils/functions.py`, `run_loop.py`, and `run.py` are adapted from
[belakaria/AL-GSA-DGSMs](https://github.com/belakaria/AL-GSA-DGSMs).
