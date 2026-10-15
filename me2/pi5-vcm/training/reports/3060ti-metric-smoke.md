# 3060 Ti training metric smoke test

**Purpose:** verify the pinned data-to-model training path and collect
initial optimization metrics. This tiny stratified run is not a full
training run and its validation Macro-F1 is not a model-quality estimate.

## Run configuration

- Device: NVIDIA GeForce RTX 3060 Ti (cuda:0)
- Precision: fp16
- Seed: 42; deterministic algorithms enabled
- Training subset: 80 examples (4 per intent)
- Validation subset: 80 examples (4 per intent)
- Batch size: 16
- Training budget: one Stage A epoch and one Stage B epoch
- Elapsed time (including mel-stat fitting): 4.16 seconds
- Checkpoint: `training\reports\3060ti-smoke-artifacts\me2_model.pt`
- Full run metadata: `training\reports\3060ti-smoke-artifacts\training_run.json`

## Metrics

| Stage | Epoch | Train loss | Train intent loss | Train slot loss | Train Macro-F1 | Validation loss | Validation intent loss | Validation slot loss | Validation Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 10.68050 | 4.43481 | 6.24570 | 0.00476 | 6.80322 | 4.14564 | 2.65758 | 0.00476 |
| B | 1 | 10.69542 | 3.90987 | 6.78555 | 0.00900 | 6.52134 | 3.25126 | 3.27007 | 0.02517 |

Metrics are from fixed, small, class-balanced subsets. Use the full
training and validation manifests for a real fit and meaningful
validation estimates; this smoke run intentionally does not do that.
