# HW4 CNN Two-file Version

This version combines all previous helper modules into only:

```text
train.py
test.py
```

The commands, inputs, and outputs remain the same.

## Expected data layout

```text
project_folder/
├─ train.py
├─ test.py
├─ data/
│  ├─ hao/
│  ├─ jin/
│  └─ gua/
└─ data_face_only/
   ├─ hao/
   ├─ jin/
   └─ gua/
```

Both `data/` and `data_face_only/` will be used if both exist.

## Run all experiments

```bash
python -u train.py --raw_dir . --run_all
```

## Final training on all available data

```bash
python -u train.py --raw_dir . --config deeper160 --final_train_all --epochs 100
```

This saves:

```text
models/best_model.pth
```

## Test hidden images

```bash
python -u test.py --data_dir hidden_test --weights models/best_model.pth --tta
```


## Outputs

Training creates:

```text
outputs/split_manifest.csv
outputs/history_baseline64.csv
outputs/history_compact128.csv
outputs/history_deeper160.csv
outputs/experiment_summary.json
models/baseline64_best.pth
models/compact128_best.pth
models/deeper160_best.pth
models/best_model.pth       # when --final_train_all is used
```

Testing creates:

```text
outputs/test_predictions.csv
outputs/test_prediction_examples.png
```
