# Results

Training and evaluation artifacts are generated here:

- `training_log.csv`
- `training_history.json`
- `training_curves.png`
- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`

Do not hand-edit metrics. Regenerate them with:

```bash
python -m src.evaluate --model-path models/best_model.keras
```

