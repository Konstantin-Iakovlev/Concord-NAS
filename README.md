## Architecture search

In this implementation FLAX automatically detects GPU device, so we do
not have to specity it. This phase takes less than 1.5 hours on Colab.
Note that torch implementation requires 6 hours.
```bash
python search.py
```


## Fine-tuning
```bash
python retrain.py --arch_path search_dir/final_architecture.json
```