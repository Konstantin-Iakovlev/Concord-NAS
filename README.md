## Architecture search

```bash
python search.py --device=cuda:0 --log-interval=1
```


## Fine-tuning

Given a directory from the first phase `search_dir`.

```bash
python retrain.py --device=cuda:0 --arch_path search_dir/final_architecture.json \
    --log-interval=1
```