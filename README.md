# Run basic experiment
python3 search.py --config mnist_basic.cfg

# train the best architecture
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
