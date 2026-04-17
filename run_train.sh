for city in Beijing Porto San_Francisco; do
    python train.py --config src/config/config_${city}.yaml
done