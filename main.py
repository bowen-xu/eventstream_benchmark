from eventstream import generate_patterns, EventStream

# 1) Generate n=8 patterns of length L=10 (types from 0..5)
pats = generate_patterns(
    n_patterns=8,
    pattern_length=10,
    n_types=6,
    gap_dist="uniform",
    gap_low=3, gap_high=9,
    seed=137,
)

# 2) Generate event stream based on the patterns:
#    - total events 40,000
#    - random event ratio 0.7 (rest from pattern instances)
stream = EventStream(
    patterns=pats,
    n_types=6,
    total_events=40_000,
    random_ratio=0.7,
    seed=2025,
    pattern_jitter="uniform",
    pattern_jitter_amount=2,
    # Random interval: poisson(base=6) with uniform ±1 jitter
    rand_interval_dist="poisson",
    rand_interval_base=6,
    rand_interval_low=None, rand_interval_high=None,
    rand_jitter="uniform",
    rand_jitter_amount=1,
    # Density: sinusoidal over time (divide intervals by r(t))
    density_mode="sin",
    density_period=8000,
    density_amp=0.6,
    density_base_rate=1.0,
    # optional cache directory to speed up repeated generation with same config
    cache_dir="./cache",
    regenerate=False,
)

# 3) Export to CSV and NumPy array
stream.to_csv("event_stream.csv")
arr = stream.to_numpy()
print(arr[:100])
# 4) (Optional) Run model on the stream
# for t, e in stream.stream():
#     model.predict(e); model.update(e)
