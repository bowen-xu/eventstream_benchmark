from eventstream import generate_patterns, EventStream

# 1) 生成 n=8 个长度 L=4 的模式（类型取自 0..5）
pats = generate_patterns(
    n_patterns=8,
    pattern_length=10,
    n_types=6,
    gap_dist="uniform",  # 模板间隔的分布
    gap_base=5,          # 无效（仅用于非 uniform）
    gap_low=3, gap_high=9,
    seed=137,
)

# 2) 基于模式集合生成事件流：
#    - 总事件数 40_000
#    - 随机事件比例 0.35（其余来自模式实例）
stream = EventStream(
    patterns=pats,
    n_types=6,
    total_events=40_000,
    random_ratio=0.7,
    seed=2025,
    # 模式间隔抖动：开 & 均匀 ±2
    pattern_jitter="uniform",
    pattern_jitter_amount=2,
    # 随机事件间隔：Poisson(base=6)，再做均匀 ±1 抖动
    rand_interval_dist="poisson",
    rand_interval_base=6,
    rand_interval_low=None, rand_interval_high=None,
    rand_jitter="uniform",
    rand_jitter_amount=1,
    # 密度：正弦随时间变化（把间隔除以 r(t)）
    density_mode="sin",
    density_period=8000,
    density_amp=0.6,
    density_base_rate=1.0,
    # 类型漂移（仅影响“随机事件”的类型分配）
    drift_mode="mixed",
    # 可选缓存
    cache_dir="./cache",
    regenerate=False,
)

# 导出、预览
stream.to_csv("event_stream.csv")
arr = stream.to_numpy()
print(arr[:100])
# 流式使用
# for t, e in stream.stream():
#     model.predict(e); model.update(e)
