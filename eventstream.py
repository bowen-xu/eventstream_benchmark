# Purpose:
#   1) 生成给定长度的模式集合（模式 = [type_0, +gap_0, type_1, +gap_1, ..., type_{L-1}]）
#   2) 基于模式集合，生成“整数时间戳 + 事件类型”的事件流
#      - 可配置随机事件比例
#      - 可配置“模式间隔抖动”（开/关、幅度、分布）
#      - 可配置“随机事件的间隔生成方式”（基准间隔、抖动、分布）
#      - 可选：时间密度（rate）随时间变化，以及类型分布漂移（用于随机事件的类型分配）
# 设计目标：高效（numpy 数组存储），流式可用，复现实验方便（seed）

from __future__ import annotations
import numpy as np
from pathlib import Path
import pickle
import hashlib
from typing import Literal, Tuple, List, Optional


JitterDist = Literal["none", "uniform", "gaussian", "laplace"]
IntervalDist = Literal["fixed", "poisson", "geometric", "uniform"]
DensityMode = Literal["none", "sin"]
DriftMode = Literal["none", "gradual", "sudden", "mixed"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _apply_jitter_int(
    base: int,
    dist: JitterDist,
    amount: int,
    rng: np.random.Generator,
    min_val: int = 1,
) -> int:
    """
    对“模板间隔”或“随机间隔”施加抖动，保证输出为 >= min_val 的整数。
    amount 的语义：
      - uniform：[-amount, +amount] 的离散均匀抖动
      - gaussian：均值0、标准差=amount 的正态随机，取整
      - laplace：均值0、b=amount 的拉普拉斯随机，取整
      - none：不抖动
    """
    if dist == "none" or amount <= 0:
        return max(min_val, int(base))

    if dist == "uniform":
        delta = rng.integers(-amount, amount + 1)
    elif dist == "gaussian":
        delta = int(np.round(rng.normal(0, amount)))
    elif dist == "laplace":
        # numpy 的 laplace(scale=b)，我们用 amount 作为 scale
        delta = int(np.round(rng.laplace(0.0, amount)))
    else:
        raise ValueError(f"Unknown jitter dist: {dist}")

    return max(min_val, int(base + delta))


def _sample_interval_int(
    base: int,
    dist: IntervalDist,
    rng: np.random.Generator,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> int:
    """
    采样“随机事件”的基础间隔（尚未施加抖动/密度缩放）。
      - fixed:  恒定 base
      - poisson: Poisson(λ=base) 的样本（保证 >=1）
      - geometric: 几何分布（p = 1/base），期望约为 base（保证 >=1）
      - uniform: [low, high] 的离散均匀（需要给 low, high）
    """
    if dist == "fixed":
        return max(1, int(base))
    elif dist == "poisson":
        # 避免出现0
        return max(1, int(rng.poisson(max(1e-9, float(base)))))
    elif dist == "geometric":
        p = 1.0 / max(1.0, float(base))
        # numpy 几何分布从1开始
        return int(rng.geometric(p))
    elif dist == "uniform":
        if low is None or high is None or low < 1 or high < low:
            raise ValueError("uniform interval requires valid [low, high] with low>=1")
        return int(rng.integers(low, high + 1))
    else:
        raise ValueError(f"Unknown interval dist: {dist}")


def _density_scale(
    t: int, mode: DensityMode, base_rate: float, period: int, amp: float
) -> float:
    """
    密度控制：返回一个“速率缩放”因子 r(t)。
    你可以把它理解为“当前时刻的事件速率 / 基准速率”，我们用它来
    缩短或拉长间隔（interval' = max(1, round(interval / r(t))))。
    - mode='none'：r(t)=1
    - mode='sin'：r(t)=base_rate * (1 + amp * sin(2π t / period))
      注意：amp ∈ [0, 1) 比较合理；base_rate 通常取1.0
    """
    if mode == "none":
        return 1.0
    elif mode == "sin":
        # 保证正值
        r = base_rate * (1.0 + amp * np.sin(2.0 * np.pi * (t / max(1, period))))
        return max(1e-6, float(r))
    else:
        raise ValueError(f"Unknown density mode: {mode}")


def _type_probs_with_drift(n_types: int, t: int, mode: DriftMode) -> np.ndarray:
    """
    给“随机事件”分配类型时使用的时间相关分布（可选概念漂移）。
      - none: 均匀分布
      - gradual: 概率在 [-shift, +shift] 范围内正弦缓慢变化
      - sudden: 周期性翻转
      - mixed: 结合 gradual + sudden
    """
    p = np.ones(n_types) / n_types
    if mode == "none":
        return p

    if mode == "gradual":
        shift = 0.3 * np.sin(t / 2000.0)
        p = p + shift * np.linspace(-1, 1, n_types)
    elif mode == "sudden":
        if (t // 10000) % 2 == 1:
            p = p[::-1]
    elif mode == "mixed":
        shift = 0.3 * np.sin(t / 3000.0)
        p = p + shift * np.linspace(-1, 1, n_types)
        if (t // 15000) % 2 == 1:
            p = p[::-1]
    else:
        raise ValueError(f"Unknown drift mode: {mode}")

    p = np.clip(p, 0.001, None)
    return p / p.sum()


class PatternSet:
    """
    模式集合：包含 n 个长度为 L 的模式。
    每个模式由：
      - types: 长度 L 的整数数组（事件类型）
      - gaps:  长度 L-1 的整数数组（相邻事件之间的“模板间隔”，单位：整数时间）
    """

    def __init__(self, types: np.ndarray, gaps: np.ndarray):
        self.types = types  # shape: [n_patterns, L]
        self.gaps = gaps  # shape: [n_patterns, L-1]

    @property
    def n_patterns(self) -> int:
        return self.types.shape[0]

    @property
    def length(self) -> int:
        return self.types.shape[1]


def generate_patterns(
    n_patterns: int,
    pattern_length: int,
    n_types: int,
    gap_dist: IntervalDist = "uniform",
    gap_base: int = 5,
    gap_low: Optional[int] = 2,
    gap_high: Optional[int] = 9,
    seed: int = 137,
) -> PatternSet:
    """
    生成模式集合。
      - 事件类型从 [0, n_types-1] 均匀采样（允许重复）
      - 模板间隔（gaps）按指定分布采样（全是整数且 >=1）

    参数
    ----
    n_patterns: 生成多少个模式
    pattern_length: 模式长度 L（事件个数）
    n_types: 事件类型总数
    gap_dist: 模板间隔的分布类型（'fixed'/'poisson'/'geometric'/'uniform'）
    gap_base: 若使用 fixed/poisson/geometric 时的基准参数
    gap_low/gap_high: 若使用 uniform 时的区间
    """
    if pattern_length < 2:
        raise ValueError("pattern_length must be >= 2")
    rng = _rng(seed)

    # 事件类型矩阵 [n_patterns, L]
    types = rng.integers(0, n_types, size=(n_patterns, pattern_length), dtype=np.int32)

    # 间隔矩阵 [n_patterns, L-1]
    gaps = np.empty((n_patterns, pattern_length - 1), dtype=np.int32)
    for i in range(n_patterns):
        for j in range(pattern_length - 1):
            gaps[i, j] = _sample_interval_int(
                base=gap_base,
                dist=gap_dist,
                rng=rng,
                low=gap_low,
                high=gap_high,
            )

    return PatternSet(types=types, gaps=gaps)


class EventStream:
    """
    从模式集合生成“整数时间戳 + 事件类型”的事件流。
    支持参数：
      - 随机事件比例 random_ratio
      - 模式间隔抖动（开/关、幅度、分布）
      - 随机事件间隔生成方式（分布 + 抖动）
      - 密度模式（控制速率随时间变化）
      - 类型漂移（仅影响随机事件的类型分配）
    """

    def __init__(
        self,
        patterns: PatternSet,
        n_types: int,
        total_events: int,
        random_ratio: float,
        seed: int = 123,
        # 模式间隔抖动
        pattern_jitter: JitterDist = "uniform",
        pattern_jitter_amount: int = 0,
        # 随机事件间隔生成
        rand_interval_dist: IntervalDist = "poisson",
        rand_interval_base: int = 5,
        rand_interval_low: Optional[int] = 2,
        rand_interval_high: Optional[int] = 9,
        rand_jitter: JitterDist = "uniform",
        rand_jitter_amount: int = 1,
        # 密度控制（对所有间隔生效）
        density_mode: DensityMode = "none",
        density_period: int = 10_000,
        density_amp: float = 0.8,
        density_base_rate: float = 1.0,
        # 随机事件的类型漂移
        drift_mode: DriftMode = "none",
        # 缓存（可选）
        cache_dir: Optional[str] = None,
        regenerate: bool = False,
    ):
        assert 0.0 <= random_ratio <= 1.0
        self.patterns = patterns
        self.n_types = n_types
        self.total_events = int(total_events)
        self.random_ratio = float(random_ratio)
        self.rng = _rng(seed)

        self.pattern_jitter = pattern_jitter
        self.pattern_jitter_amount = int(pattern_jitter_amount)

        self.rand_interval_dist = rand_interval_dist
        self.rand_interval_base = int(rand_interval_base)
        self.rand_interval_low = (
            None if rand_interval_low is None else int(rand_interval_low)
        )
        self.rand_interval_high = (
            None if rand_interval_high is None else int(rand_interval_high)
        )
        self.rand_jitter = rand_jitter
        self.rand_jitter_amount = int(rand_jitter_amount)

        self.density_mode = density_mode
        self.density_period = int(density_period)
        self.density_amp = float(density_amp)
        self.density_base_rate = float(density_base_rate)

        self.drift_mode = drift_mode

        # 缓存键
        self.cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = (
                f"{patterns.n_patterns}-{patterns.length}-{n_types}-"
                f"{total_events}-{random_ratio}-{seed}-"
                f"{pattern_jitter}-{pattern_jitter_amount}-"
                f"{rand_interval_dist}-{rand_interval_base}-"
                f"{rand_interval_low}-{rand_interval_high}-"
                f"{rand_jitter}-{rand_jitter_amount}-"
                f"{density_mode}-{density_period}-{density_amp}-{density_base_rate}-"
                f"{drift_mode}"
            )
            h = hashlib.md5(key.encode()).hexdigest()[:10]
            self.cache_path = cache_dir / f"stream_{h}.pkl"

        # 生成或加载
        if self.cache_path and self.cache_path.exists() and not regenerate:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.timestamps = data["timestamps"]
            self.types = data["types"]
            self.is_pattern = data["is_pattern"]
        else:
            self.timestamps, self.types, self.is_pattern = self._generate_stream()
            if self.cache_path:
                with open(self.cache_path, "wb") as f:
                    pickle.dump({"timestamps": self.timestamps, "types": self.types, "is_pattern": self.is_pattern}, f)

    # ------------------------ 核心生成 ------------------------

    def _generate_stream(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        改进版事件流生成逻辑：
        - 模式实例和随机事件按比例穿插，而不是整块交替。
        - 模式内部事件保持完整（连续写入，不被随机事件打断）。
        - 随机事件在模式之间插入，以满足 random_ratio。
        """
        rng = self.rng
        L = self.patterns.length
        N = self.total_events

        # 目标比例与目标事件数
        target_random = int(round(N * self.random_ratio))
        target_pattern = N - target_random

        timestamps = np.empty(N, dtype=np.int64)
        types = np.empty(N, dtype=np.int32)
        is_pattern = np.empty(N, dtype=bool)

        t = 0
        i = 0
        n_random = 0
        n_pattern = 0

        # 辅助函数：插入一个随机事件
        def insert_random_event(t_current: int) -> int:
            nonlocal i, n_random
            # 采样随机间隔
            base_int = _sample_interval_int(
                base=self.rand_interval_base,
                dist=self.rand_interval_dist,
                rng=rng,
                low=self.rand_interval_low,
                high=self.rand_interval_high,
            )
            base_int = _apply_jitter_int(
                base=base_int,
                dist=self.rand_jitter,
                amount=self.rand_jitter_amount,
                rng=rng,
                min_val=1,
            )
            r = _density_scale(
                t=t_current,
                mode=self.density_mode,
                base_rate=self.density_base_rate,
                period=self.density_period,
                amp=self.density_amp,
            )
            gap = max(1, int(round(base_int / r)))
            t_next = t_current + gap

            p = _type_probs_with_drift(self.n_types, t_next, self.drift_mode)
            et = int(rng.choice(self.n_types, p=p))

            timestamps[i] = t_next
            types[i] = et
            is_pattern[i] = False
            i += 1
            n_random += 1
            return t_next

        # 辅助函数：插入一个完整模式实例
        def insert_pattern_instance(t_current: int) -> int:
            nonlocal i, n_pattern
            pid = int(rng.integers(0, self.patterns.n_patterns))
            p_types = self.patterns.types[pid]
            p_gaps = self.patterns.gaps[pid]
            for j in range(L):
                timestamps[i] = t_current
                types[i] = p_types[j]
                is_pattern[i] = True
                i += 1
                n_pattern += 1
                if i >= N:
                    return t_current
                if j < L - 1:
                    gap = int(p_gaps[j])
                    gap = _apply_jitter_int(
                        base=gap,
                        dist=self.pattern_jitter,
                        amount=self.pattern_jitter_amount,
                        rng=rng,
                        min_val=1,
                    )
                    r = _density_scale(
                        t=t_current,
                        mode=self.density_mode,
                        base_rate=self.density_base_rate,
                        period=self.density_period,
                        amp=self.density_amp,
                    )
                    eff = max(1, int(round(gap / r)))
                    t_current += eff
            # 模式结束后略加间隔
            t_current += 1
            return t_current

        # 主循环：逐事件生成
        while i < N:
            # 当前随机事件比例
            ratio_now = n_random / max(1, (n_random + n_pattern))
            # 决策：若当前比例 < 目标比例，则插入随机事件，否则插入模式
            if (ratio_now < self.random_ratio or n_pattern + L > target_pattern) and n_random < target_random:
                t = insert_random_event(t)
            else:
                t = insert_pattern_instance(t)

        # 裁剪（防止越界）
        timestamps = timestamps[:N]
        types = types[:N]
        is_pattern = is_pattern[:N]

        return timestamps, types, is_pattern

    # ------------------------ 导出与迭代 ------------------------

    def to_numpy(self) -> np.ndarray:
        """返回 shape=[N,3] 的 ndarray（timestamp,event_type,is_pattern）"""
        return np.stack([self.timestamps, self.types, self.is_pattern], axis=1)

    def to_csv(self, path: str):
        """保存为 CSV：timestamp,event_type"""
        arr = self.to_numpy()
        np.savetxt(
            path,
            arr,
            delimiter=",",
            header="timestamp,event_type,is_pattern",
            comments="",
            fmt="%d,%d,%d",
        )

    def stream(self):
        """流式迭代器：yield (timestamp:int, event_type:int)"""
        for t, e in zip(self.timestamps, self.types):
            yield int(t), int(e)
