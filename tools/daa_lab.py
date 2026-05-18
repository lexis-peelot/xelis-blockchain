#!/usr/bin/env python3
import argparse
import copy
import csv
import math
import random
import shlex
import sys
import textwrap
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path


SHIFT = 20
SCALE = 1 << SHIFT
MILLIS_PER_SECOND = 1000
DEFAULT_TARGET_MS = 5000
DEFAULT_CURRENT_WINDOW = 50
DEFAULT_WINDOWED_WINDOW = 25
DEFAULT_STABLE_LIMIT = 24
DEFAULT_MIN_MEASUREMENT_COUNT = 2
DEFAULT_GAMMA_EFFECTIVE_COUNT = 80
DEFAULT_GAMMA_FAST_EFFECTIVE_COUNT = 52
DEFAULT_GAMMA_FAST_THRESHOLD = 1.44
DEFAULT_GAMMA_PRIOR_COUNT = 12
DEFAULT_GAMMA_MAX_RATE_INCREASE_PERCENT = 103
DEFAULT_ASERT_HALF_LIFE_BLOCKS = 72
MAX_GAMMA_REPLAY_EVENTS = 256
SCENARIOS = (
    "steady",
    "step-down",
    "step-up",
    "big-drop",
    "rental",
    "adaptive-rental",
    "ramp-up",
    "step-cycle",
)
COMPARE_MODES = ("current", "windowed", "replay-gamma", "asert")
RATIO_SAMPLE_OFFSETS = (1, 10, 25, 50, 100, 200, 500, 1000)
PLOT_FOOTER_WIDTH = 170
STEP_MULTIPLIERS = {"step-down": 0.5, "step-up": 2.0, "big-drop": 0.1}

LEGACY_PROCESS_NOISE = SCALE * SHIFT // MILLIS_PER_SECOND
DEFAULT_PROCESS_NOISE_DIVISOR = 10_000
INITIAL_RELATIVE_COVAR = SCALE // 1000


def current_filter(z, x_prev, p_prev, reduce_drop):
    if reduce_drop and z < x_prev:
        z = x_prev - (x_prev - z) // 4

    z_scaled = z * SCALE
    r = z_scaled * 2
    x_scaled = x_prev * SCALE
    p_pred = ((x_scaled * LEGACY_PROCESS_NOISE) >> SHIFT) + p_prev
    k = (p_pred << SHIFT) // (p_pred + r + 1)

    if z_scaled >= x_scaled:
        x_new = x_scaled + ((k * (z_scaled - x_scaled)) >> SHIFT)
    else:
        x_new = x_scaled - ((k * (x_scaled - z_scaled)) >> SHIFT)

    p_new = ((SCALE - k) * p_pred) >> SHIFT
    return x_new >> SHIFT, p_new


def windowed_filter(z, x_prev, p_prev, measurement_count, process_noise):
    measurement_noise = SCALE // max(1, measurement_count)
    if p_prev > measurement_noise:
        p_prev = INITIAL_RELATIVE_COVAR

    p_pred = p_prev + process_noise
    k = (p_pred << SHIFT) // (p_pred + measurement_noise + 1)

    if z >= x_prev:
        x_new = x_prev + ((k * (z - x_prev)) >> SHIFT)
    else:
        x_new = x_prev - ((k * (x_prev - z)) >> SHIFT)

    p_new = ((SCALE - k) * p_pred) >> SHIFT
    return x_new, p_new


@dataclass(frozen=True)
class OptimizeParams:
    effective_count: int
    fast_effective_count: int
    fast_threshold_percent: int
    prior_count: int
    max_rate_increase_percent: int
    min_measurement_count: int


OPTIMIZE_PARAM_FIELDS = tuple(OptimizeParams.__dataclass_fields__)


@dataclass(frozen=True)
class OptimizeCase:
    name: str
    scenario: str
    blocks: int
    step_at: int
    weight: float = 1.0
    side_rate: float = 0.0
    initial_ratio: float = 1.0
    minimum_ratio: float = 0.0
    base_hashrate: int = 1_000_000
    warmup: int = 0


def clone_args(args, **updates):
    run_args = copy.copy(args)
    for key, value in updates.items():
        setattr(run_args, key, value)
    return run_args


def hashrate_multiplier(scenario, height, step_at, difficulty, base_difficulty):
    if scenario == "steady" or height < step_at:
        return 1.0
    if scenario in STEP_MULTIPLIERS:
        return STEP_MULTIPLIERS[scenario]
    if scenario == "rental":
        return 10.0 if height < step_at + 100 else 1.0
    if scenario == "adaptive-rental":
        return 10.0 if difficulty < base_difficulty * 10 else 1.0
    if scenario == "ramp-up":
        return min(2.0, 1.0 + (height - step_at) / 1000)
    if scenario == "step-cycle":
        cycle = (2.0, 0.5, 1.5, 0.75, 1.25, 1.0)
        return cycle[((height - step_at) // 250) % len(cycle)]
    raise ValueError(f"unknown scenario: {scenario}")


def sample_solve_time(mean_ms, rng, randomize):
    if not randomize:
        return mean_ms

    u = max(rng.random(), 1e-12)
    return -math.log(u) * mean_ms


def sample_poisson(lam, rng):
    if lam <= 0:
        return 0

    limit = math.exp(-lam)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def gamma_estimate(alpha, beta):
    return max(1, round(alpha / beta))


def gamma_replay_update(alpha, beta, measurement_count, observed_hashrate, effective_count, max_rate_increase_percent):
    replay_events = min(max(1, measurement_count), MAX_GAMMA_REPLAY_EVENTS)
    event_exposure = (measurement_count / observed_hashrate) / replay_events
    decay = (effective_count - 1) / effective_count
    previous_estimate = gamma_estimate(alpha, beta)

    for _ in range(replay_events):
        alpha = alpha * decay + 1
        beta = beta * decay + event_exposure

    estimate = gamma_estimate(alpha, beta)
    if max_rate_increase_percent > 0:
        max_rate = previous_estimate * (max_rate_increase_percent / 100) ** replay_events
        if estimate > max_rate:
            estimate = max(1, round(max_rate))
            beta = alpha / estimate

    return alpha, beta, estimate


def measure_observation(args, intervals):
    source = args.measurement_source
    if source == "dag-order":
        span = list(intervals)[-args.stable_limit:]
    else:
        span = list(intervals)[-(args.window - 1):]

    if not span:
        return 1, 1, 1

    main_blocks = len(span)
    observed_blocks = sum(1 + row["side_blocks"] for row in span)
    time_span = max(observed_blocks, round(sum(row["solve_time_ms"] for row in span)))
    observed_work = sum(row["difficulty"] * (1 + row["side_blocks"]) for row in span)
    solve_time = max(1, round(time_span / main_blocks))

    if args.measurement == "work" and args.mode in ("replay-gamma", "windowed"):
        observed_hashrate = max(1, observed_work * MILLIS_PER_SECOND // time_span)
    else:
        observed_hashrate = span[-1]["difficulty"] * MILLIS_PER_SECOND // solve_time

    return observed_hashrate, solve_time, max(1, observed_blocks)


def asert_update(args, intervals):
    span = list(intervals)[-args.stable_limit:]
    if not span:
        return 1

    observed_blocks = sum(1 + row["side_blocks"] for row in span)
    elapsed_ms = sum(row["solve_time_ms"] for row in span)
    anchor_difficulty = span[0]["difficulty"]
    half_life_ms = args.asert_half_life_blocks * args.target_ms
    exponent = (args.target_ms * observed_blocks - elapsed_ms) / half_life_ms
    return max(1, round(anchor_difficulty * (2 ** exponent)))


def simulate(args):
    args = with_default_window(args)
    rng = random.Random(args.seed)
    base_difficulty = args.base_hashrate * args.target_ms // MILLIS_PER_SECOND
    minimum_difficulty = max(1, round(base_difficulty * args.minimum_ratio))
    difficulty = max(minimum_difficulty, round(base_difficulty * args.initial_ratio))
    p = SCALE
    process_noise = SCALE // args.process_noise_divisor
    gamma_alpha = float(args.gamma_prior_count)
    gamma_beta = gamma_alpha / max(1, difficulty * MILLIS_PER_SECOND // args.target_ms)
    max_history = max(args.window - 1, args.stable_limit)
    intervals = deque(maxlen=max_history)
    if args.measurement_source == "fixed-window":
        for _ in range(args.window - 1):
            intervals.append({
                "difficulty": difficulty,
                "solve_time_ms": float(args.target_ms),
                "side_blocks": 0,
            })
    rows = []
    states_before = deque(maxlen=max_history)
    side_debt = 0.0

    for height in range(args.blocks):
        states_before.append((gamma_alpha, gamma_beta))
        multiplier = hashrate_multiplier(args.scenario, height, args.step_at, difficulty, base_difficulty)
        hashrate = args.base_hashrate * multiplier
        mean_solve_time = difficulty * (1.0 + args.side_rate) * MILLIS_PER_SECOND / hashrate
        actual_solve_time = sample_solve_time(mean_solve_time, rng, args.random)
        if args.random:
            side_blocks = sample_poisson(args.side_rate, rng)
        else:
            side_debt += args.side_rate
            side_blocks = int(side_debt)
            side_debt -= side_blocks
        intervals.append({
            "difficulty": difficulty,
            "solve_time_ms": actual_solve_time,
            "side_blocks": side_blocks,
        })

        z, window_solve_time, measurement_count = measure_observation(args, intervals)
        x_prev = difficulty * MILLIS_PER_SECOND // args.target_ms

        if args.mode == "asert" and measurement_count < args.min_measurement_count:
            x_new = x_prev
        elif args.mode == "asert":
            x_new = asert_update(args, intervals) * MILLIS_PER_SECOND // args.target_ms
        elif args.mode == "replay-gamma" and measurement_count < args.min_measurement_count:
            x_new = x_prev
        elif args.mode == "replay-gamma":
            span = list(intervals)[-args.stable_limit:]
            first_state_index = len(intervals) - len(span)
            gamma_alpha, gamma_beta = states_before[first_state_index]
            gamma_effective_count = args.gamma_effective_count
            if args.gamma_fast_effective_count > 0:
                ratio = z / max(1, x_prev)
                if ratio > args.gamma_fast_threshold or ratio < 1 / args.gamma_fast_threshold:
                    gamma_effective_count = args.gamma_fast_effective_count

            gamma_alpha, gamma_beta, x_new = gamma_replay_update(
                gamma_alpha,
                gamma_beta,
                measurement_count,
                z,
                gamma_effective_count,
                args.gamma_max_rate_increase_percent,
            )
        elif args.mode == "windowed" and measurement_count < args.min_measurement_count:
            x_new = x_prev
        elif args.mode == "current":
            x_new, p = current_filter(z, x_prev, p, True)
        else:
            x_new, p = windowed_filter(z, x_prev, p, measurement_count, process_noise)

        next_difficulty = x_new * args.target_ms // MILLIS_PER_SECOND
        if next_difficulty < minimum_difficulty:
            difficulty = minimum_difficulty
            p = INITIAL_RELATIVE_COVAR if args.mode == "windowed" else SCALE
        else:
            difficulty = next_difficulty
        ideal_difficulty = hashrate * args.target_ms / MILLIS_PER_SECOND
        rows.append({
            "height": height,
            "hashrate": hashrate,
            "difficulty": difficulty,
            "ideal_difficulty": ideal_difficulty,
            "ratio": difficulty / ideal_difficulty,
            "actual_solve_time_ms": actual_solve_time,
            "window_solve_time_ms": window_solve_time,
            "observed_hashrate": z,
            "measurement_count": measurement_count,
            "side_blocks": side_blocks,
            "covariance": p,
            "gamma_alpha": gamma_alpha,
            "gamma_beta": gamma_beta,
        })

    return rows


def summarize(rows, step_at, window):
    post = rows[step_at:]
    after_window = rows[min(len(rows), step_at + window):]
    settled = rows[-100:] if len(rows) >= 100 else rows
    ratios = [row["ratio"] for row in post]
    after_window_ratios = [row["ratio"] for row in after_window] or ratios
    settled_error = mean(abs(row["ratio"] - 1.0) for row in settled)
    avg_window = mean(row["window_solve_time_ms"] for row in settled)
    avg_actual = mean(row["actual_solve_time_ms"] for row in settled)

    return {
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "post_window_min_ratio": min(after_window_ratios),
        "post_window_max_ratio": max(after_window_ratios),
        "settle_10pct_blocks": first_stable_offset(rows, step_at, 0.10),
        "settle_5pct_blocks": first_stable_offset(rows, step_at, 0.05),
        "avg_ordered_block_time_ms": ordered_block_time_ms(rows),
        "post_window_ordered_block_time_ms": ordered_block_time_ms(after_window or post),
        "settled_avg_abs_error": settled_error,
        "settled_window_solve_time_ms": avg_window,
        "settled_actual_solve_time_ms": avg_actual,
        "settled_ordered_block_time_ms": ordered_block_time_ms(settled),
        "settled_side_rate": side_rate(settled),
        "final_ratio": rows[-1]["ratio"],
    }


def ordered_block_time_ms(rows):
    emitted_blocks = sum(1 + row["side_blocks"] for row in rows)
    return sum(row["actual_solve_time_ms"] for row in rows) / emitted_blocks


def side_rate(rows):
    return sum(row["side_blocks"] for row in rows) / len(rows)


def first_stable_offset(rows, step_at, tolerance):
    suffix_max_error = 0.0
    stable_offset = None

    for i in range(len(rows) - 1, step_at - 1, -1):
        suffix_max_error = max(suffix_max_error, abs(rows[i]["ratio"] - 1.0))
        if suffix_max_error <= tolerance:
            stable_offset = i - step_at

    return stable_offset


def ratio_samples(rows, step_at):
    samples = {}
    for offset in RATIO_SAMPLE_OFFSETS:
        index = step_at + offset - 1
        if index < len(rows):
            samples[f"ratio_step_plus_{offset}"] = rows[index]["ratio"]
    return samples


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_trials(args):
    args = with_default_window(args)
    span = measurement_span(args)
    summaries = []
    tail_abs_errors = []
    tail_p95_abs_errors = []
    tail_actual_solve_times = []
    tail_ordered_block_times = []
    tail_side_rates = []

    for trial in range(args.trials):
        rows = simulate(clone_args(args, random=True, seed=args.seed + trial))
        summaries.append(summarize(rows, args.step_at, span))

        tail = rows[-min(1000, len(rows)):]
        tail_errors = [abs(row["ratio"] - 1.0) for row in tail]
        tail_abs_errors.append(mean(tail_errors))
        tail_p95_abs_errors.append(percentile(tail_errors, 0.95))
        tail_actual_solve_times.append(mean(row["actual_solve_time_ms"] for row in tail))
        tail_ordered_block_times.append(ordered_block_time_ms(tail))
        tail_side_rates.append(side_rate(tail))

    return {
        "avg_tail_abs_error": mean(tail_abs_errors),
        "avg_tail_p95_abs_error": mean(tail_p95_abs_errors),
        "worst_post_window_min_ratio": min(summary["post_window_min_ratio"] for summary in summaries),
        "worst_post_window_max_ratio": max(summary["post_window_max_ratio"] for summary in summaries),
        "avg_settled_avg_abs_error": mean(summary["settled_avg_abs_error"] for summary in summaries),
        "avg_tail_actual_solve_time_ms": mean(tail_actual_solve_times),
        "avg_tail_ordered_block_time_ms": mean(tail_ordered_block_times),
        "avg_tail_side_rate": mean(tail_side_rates),
        "avg_final_ratio": mean(summary["final_ratio"] for summary in summaries),
    }


def optimize_cases():
    return (
        OptimizeCase(
            "startup",
            "steady",
            1000,
            1,
            initial_ratio=0.5128,
            minimum_ratio=0.5128,
            base_hashrate=3900,
        ),
        OptimizeCase("steady", "steady", 2000, 100, warmup=200),
        OptimizeCase("step_up", "step-up", 1500, 250, warmup=250),
        OptimizeCase("step_down", "step-down", 1500, 250, warmup=250),
        OptimizeCase("step_cycle", "step-cycle", 2000, 250, warmup=250),
        OptimizeCase("ramp_up", "ramp-up", 1500, 250, warmup=250),
        OptimizeCase("side_blocks", "steady", 2000, 100, side_rate=0.30, warmup=200),
    )


def default_optimize_params(args):
    return OptimizeParams(
        args.gamma_effective_count,
        args.gamma_fast_effective_count,
        round(args.gamma_fast_threshold * 100),
        args.gamma_prior_count,
        round(args.gamma_max_rate_increase_percent),
        args.min_measurement_count,
    )


def suggest_optimize_params(trial):
    return OptimizeParams(
        trial.suggest_int("effective_count", 40, 260, step=4),
        trial.suggest_int("fast_effective_count", 4, 80, step=2),
        trial.suggest_int("fast_threshold_percent", 105, 180),
        trial.suggest_int("prior_count", 1, 12),
        trial.suggest_int("max_rate_increase_percent", 101, 110),
        trial.suggest_int("min_measurement_count", DEFAULT_MIN_MEASUREMENT_COUNT, DEFAULT_STABLE_LIMIT),
    )


def apply_optimize_params(args, params):
    return clone_args(
        args,
        mode="replay-gamma",
        measurement_source="dag-order",
        stable_limit=DEFAULT_STABLE_LIMIT,
        gamma_effective_count=params.effective_count,
        gamma_fast_effective_count=params.fast_effective_count,
        gamma_fast_threshold=params.fast_threshold_percent / 100,
        gamma_prior_count=params.prior_count,
        gamma_max_rate_increase_percent=params.max_rate_increase_percent,
        min_measurement_count=params.min_measurement_count,
        compare=False,
        csv=None,
        trials=1,
        random=True,
    )


def case_args(args, params, case, seed):
    return clone_args(
        apply_optimize_params(args, params),
        scenario=case.scenario,
        blocks=case.blocks,
        step_at=case.step_at,
        side_rate=case.side_rate,
        initial_ratio=case.initial_ratio,
        minimum_ratio=case.minimum_ratio,
        base_hashrate=case.base_hashrate,
        seed=seed,
    )


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_rows(rows, start):
    sample = rows[min(start, len(rows) - 1):]
    ratios = [row["ratio"] for row in sample]
    errors = [abs(ratio - 1.0) for ratio in ratios]
    volatility = mean(
        abs(
            math.log(max(rows[i]["difficulty"], 1) / max(rows[i - 1]["difficulty"], 1))
            - math.log(max(rows[i - 1]["difficulty"], 1) / max(rows[i - 2]["difficulty"], 1))
        )
        for i in range(max(start, 2), len(rows))
    )

    return {
        "sample_count": len(sample),
        "avg_abs": mean(errors),
        "mse": mean((ratio - 1.0) ** 2 for ratio in ratios),
        "volatility": volatility,
    }


def score_metrics(metrics, volatility_weight, loss_name):
    loss = metrics["mse"] if loss_name == "mse" else metrics["avg_abs"]
    return loss + volatility_weight * metrics["volatility"]


def score_params(args, params, cases, seeds, trial=None):
    total = 0.0
    total_weight = 0.0
    rows = []
    step = 0

    for case in cases:
        case_scores = []
        case_weighted_total = 0.0
        case_sample_count = 0
        for seed in seeds:
            run_args = case_args(args, params, case, seed + step * 10_000)
            simulated = simulate(run_args)
            start = max(case.warmup, case.step_at)
            metrics = evaluate_rows(simulated, start)
            score = score_metrics(metrics, args.optimize_volatility_weight, args.optimize_loss)
            case_scores.append(score)
            case_weighted_total += score * metrics["sample_count"]
            case_sample_count += metrics["sample_count"]
            step += 1

        case_score = mean(case_scores)
        total += case.weight * case_weighted_total
        total_weight += case.weight * case_sample_count
        rows.append((case.name, case_score))

        if trial is not None:
            trial.report(total / total_weight, len(rows))
            if trial.should_prune():
                raise TrialPruned()

    return total / total_weight, rows


def optimize_seeds(first, count):
    return tuple(first + i * 7919 for i in range(count))


def print_params(prefix, params):
    values = " ".join(f"{field}={getattr(params, field)}" for field in OPTIMIZE_PARAM_FIELDS)
    print(f"{prefix}: {values}")


def print_score_rows(title, rows):
    print(title)
    print("case\tscore")
    for name, score in rows:
        print(f"{name}\t{score:.6f}")


def params_from_trial(trial):
    return OptimizeParams(*(trial.params[field] for field in OPTIMIZE_PARAM_FIELDS))


def write_study_csv(path, study):
    fieldnames = ["number", "value", "state", *OPTIMIZE_PARAM_FIELDS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            row.update(trial.params)
            writer.writerow(row)


def optimize_params_note(params):
    return (
        "optimized replay-gamma: "
        f"effective_count={params.effective_count}, "
        f"fast_effective_count={params.fast_effective_count}, "
        f"fast_threshold={params.fast_threshold_percent / 100:.2f}, "
        f"prior_count={params.prior_count}, "
        f"max_rate_increase={params.max_rate_increase_percent}%, "
        f"min_measurement_count={params.min_measurement_count}"
    )


def plot_optimization(args, baseline, best):
    plot_args = clone_args(args, random=True, trials=1, csv=None, optimize=False)

    baseline_args = with_default_window(apply_optimize_params(plot_args, baseline))
    best_args = with_default_window(apply_optimize_params(plot_args, best))
    runs = [
        ("replay-gamma baseline", simulate(baseline_args)),
        ("replay-gamma optimized", simulate(best_args)),
    ]

    if args.compare:
        asert_args = with_default_window(clone_args(plot_args, mode="asert"))
        runs.append(("asert", simulate(asert_args)))

    note = f"loss={args.optimize_loss}, volatility_weight={args.optimize_volatility_weight}; {optimize_params_note(best)}"
    write_plot(args.plot, plot_args, runs, note)


def print_optimization(args):
    global TrialPruned
    try:
        import optuna
        from optuna.exceptions import ExperimentalWarning
        from optuna.exceptions import TrialPruned
    except ImportError as e:
        raise SystemExit("Optuna is required for --optimize. Install it with: python -m pip install optuna") from e

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cases = optimize_cases()
    train_seeds = optimize_seeds(args.seed, args.optimize_seeds)
    validate_seeds = optimize_seeds(args.seed + 1_000_003, args.optimize_validate_seeds)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        n_startup_trials=min(40, max(10, args.optimize_trials // 5)),
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(40, max(10, args.optimize_trials // 5)))
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    baseline = default_optimize_params(args)
    baseline_train, baseline_train_rows = score_params(args, baseline, cases, train_seeds)

    def objective(trial):
        params = suggest_optimize_params(trial)
        score, _ = score_params(args, params, cases, train_seeds, trial)
        return score

    study.enqueue_trial({field: getattr(baseline, field) for field in OPTIMIZE_PARAM_FIELDS})

    def progress_callback(study, _trial):
        if args.optimize_progress <= 0 or len(study.trials) % args.optimize_progress != 0:
            return
        if study.best_trial is not None:
            print(f"progress\ttrials={len(study.trials)}\tbest={study.best_value:.6f}", flush=True)

    study.optimize(
        objective,
        n_trials=args.optimize_trials,
        timeout=args.optimize_timeout,
        callbacks=[progress_callback],
    )

    best = params_from_trial(study.best_trial)
    best_train, best_train_rows = score_params(args, best, cases, train_seeds)
    baseline_validate, baseline_validate_rows = score_params(args, baseline, cases, validate_seeds)
    best_validate, best_validate_rows = score_params(args, best, cases, validate_seeds)

    print(f"optuna_trials: {len(study.trials)}")
    print(f"optimize_loss: {args.optimize_loss}")
    print(f"optimize_volatility_weight: {args.optimize_volatility_weight:.6f}")
    print(f"train_seeds: {','.join(map(str, train_seeds))}")
    print(f"validate_seeds: {','.join(map(str, validate_seeds))}")
    print_params("baseline", baseline)
    print_params("best", best)
    print()
    print("split\tbaseline\tbest\timprovement_percent")
    for split, baseline_score, best_score in (
        ("train", baseline_train, best_train),
        ("validate", baseline_validate, best_validate),
    ):
        improvement = (baseline_score - best_score) * 100 / baseline_score
        print(f"{split}\t{baseline_score:.6f}\t{best_score:.6f}\t{improvement:.2f}")

    print()
    print_score_rows("baseline_validate_by_case", baseline_validate_rows)
    print()
    print_score_rows("best_validate_by_case", best_validate_rows)

    if args.optimize_output:
        write_study_csv(args.optimize_output, study)
        print(f"study_csv: {args.optimize_output}")
    if args.plot:
        plot_optimization(args, baseline, best)
        print(f"plot: {args.plot}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rolling_ordered_block_times(rows, window):
    result = []
    queued = deque()
    elapsed_ms = 0.0
    emitted_blocks = 0

    for row in rows:
        queued.append(row)
        elapsed_ms += row["actual_solve_time_ms"]
        emitted_blocks += 1 + row["side_blocks"]
        if len(queued) > window:
            old = queued.popleft()
            elapsed_ms -= old["actual_solve_time_ms"]
            emitted_blocks -= 1 + old["side_blocks"]
        result.append(elapsed_ms / emitted_blocks / 1000)

    return result


def plot_change_heights(args):
    if args.scenario == "step-cycle":
        return range(args.step_at, args.blocks, 250)
    if args.scenario in ("steady",):
        return ()
    return (args.step_at,)


def plot_footer(args, block_time_summary, note):
    lines = [block_time_summary]
    if note:
        lines.append(note)
    command = getattr(args, "cli_command", None)
    if command:
        lines.append("CLI: " + command)
    return "\n".join(textwrap.fill(line, width=PLOT_FOOTER_WIDTH) for line in lines if line)


def write_plot(path, args, runs, note=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Matplotlib is required for --plot. Install it with: python -m pip install matplotlib") from e

    output = Path(path)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)

    base_difficulty = args.base_hashrate * args.target_ms / MILLIS_PER_SECOND
    heights = [row["height"] for row in runs[0][1]]
    actual_hashrate = [row["hashrate"] / args.base_hashrate for row in runs[0][1]]
    title = args.plot_title or f"{args.scenario}: {args.blocks}-block simulation, seed {args.seed}"
    block_time_summary = "; ".join(
        f"{label} avg_ordered_block_time={ordered_block_time_ms(rows) / 1000:.3f}s"
        for label, rows in runs
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )

    for label, rows in runs:
        difficulty = [row["difficulty"] / base_difficulty for row in rows]
        axes[0].plot(heights, difficulty, label=label, linewidth=1.5)
    axes[0].plot(heights, actual_hashrate, color="black", linestyle="--", linewidth=1.1, label="ideal")
    axes[0].set_ylabel("difficulty x baseline")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=min(4, len(runs) + 1), fontsize=9)

    for label, rows in runs:
        axes[1].plot(heights, [row["ratio"] for row in rows], label=label, linewidth=1.5)
    axes[1].axhline(1.0, color="black", linewidth=1.0)
    axes[1].axhspan(0.95, 1.05, color="green", alpha=0.08)
    axes[1].axhspan(0.90, 1.10, color="gold", alpha=0.08)
    axes[1].set_ylabel("difficulty / ideal")
    axes[1].set_xlabel("block height")
    axes[1].grid(True, alpha=0.25)

    for label, rows in runs:
        axes[2].plot(
            heights,
            rolling_ordered_block_times(rows, args.plot_block_time_window),
            label=label,
            linewidth=1.5,
        )
    axes[2].axhline(args.target_ms / 1000, color="black", linewidth=1.0)
    axes[2].set_ylabel(f"{args.plot_block_time_window}-block ordered avg time (s)")
    axes[2].set_xlabel("block height")
    axes[2].grid(True, alpha=0.25)

    for ax in axes:
        for height in plot_change_heights(args):
            ax.axvline(height, color="gray", alpha=0.22, linewidth=1)

    fig.suptitle(title)
    footer = plot_footer(args, block_time_summary, note)
    if footer:
        footer_lines = footer.count("\n") + 1
        bottom = min(0.16, 0.018 * footer_lines + 0.012)
        fig.text(0.5, 0.008, footer, ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout(rect=(0, bottom, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=args.plot_dpi)
    plt.close(fig)


def print_config(args):
    print(f"measurement_source: {args.measurement_source}")
    if args.measurement_source == "dag-order":
        print(f"dag_measurement_span: {args.stable_limit}")
        print(f"side_rate: {args.side_rate:.6f}")
        print(f"min_measurement_count: {args.min_measurement_count}")
    else:
        print(f"fixed_window_size: {args.window}")
    if args.initial_ratio != 1.0 or args.minimum_ratio != 0.0:
        print(f"initial_ratio: {args.initial_ratio:.6f}")
        print(f"minimum_ratio: {args.minimum_ratio:.6f}")
    if args.mode == "windowed":
        print(f"measurement: {args.measurement}")
        print(f"process_noise_divisor: {args.process_noise_divisor}")
        print(f"process_noise_covar: {SCALE // args.process_noise_divisor}")
    if args.mode == "replay-gamma":
        for field in (
            "gamma_effective_count",
            "gamma_fast_effective_count",
            "gamma_fast_threshold",
            "gamma_prior_count",
            "gamma_max_rate_increase_percent",
        ):
            print(f"{field}: {getattr(args, field)}")
    if args.mode == "asert":
        print(f"asert_half_life_blocks: {args.asert_half_life_blocks}")


def print_metrics(metrics):
    for key, value in metrics.items():
        if value is None:
            print(f"{key}: none")
        elif isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")


def print_run(args):
    args = with_default_window(args)
    if args.trials > 1:
        summary = summarize_trials(args)
        print(f"mode={args.mode} scenario={args.scenario} blocks={args.blocks} random=True trials={args.trials}")
        print_config(args)
        print_metrics(summary)
        return

    rows = simulate(args)
    summary = summarize(rows, args.step_at, measurement_span(args))

    print(f"mode={args.mode} scenario={args.scenario} blocks={args.blocks} random={args.random}")
    print_config(args)
    print_metrics({**summary, **ratio_samples(rows, args.step_at)})

    if args.csv:
        write_csv(args.csv, rows)
        print(f"csv: {args.csv}")
    if args.plot and not args.compare:
        write_plot(args.plot, args, [(args.mode, rows)])
        print(f"plot: {args.plot}")


def selected_scenarios(args):
    if args.scenario == "all":
        return SCENARIOS
    return (args.scenario,)


def with_default_window(args):
    measurement_source = args.measurement_source
    if measurement_source is None:
        measurement_source = "fixed-window" if args.mode == "current" else "dag-order"

    window = args.window
    if window is None:
        window = DEFAULT_CURRENT_WINDOW if measurement_source == "fixed-window" else DEFAULT_WINDOWED_WINDOW

    return clone_args(args, measurement_source=measurement_source, window=window)


def measurement_span(args):
    return args.stable_limit if args.measurement_source == "dag-order" else args.window


def print_scenario_runs(args):
    scenarios = selected_scenarios(args)
    for index, scenario in enumerate(scenarios):
        if len(scenarios) > 1:
            if index > 0:
                print()
            print(f"===== {scenario} =====")

        run_args = clone_args(args, scenario=scenario)
        if args.compare:
            for mode in COMPARE_MODES:
                print_run(clone_args(run_args, mode=mode))
                print()
            if args.plot:
                plot_runs = [
                    (mode, simulate(with_default_window(clone_args(run_args, mode=mode))))
                    for mode in COMPARE_MODES
                ]
                write_plot(args.plot, run_args, plot_runs)
                print(f"plot: {args.plot}")
        else:
            print_run(run_args)


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def validate_args(args):
    require(args.window is None or args.window >= 2, "--window must be at least 2")
    require(args.initial_ratio > 0, "--initial-ratio must be positive")
    require(args.gamma_effective_count > 1, "--gamma-effective-count must be greater than 1")
    require(
        args.gamma_fast_effective_count != 1,
        "--gamma-fast-effective-count must be 0 or greater than 1",
    )
    require(args.gamma_fast_threshold > 1, "--gamma-fast-threshold must be greater than 1")
    require(args.blocks > args.step_at, "--blocks must be greater than --step-at")

    for field, message in (
        ("stable_limit", "--stable-limit must be at least 1"),
        ("min_measurement_count", "--min-measurement-count must be at least 1"),
        ("trials", "--trials must be at least 1"),
        ("plot_block_time_window", "--plot-block-time-window must be at least 1"),
        ("optimize_trials", "--optimize-trials must be at least 1"),
        ("optimize_seeds", "--optimize-seeds must be at least 1"),
        ("optimize_validate_seeds", "--optimize-validate-seeds must be at least 1"),
    ):
        require(getattr(args, field) >= 1, message)

    for field, message in (
        ("process_noise_divisor", "--process-noise-divisor must be positive"),
        ("gamma_prior_count", "--gamma-prior-count must be positive"),
        ("asert_half_life_blocks", "--asert-half-life-blocks must be positive"),
        ("plot_dpi", "--plot-dpi must be positive"),
    ):
        require(getattr(args, field) > 0, message)

    for field, message in (
        ("side_rate", "--side-rate cannot be negative"),
        ("minimum_ratio", "--minimum-ratio cannot be negative"),
        ("gamma_fast_effective_count", "--gamma-fast-effective-count cannot be negative"),
        ("gamma_max_rate_increase_percent", "--gamma-max-rate-increase-percent cannot be negative"),
        ("optimize_progress", "--optimize-progress cannot be negative"),
        ("optimize_volatility_weight", "--optimize-volatility-weight cannot be negative"),
    ):
        require(getattr(args, field) >= 0, message)

    require(not (args.trials > 1 and args.csv), "--trials cannot be combined with --csv")
    require(not (args.trials > 1 and args.plot), "--trials cannot be combined with --plot")
    require(
        not (args.csv and (args.compare or args.scenario == "all")),
        "--csv can only be used with one mode and one scenario",
    )
    require(not (args.plot and args.scenario == "all"), "--plot can only be used with one scenario")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop XELIS DAA simulator")
    parser.add_argument("--mode", choices=["current", "windowed", "replay-gamma", "asert"], default="replay-gamma")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="step-down")
    parser.add_argument("--blocks", type=int, default=1000)
    parser.add_argument("--step-at", type=int, default=100)
    parser.add_argument("--window", type=int)
    parser.add_argument("--measurement-source", choices=["fixed-window", "dag-order"])
    parser.add_argument("--stable-limit", type=int, default=DEFAULT_STABLE_LIMIT)
    parser.add_argument("--min-measurement-count", type=int, default=DEFAULT_MIN_MEASUREMENT_COUNT)
    parser.add_argument("--side-rate", type=float, default=0.0)
    parser.add_argument("--initial-ratio", type=float, default=1.0)
    parser.add_argument("--minimum-ratio", type=float, default=0.0)
    parser.add_argument("--measurement", choices=["avg-time", "work"], default="work")
    parser.add_argument("--process-noise-divisor", type=int, default=DEFAULT_PROCESS_NOISE_DIVISOR)
    parser.add_argument("--gamma-effective-count", type=int, default=DEFAULT_GAMMA_EFFECTIVE_COUNT)
    parser.add_argument("--gamma-fast-effective-count", type=int, default=DEFAULT_GAMMA_FAST_EFFECTIVE_COUNT)
    parser.add_argument("--gamma-fast-threshold", type=float, default=DEFAULT_GAMMA_FAST_THRESHOLD)
    parser.add_argument("--gamma-prior-count", type=int, default=DEFAULT_GAMMA_PRIOR_COUNT)
    parser.add_argument("--gamma-max-rate-increase-percent", type=float, default=DEFAULT_GAMMA_MAX_RATE_INCREASE_PERCENT)
    parser.add_argument("--asert-half-life-blocks", type=float, default=DEFAULT_ASERT_HALF_LIFE_BLOCKS)
    parser.add_argument("--target-ms", type=int, default=DEFAULT_TARGET_MS)
    parser.add_argument("--base-hashrate", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--csv")
    parser.add_argument("--plot")
    parser.add_argument("--plot-title")
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument("--plot-block-time-window", type=int, default=25)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optimize-trials", type=int, default=300)
    parser.add_argument("--optimize-seeds", type=int, default=6)
    parser.add_argument("--optimize-validate-seeds", type=int, default=20)
    parser.add_argument("--optimize-timeout", type=int)
    parser.add_argument("--optimize-output")
    parser.add_argument("--optimize-progress", type=int, default=25)
    parser.add_argument("--optimize-loss", choices=["abs", "mse"], default="mse")
    parser.add_argument("--optimize-volatility-weight", type=float, default=1.0)
    args = parser.parse_args()
    args.cli_command = shlex.join(sys.argv)
    validate_args(args)
    if args.optimize:
        print_optimization(args)
    else:
        print_scenario_runs(args)


if __name__ == "__main__":
    main()
