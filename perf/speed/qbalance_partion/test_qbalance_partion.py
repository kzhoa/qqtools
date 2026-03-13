"""
tested by qq
date: 2026-03-13

results:
============================================================
Summary Table
============================================================
Summary Table: implementation aggregates
Model             | Cases | Avg Time (s) | Avg Imbalance | Avg CV   | Fastest Wins | Balance Wins
------------------+-------+--------------+---------------+----------+--------------+-------------
NumPy+Bucket      | 20    | 0.004521     | 1.006504      | 0.004193 | 18           | 0
Numba+Bucket      | 20    | 0.006792     | 1.006504      | 0.004193 | 2            | 0
Numba+ArrayArgmin | 20    | 0.007679     | 1.000136      | 0.000062 | 0            | 20
Numba+Heap        | 20    | 0.007852     | 1.000136      | 0.000062 | 0            | 20
Numba+ArrayScan   | 20    | 0.007999     | 1.000136      | 0.000062 | 0            | 20
Python+BucketLoop | 20    | 0.009229     | 1.006504      | 0.004193 | 0            | 0
Python+Heapq      | 20    | 0.026457     | 1.000136      | 0.000062 | 0            | 20
NumPy+ArrayArgmin | 20    | 0.069653     | 1.000136      | 0.000062 | 0            | 20
------------------------------------------------------------------------------------------
Fastest by average time: NumPy+Bucket (0.004521s)
Best by average imbalance: Numba+ArrayArgmin (1.0001)
Pivot Table: time(s) by N,K
N      | K  | NumPy+ArrayArgmin | NumPy+Bucket | Numba+ArrayArgmin | Numba+ArrayScan | Numba+Bucket | Numba+Heap | Python+BucketLoop | Python+Heapq
-------+----+-------------------+--------------+-------------------+-----------------+--------------+------------+-------------------+-------------
100    | 8  | 0.000365          | 0.000078     | 0.000083          | 0.000080        | 0.000068     | 0.000074   | 0.000109          | 0.000151
1000   | 8  | 0.001487          | 0.000160     | 0.000202          | 0.000193        | 0.000180     | 0.000197   | 0.000268          | 0.000736
10000  | 8  | 0.013417          | 0.000825     | 0.001384          | 0.001345        | 0.001382     | 0.001455   | 0.001806          | 0.004926
50000  | 8  | 0.067697          | 0.003777     | 0.007042          | 0.006978        | 0.007004     | 0.006979   | 0.008956          | 0.024322
100000 | 2  | 0.136439          | 0.007527     | 0.013055          | 0.013153        | 0.012412     | 0.013625   | 0.018042          | 0.044265
100000 | 4  | 0.143729          | 0.007664     | 0.013697          | 0.013764        | 0.012665     | 0.014479   | 0.017677          | 0.045012
100000 | 8  | 0.140523          | 0.007825     | 0.014629          | 0.016173        | 0.013147     | 0.015319   | 0.017912          | 0.049180
100000 | 16 | 0.140748          | 0.013863     | 0.016426          | 0.016822        | 0.013803     | 0.016894   | 0.019275          | 0.061090
100000 | 32 | 0.142587          | 0.009547     | 0.018255          | 0.018938        | 0.014400     | 0.018216   | 0.020023          | 0.061490
100000 | 64 | 0.153876          | 0.010936     | 0.021653          | 0.022546        | 0.015921     | 0.020606   | 0.021324          | 0.067335
200000 | 8  | 0.287301          | 0.018444     | 0.031092          | 0.033940        | 0.028675     | 0.032826   | 0.037952          | 0.111515
Pivot Table: imbalance by N,K
N      | K  | NumPy+ArrayArgmin | NumPy+Bucket | Numba+ArrayArgmin | Numba+ArrayScan | Numba+Bucket | Numba+Heap | Python+BucketLoop | Python+Heapq
-------+----+-------------------+--------------+-------------------+-----------------+--------------+------------+-------------------+-------------
100    | 8  | 1.002592          | 1.097161     | 1.002592          | 1.002592        | 1.097161     | 1.002592   | 1.097161          | 1.002592
1000   | 8  | 1.000046          | 1.007236     | 1.000046          | 1.000046        | 1.007236     | 1.000046   | 1.007236          | 1.000046
10000  | 8  | 1.000000          | 1.000706     | 1.000000          | 1.000000        | 1.000706     | 1.000000   | 1.000706          | 1.000000
50000  | 8  | 1.000000          | 1.000139     | 1.000000          | 1.000000        | 1.000139     | 1.000000   | 1.000139          | 1.000000
100000 | 2  | 1.000000          | 1.000010     | 1.000000          | 1.000000        | 1.000010     | 1.000000   | 1.000010          | 1.000000
100000 | 4  | 1.000000          | 1.000030     | 1.000000          | 1.000000        | 1.000030     | 1.000000   | 1.000030          | 1.000000
100000 | 8  | 1.000000          | 1.000070     | 1.000000          | 1.000000        | 1.000070     | 1.000000   | 1.000070          | 1.000000
100000 | 16 | 1.000000          | 1.000151     | 1.000000          | 1.000000        | 1.000151     | 1.000000   | 1.000151          | 1.000000
100000 | 32 | 1.000000          | 1.000311     | 1.000000          | 1.000000        | 1.000311     | 1.000000   | 1.000311          | 1.000000
100000 | 64 | 1.000000          | 1.000629     | 1.000000          | 1.000000        | 1.000629     | 1.000000   | 1.000629          | 1.000000
200000 | 8  | 1.000000          | 1.000035     | 1.000000          | 1.000000        | 1.000035     | 1.000000   | 1.000035          | 1.000000
"""

import gc
import heapq
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from qqtools.data.qbalanced_partition import qbalanced_partition

try:
    import numba
except ModuleNotFoundError:
    numba = None


PartitionList = List[List[int]]
ImplementationFunc = Callable[[np.ndarray, int], PartitionList]


def assignment_to_partitions(assignment: np.ndarray, num_parts: int) -> PartitionList:
    return [np.where(assignment == i)[0].tolist() for i in range(num_parts)]


HAS_NUMBA = numba is not None


if HAS_NUMBA:
    # 1. Numba array-argmin baseline
    @numba.njit(cache=True)
    def _balanced_partition_numba_array_argmin_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        n = len(sizes)
        sort_idx = np.argsort(-sizes)
        assignment = np.zeros(n, dtype=np.int32)
        heap_sums = np.zeros(num_parts, dtype=np.float64)

        for idx in sort_idx:
            p_id = np.argmin(heap_sums)
            assignment[idx] = p_id
            heap_sums[p_id] += sizes[idx]

        return assignment

    # 2. Numba array-scan baseline
    @numba.njit(cache=True)
    def _balanced_partition_numba_array_scan_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        n = len(sizes)
        sort_idx = np.argsort(-sizes)
        assignment = np.zeros(n, dtype=np.int32)
        heap_sums = np.zeros(num_parts, dtype=np.float64)

        for idx in sort_idx:
            min_idx = 0
            min_val = heap_sums[0]
            for i in range(1, num_parts):
                if heap_sums[i] < min_val:
                    min_val = heap_sums[i]
                    min_idx = i

            assignment[idx] = min_idx
            heap_sums[min_idx] += sizes[idx]

        return assignment

    @numba.njit(cache=True)
    def _heap_entry_less(
        heap_sums: np.ndarray,
        heap_ids: np.ndarray,
        left: int,
        right: int,
    ) -> bool:
        if heap_sums[left] < heap_sums[right]:
            return True
        if heap_sums[left] > heap_sums[right]:
            return False
        return heap_ids[left] < heap_ids[right]

    @numba.njit(cache=True)
    def _heap_sift_down(
        heap_sums: np.ndarray,
        heap_ids: np.ndarray,
        size: int,
    ) -> None:
        root = 0
        while True:
            left = 2 * root + 1
            if left >= size:
                return

            child = left
            right = left + 1
            if right < size and _heap_entry_less(heap_sums, heap_ids, right, left):
                child = right

            if _heap_entry_less(heap_sums, heap_ids, child, root):
                sum_tmp = heap_sums[root]
                heap_sums[root] = heap_sums[child]
                heap_sums[child] = sum_tmp

                id_tmp = heap_ids[root]
                heap_ids[root] = heap_ids[child]
                heap_ids[child] = id_tmp
                root = child
                continue

            return

    # 3. Numba heap baseline: a numba-friendly binary min-heap equivalent to the heapq strategy
    @numba.njit(cache=True)
    def _balanced_partition_numba_heap_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        n = len(sizes)
        sort_idx = np.argsort(-sizes)
        assignment = np.zeros(n, dtype=np.int32)
        heap_sums = np.zeros(num_parts, dtype=np.float64)
        heap_ids = np.arange(num_parts, dtype=np.int32)

        for idx in sort_idx:
            p_id = heap_ids[0]
            assignment[idx] = p_id
            heap_sums[0] += sizes[idx]
            _heap_sift_down(heap_sums, heap_ids, num_parts)

        return assignment

    # 4. Numba bucket baseline: jit version of the bucket assignment strategy
    @numba.njit(cache=True)
    def _balanced_partition_numba_bucket_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        n = len(sizes)
        sort_idx = np.argsort(-sizes)
        assignment = np.zeros(n, dtype=np.int32)

        for i in range(n):
            assignment[sort_idx[i]] = i % num_parts

        return assignment

else:

    def _balanced_partition_numba_array_argmin_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        raise RuntimeError("numba is not installed, cannot run the Numba array-argmin baseline.")

    def _balanced_partition_numba_array_scan_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        raise RuntimeError("numba is not installed, cannot run the Numba array-scan baseline.")

    def _balanced_partition_numba_heap_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        raise RuntimeError("numba is not installed, cannot run the Numba heap baseline.")

    def _balanced_partition_numba_bucket_assignment(
        sizes: np.ndarray,
        num_parts: int,
    ) -> np.ndarray:
        raise RuntimeError("numba is not installed, cannot run the Numba bucket baseline.")


def balanced_partition_numba_array_argmin(sizes: np.ndarray, num_parts: int) -> PartitionList:
    assignment = _balanced_partition_numba_array_argmin_assignment(sizes, num_parts)
    return assignment_to_partitions(assignment, num_parts)


def balanced_partition_numba_array_scan(sizes: np.ndarray, num_parts: int) -> PartitionList:
    assignment = _balanced_partition_numba_array_scan_assignment(sizes, num_parts)
    return assignment_to_partitions(assignment, num_parts)


def balanced_partition_numba_heap(sizes: np.ndarray, num_parts: int) -> PartitionList:
    assignment = _balanced_partition_numba_heap_assignment(sizes, num_parts)
    return assignment_to_partitions(assignment, num_parts)


def balanced_partition_numba_bucket(sizes: np.ndarray, num_parts: int) -> PartitionList:
    assignment = _balanced_partition_numba_bucket_assignment(sizes, num_parts)
    return assignment_to_partitions(assignment, num_parts)


def balanced_partition_numba(sizes: np.ndarray, num_parts: int) -> PartitionList:
    return balanced_partition_numba_array_scan(sizes, num_parts)


# 5. Python heapq implementation
def balanced_partition_python_heapq(sizes: np.ndarray, num_parts: int) -> PartitionList:
    n = len(sizes)
    sort_idx = np.argsort(-sizes)
    assignment = np.zeros(n, dtype=np.int32)
    heap = [(0.0, i) for i in range(num_parts)]
    heapq.heapify(heap)

    for idx in sort_idx:
        current_sum, p_id = heapq.heappop(heap)
        assignment[idx] = p_id
        heapq.heappush(heap, (current_sum + sizes[idx], p_id))

    return assignment_to_partitions(assignment, num_parts)


# 6. NumPy array-argmin implementation
def balanced_partition_numpy_array_argmin(sizes: np.ndarray, num_parts: int) -> PartitionList:
    n = len(sizes)
    sort_idx = np.argsort(-sizes)
    assignment = np.zeros(n, dtype=np.int32)
    heap_sums = np.zeros(num_parts, dtype=np.float64)

    for idx in sort_idx:
        p_id = int(np.argmin(heap_sums))
        assignment[idx] = p_id
        heap_sums[p_id] += sizes[idx]

    return assignment_to_partitions(assignment, num_parts)


# 7. Python bucket assignment strategy
def balanced_partition_python_bucket_loop(sizes: np.ndarray, num_parts: int) -> PartitionList:
    n = len(sizes)
    sort_idx = np.argsort(-sizes)
    assignment = np.zeros(n, dtype=np.int32)

    for i, idx in enumerate(sort_idx):
        assignment[idx] = i % num_parts

    return assignment_to_partitions(assignment, num_parts)


ALL_IMPLEMENTATIONS: List[Tuple[str, ImplementationFunc]] = []
if HAS_NUMBA:
    ALL_IMPLEMENTATIONS.extend(
        [
            ("Numba+ArrayArgmin", balanced_partition_numba_array_argmin),
            ("Numba+ArrayScan", balanced_partition_numba_array_scan),
            ("Numba+Heap", balanced_partition_numba_heap),
            ("Numba+Bucket", balanced_partition_numba_bucket),
        ]
    )
ALL_IMPLEMENTATIONS.extend(
    [
        ("Python+Heapq", balanced_partition_python_heapq),
        ("NumPy+ArrayArgmin", balanced_partition_numpy_array_argmin),
        ("qbalanced_partition", qbalanced_partition),
        ("Python+BucketLoop", balanced_partition_python_bucket_loop),
    ]
)

BenchmarkCase = Dict[str, int | str | bool]
SUMMARY_ROWS: List[Dict[str, float | int | str]] = []


def warmup_numba() -> None:
    if not HAS_NUMBA:
        print(
            "Note: numba is not installed in the current environment. Numba baselines and the Numba-only benchmark will be skipped."
        )
        return

    warmup_sizes = np.random.default_rng(0).random(1024).astype(np.float64)
    _balanced_partition_numba_array_argmin_assignment(warmup_sizes, 4)
    _balanced_partition_numba_array_scan_assignment(warmup_sizes, 4)
    _balanced_partition_numba_heap_assignment(warmup_sizes, 4)
    _balanced_partition_numba_bucket_assignment(warmup_sizes, 4)


def build_benchmark_cases() -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    n_sweep = [100, 1000, 10000, 50000, 100000, 200000]
    fixed_k = 8
    for size_n in n_sweep:
        cases.append(
            {
                "suite": "ParameterSweep",
                "case_name": f"N{size_n}_K{fixed_k}_Uniform",
                "n": size_n,
                "k": fixed_k,
                "distribution": "uniform",
                "seed": 1000 + size_n,
                "numba_only": False,
            }
        )

    k_sweep = [2, 4, 8, 16, 32, 64]
    fixed_n = 100000
    existing_case_names = {str(case["case_name"]) for case in cases}
    for num_parts in k_sweep:
        case_name = f"N{fixed_n}_K{num_parts}_Uniform"
        if case_name in existing_case_names:
            continue
        cases.append(
            {
                "suite": "ParameterSweep",
                "case_name": case_name,
                "n": fixed_n,
                "k": num_parts,
                "distribution": "uniform",
                "seed": 2000 + num_parts,
                "numba_only": False,
            }
        )

    distribution_cases = [
        ("Uniform", "uniform"),
        ("Normal", "normal"),
        ("SkewedBeta", "beta"),
        ("LongTailExp", "exponential"),
        ("RepeatedLevels", "repeated_levels"),
        ("FewHugeValues", "few_huge"),
    ]
    for index, (label, distribution) in enumerate(distribution_cases):
        cases.append(
            {
                "suite": "DistributionBenchmark",
                "case_name": f"N10000_K8_{label}",
                "n": 10000,
                "k": 8,
                "distribution": distribution,
                "seed": 3000 + index,
                "numba_only": False,
            }
        )

    numba_cases = [
        ("NumbaFocus_N1000_K8", 1000, 8),
        ("NumbaFocus_N10000_K8", 10000, 8),
        ("NumbaFocus_N50000_K8", 50000, 8),
    ]
    for index, (case_name, size_n, num_parts) in enumerate(numba_cases):
        cases.append(
            {
                "suite": "NumbaOnly",
                "case_name": case_name,
                "n": size_n,
                "k": num_parts,
                "distribution": "uniform",
                "seed": 4000 + index,
                "numba_only": True,
            }
        )

    return cases


BENCHMARK_CASES = build_benchmark_cases()


def generate_sizes(case: BenchmarkCase) -> np.ndarray:
    size_n = int(case["n"])
    seed = int(case["seed"])
    distribution = str(case["distribution"])
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        return rng.random(size_n).astype(np.float64)
    if distribution == "normal":
        return np.clip(rng.normal(0.5, 0.2, size_n), 0.0, 1.0).astype(np.float64)
    if distribution == "beta":
        return rng.beta(2.0, 5.0, size_n).astype(np.float64)
    if distribution == "exponential":
        return np.clip(rng.exponential(0.3, size_n), 0.0, 1.0).astype(np.float64)
    if distribution == "repeated_levels":
        levels = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=np.float64)
        return rng.choice(levels, size=size_n, replace=True).astype(np.float64)
    if distribution == "few_huge":
        base = rng.random(size_n).astype(np.float64) * 0.05
        huge_count = max(1, size_n // 100)
        huge_indices = rng.choice(size_n, size=huge_count, replace=False)
        base[huge_indices] = rng.uniform(0.9, 1.0, size=huge_count)
        return base

    raise ValueError(f"Unsupported distribution: {distribution}")


def format_cell(value: object) -> str:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.6f}"
        return "inf"
    if isinstance(value, bool):
        return "Y" if value else ""
    return str(value)


def print_table(title: str, columns: Sequence[Tuple[str, str]], rows: Sequence[Dict[str, object]]) -> None:
    print(title)
    if not rows:
        print("  (no rows)")
        return

    widths: List[int] = []
    for header, key in columns:
        max_width = len(header)
        for row in rows:
            max_width = max(max_width, len(format_cell(row.get(key, ""))))
        widths.append(max_width)

    header_line = " | ".join(header.ljust(widths[index]) for index, (header, _) in enumerate(columns))
    separator_line = "-+-".join("-" * widths[index] for index in range(len(columns)))
    print(header_line)
    print(separator_line)

    for row in rows:
        line = " | ".join(format_cell(row.get(key, "")).ljust(widths[index]) for index, (_, key) in enumerate(columns))
        print(line)


def print_case_table(case: BenchmarkCase, case_results: Sequence[Dict[str, float | int | str]]) -> None:
    sorted_results = sorted(
        case_results,
        key=lambda row: (float(row["time"]), float(row["balance"]), str(row["implementation"])),
    )
    table_rows: List[Dict[str, object]] = []
    for row in sorted_results:
        table_rows.append(
            {
                "implementation": str(row["implementation"]),
                "time": float(row["time"]),
                "balance": float(row["balance"]),
                "cv": float(row["cv"]),
                "fastest": bool(row["fastest_win"]),
                "best_balance": bool(row["balance_win"]),
            }
        )

    print_table(
        (
            f"Case Table: {case['case_name']} "
            f"[suite={case['suite']}, N={case['n']}, K={case['k']}, dist={case['distribution']}]"
        ),
        [
            ("Implementation", "implementation"),
            ("Time (s)", "time"),
            ("Imbalance", "balance"),
            ("CV", "cv"),
            ("Fastest", "fastest"),
            ("BestBalance", "best_balance"),
        ],
        table_rows,
    )


def print_pivot_table(metric_key: str, title: str) -> None:
    pivot_rows = [row for row in SUMMARY_ROWS if str(row["suite"]) == "ParameterSweep"]
    if not pivot_rows:
        return

    implementations = sorted({str(row["implementation"]) for row in pivot_rows})
    grouped_rows: Dict[Tuple[int, int], Dict[str, Dict[str, float | int | str]]] = defaultdict(dict)
    for row in pivot_rows:
        grouped_rows[(int(row["n"]), int(row["k"]))][str(row["implementation"])] = row

    table_rows: List[Dict[str, object]] = []
    for (size_n, num_parts), row_map in sorted(grouped_rows.items()):
        entry: Dict[str, object] = {
            "n": size_n,
            "k": num_parts,
        }
        for impl_name in implementations:
            metric_value = row_map.get(impl_name, {}).get(metric_key, "")
            entry[impl_name] = metric_value
        table_rows.append(entry)

    columns: List[Tuple[str, str]] = [("N", "n"), ("K", "k")]
    columns.extend((impl_name, impl_name) for impl_name in implementations)
    print_table(title, columns, table_rows)


def evaluate_partition(
    sizes: np.ndarray,
    partitions: PartitionList,
    num_parts: int,
    verbose: bool = False,
) -> Tuple[float, float, List[float]]:
    all_indices: List[int] = []
    for part in partitions:
        all_indices.extend(part)

    n = len(sizes)
    if len(partitions) != num_parts:
        if verbose:
            print(f"  Warning: partition count mismatch: {len(partitions)} != {num_parts}")
        return float("inf"), float("inf"), []

    if len(all_indices) != n or len(set(all_indices)) != n:
        if verbose:
            print(
                f"  Warning: assigned element count mismatch: " f"unique indices {len(set(all_indices))}, expected {n}"
            )
        return float("inf"), float("inf"), []

    part_loads = [float(np.sum(sizes[np.array(part, dtype=np.int64)])) if part else 0.0 for part in partitions]
    avg_load = sum(part_loads) / num_parts
    if avg_load == 0:
        return float("inf"), float("inf"), part_loads

    balance_ratio = max(part_loads) / avg_load
    std_dev = float(np.std(part_loads))
    cv = std_dev / avg_load

    if verbose:
        print(f"  Load range: {min(part_loads):.4f} ~ {max(part_loads):.4f}, average: {avg_load:.4f}")
        print(f"  Imbalance ratio: {balance_ratio:.4f}, coefficient of variation: {cv:.4f}")

    return balance_ratio, cv, part_loads


def benchmark_case(
    case: BenchmarkCase,
    sizes: np.ndarray,
    implementations: Sequence[Tuple[str, ImplementationFunc]],
) -> List[Dict[str, float | int | str]]:
    suite_name = str(case["suite"])
    case_name = str(case["case_name"])
    num_parts = int(case["k"])

    print(f"\nTest case: {case_name} (N={len(sizes)}, partitions={num_parts})")
    print("-" * 60)

    case_results: List[Dict[str, float | int | str]] = []
    scenario_key = f"{suite_name}/{case_name}"

    for impl_name, impl_func in implementations:
        gc.collect()
        start = time.perf_counter()
        partitions = impl_func(sizes, num_parts)
        elapsed = time.perf_counter() - start
        balance, cv, _ = evaluate_partition(sizes, partitions, num_parts)

        row: Dict[str, float | int | str] = {
            "suite": suite_name,
            "scenario": scenario_key,
            "case_name": case_name,
            "implementation": impl_name,
            "time": elapsed,
            "balance": balance,
            "cv": cv,
            "n": len(sizes),
            "k": num_parts,
        }
        case_results.append(row)

    fastest_time = min(float(row["time"]) for row in case_results)
    best_balance = min(float(row["balance"]) for row in case_results)
    for row in case_results:
        row["fastest_win"] = int(abs(float(row["time"]) - fastest_time) < 1e-12)
        row["balance_win"] = int(abs(float(row["balance"]) - best_balance) < 1e-12)
        SUMMARY_ROWS.append(row.copy())

    print_case_table(case, case_results)

    return case_results


def run_benchmark_cases() -> None:
    print("\n" + "=" * 60)
    print("Expanded Benchmark Matrix")
    print("=" * 60)

    for case in BENCHMARK_CASES:
        if bool(case["numba_only"]) and not HAS_NUMBA:
            print(f"\nSkipping {case['case_name']}: numba is not installed.")
            continue

        sizes = generate_sizes(case)
        benchmark_case(case, sizes, ALL_IMPLEMENTATIONS)


def test_correctness() -> None:
    print("\n" + "=" * 60)
    print("Correctness Check")
    print("=" * 60)

    sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    num_parts = 2

    print(f"Input data: {sizes}")
    print(f"Partition count: {num_parts}\n")

    for impl_name, impl_func in ALL_IMPLEMENTATIONS:
        partitions = impl_func(sizes, num_parts)
        balance, cv, _ = evaluate_partition(sizes, partitions, num_parts, verbose=True)

        print(f"{impl_name}:")
        for index, part in enumerate(partitions):
            part_sizes = [float(sizes[idx]) for idx in part]
            part_sum = sum(part_sizes)
            print(f"  Partition {index}: indices {part}, values {part_sizes}, sum={part_sum:.1f}")

        if np.isfinite(balance):
            print(
                f"  ✓ Correct: all elements were assigned, "
                f"imbalance ratio={balance:.4f}, coefficient of variation={cv:.4f}"
            )
        else:
            print("  ✗ Error: element assignment is incomplete")
        print()


def print_summary() -> None:
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)

    if not SUMMARY_ROWS:
        print("No benchmark results are available for summary.")
        return

    aggregate: Dict[str, Dict[str, float | int]] = defaultdict(
        lambda: {
            "cases": 0,
            "time_sum": 0.0,
            "balance_sum": 0.0,
            "cv_sum": 0.0,
            "fastest_wins": 0,
            "balanced_wins": 0,
        }
    )

    for row in SUMMARY_ROWS:
        aggregate[str(row["implementation"])]["cases"] += 1
        aggregate[str(row["implementation"])]["time_sum"] += float(row["time"])
        aggregate[str(row["implementation"])]["balance_sum"] += float(row["balance"])
        aggregate[str(row["implementation"])]["cv_sum"] += float(row["cv"])
        aggregate[str(row["implementation"])]["fastest_wins"] += int(row["fastest_win"])
        aggregate[str(row["implementation"])]["balanced_wins"] += int(row["balance_win"])

    summary_table_rows: List[Dict[str, object]] = []
    for impl_name, stats in sorted(
        aggregate.items(),
        key=lambda item: (
            item[1]["time_sum"] / item[1]["cases"],
            item[1]["balance_sum"] / item[1]["cases"],
        ),
    ):
        cases = int(stats["cases"])
        summary_table_rows.append(
            {
                "model": impl_name,
                "cases": cases,
                "avg_time": float(stats["time_sum"]) / cases,
                "avg_balance": float(stats["balance_sum"]) / cases,
                "avg_cv": float(stats["cv_sum"]) / cases,
                "fastest_wins": int(stats["fastest_wins"]),
                "balance_wins": int(stats["balanced_wins"]),
            }
        )

    print_table(
        "Summary Table: implementation aggregates",
        [
            ("Model", "model"),
            ("Cases", "cases"),
            ("Avg Time (s)", "avg_time"),
            ("Avg Imbalance", "avg_balance"),
            ("Avg CV", "avg_cv"),
            ("Fastest Wins", "fastest_wins"),
            ("Balance Wins", "balance_wins"),
        ],
        summary_table_rows,
    )

    fastest_overall = min(
        aggregate.items(),
        key=lambda item: item[1]["time_sum"] / item[1]["cases"],
    )
    balanced_overall = min(
        aggregate.items(),
        key=lambda item: item[1]["balance_sum"] / item[1]["cases"],
    )
    print("-" * 90)
    print(
        f"Fastest by average time: {fastest_overall[0]} "
        f"({fastest_overall[1]['time_sum'] / fastest_overall[1]['cases']:.6f}s)"
    )
    print(
        f"Best by average imbalance: {balanced_overall[0]} "
        f"({balanced_overall[1]['balance_sum'] / balanced_overall[1]['cases']:.4f})"
    )
    print_pivot_table("time", "Pivot Table: time(s) by N,K")
    print_pivot_table("balance", "Pivot Table: imbalance by N,K")


def run_all_tests() -> None:
    SUMMARY_ROWS.clear()
    warmup_numba()

    print("Balanced Partition Benchmark Suite")
    print("=" * 60)

    test_correctness()
    run_benchmark_cases()
    print_summary()


if __name__ == "__main__":
    run_all_tests()
