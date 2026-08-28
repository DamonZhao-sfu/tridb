"""Task registry and resource profiles for the OpenEvolve CSR workload."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASKS = Path(__file__).resolve().parent / "tasks"
CALIBRATION = ROOT / "experiments/cross_session_reuse/calibration"


@dataclass(frozen=True)
class TaskVariant:
    family: str
    task_id: str
    initial_program: Path
    evaluator: Path
    system_message: str
    diff_based_evolution: bool = False
    max_tokens: int = 2048
    evaluator_timeout: int = 60
    environment: dict[str, str] = field(default_factory=dict)
    priority: str = "coverage"


def _circle(n: int) -> TaskVariant:
    return TaskVariant(
        family="circle_packing",
        task_id=f"circle_c{n}",
        initial_program=TASKS / "circle_packing_initial.py",
        evaluator=TASKS / "circle_packing_evaluator.py",
        system_message=(
            f"/no_think Improve a constructor that packs exactly {n} positive-radius, "
            "non-overlapping circles inside a unit square. Return complete Python code "
            "defining run_packing(); maximize the independently recomputed sum of radii."
        ),
        environment={"CSR_CIRCLE_N": str(n)},
        priority="focus",
    )


def _gpu(label: str, rows: int, columns: int) -> TaskVariant:
    return TaskVariant(
        family="gpu_kernel_optimization",
        task_id=f"gpu_fused_silu_{label}",
        initial_program=TASKS / "gpu_fused_silu_initial.py",
        evaluator=TASKS / "gpu_fused_silu_evaluator.py",
        system_message=(
            "/no_think Optimize fused_silu(x, bias) for an NVIDIA GPU while preserving "
            "float32 correctness on unseen inputs. You may use PyTorch, torch.compile, "
            "or Triton. Return complete Python code. Never cache outputs or specialize "
            f"to values. The primary benchmark shape is ({rows}, {columns})."
        ),
        max_tokens=3072,
        evaluator_timeout=180,
        environment={
            "CSR_GPU_ROWS": str(rows),
            "CSR_GPU_COLUMNS": str(columns),
            "CUDA_VISIBLE_DEVICES": "1",
        },
        priority="focus",
    )


def _repair(variant: str, target: str) -> TaskVariant:
    return TaskVariant(
        family="code_repair",
        task_id=f"code_repair_{variant}",
        initial_program=TASKS / "code_repair_initial.py",
        evaluator=TASKS / "code_repair_evaluator.py",
        system_message=(
            f"/no_think Repair the function {target} in the supplied Python module. "
            "Return complete code and preserve all public function signatures. Hidden "
            "tests include edge cases; maximize the independently computed pass rate."
        ),
        environment={"CSR_REPAIR_VARIANT": variant},
        priority="focus",
    )


def all_variants() -> list[TaskVariant]:
    """Return all currently implemented task variants in stable order."""
    return [
        TaskVariant(
            family="continuous_function_optimization",
            task_id="function_f1",
            initial_program=(
                ROOT
                / "experiments/e0/openevolve_function_minimization/initial_program.py"
            ),
            evaluator=CALIBRATION / "function_f1_evaluator.py",
            system_message=(
                "/no_think Improve the deterministic function-minimization search. "
                "Modify only the EVOLVE block. Keep run_search() bounded and "
                "reproducible; maximize combined_score."
            ),
            diff_based_evolution=True,
            max_tokens=1024,
        ),
        _circle(12),
        _circle(18),
        _circle(26),
        _gpu("256x256", 256, 256),
        _gpu("1024x1024", 1024, 1024),
        _gpu("4096x256", 4096, 256),
        _repair("chunk", "chunked"),
        _repair("path", "normalize_path"),
        _repair("record", "parse_record"),
        TaskVariant(
            family="symbolic_regression",
            task_id="symbolic_sine_quadratic",
            initial_program=TASKS / "symbolic_regression_initial.py",
            evaluator=TASKS / "symbolic_regression_evaluator.py",
            system_message=(
                "/no_think Improve predict(x) to fit a hidden smooth function over "
                "[-3, 3]. Return complete, vectorized Python code; maximize held-out "
                "accuracy while keeping outputs finite."
            ),
        ),
        TaskVariant(
            family="text_to_sql",
            task_id="text_to_sql_analytics",
            initial_program=TASKS / "text_to_sql_initial.py",
            evaluator=TASKS / "text_to_sql_evaluator.py",
            system_message=(
                "/no_think Improve generate_sql(question) for the documented SQLite "
                "analytics schema. Return complete Python code. SQL is executed against "
                "hidden data and must answer every supported question correctly."
            ),
        ),
        TaskVariant(
            family="prompt_optimization",
            task_id="prompt_json_extraction",
            initial_program=TASKS / "prompt_optimization_initial.py",
            evaluator=TASKS / "prompt_optimization_evaluator.py",
            system_message=(
                "/no_think Improve build_prompt(text) for deterministic JSON field "
                "extraction. The returned prompt is tested by a fixed local model on "
                "held-out examples. Return complete Python code and keep prompts concise."
            ),
            max_tokens=2048,
            evaluator_timeout=300,
        ),
    ]


def profile_variants(profile: str) -> list[TaskVariant]:
    variants = all_variants()
    if profile == "coverage":
        canonical_ids = {
            "function_f1",
            "circle_c18",
            "gpu_fused_silu_1024x1024",
            "code_repair_chunk",
            "symbolic_sine_quadratic",
            "text_to_sql_analytics",
            "prompt_json_extraction",
        }
        return [variant for variant in variants if variant.task_id in canonical_ids]
    if profile == "focus":
        return variants
    raise ValueError(f"unknown workload profile: {profile}")


def seeds_for(variant: TaskVariant, profile: str) -> tuple[int, ...]:
    if profile == "focus" and variant.priority == "focus":
        return (101, 102)
    return (101,)
