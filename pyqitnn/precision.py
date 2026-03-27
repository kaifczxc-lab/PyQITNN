from __future__ import annotations

_PRECISION_MODE_ALIASES = {
    "mixed_bf16_native": "qts_fp32_rest_bf16",
    "bf16": "qts_fp32_rest_bf16",
}
_PRECISION_MODE_SET = {"fp32", "qts_fp32_rest_bf16"}


def precision_mode_choices() -> tuple[str, ...]:
    return tuple(sorted(_PRECISION_MODE_SET))


def normalize_precision_mode(value: str) -> str:
    mode = value.strip().lower()
    mode = _PRECISION_MODE_ALIASES.get(mode, mode)
    if mode not in _PRECISION_MODE_SET:
        wanted = ", ".join(sorted(_PRECISION_MODE_SET))
        raise RuntimeError(f"precision_mode must be one of: {wanted}")
    return mode


def resolve_precision_mode(
    precision_mode: str | None,
    mixed_precision: bool | None,
    *,
    default_mode: str = "fp32",
) -> tuple[str, bool]:
    mode = normalize_precision_mode(default_mode)
    if precision_mode is not None:
        mode = normalize_precision_mode(str(precision_mode))

    if mixed_precision is not None:
        legacy_mode = "qts_fp32_rest_bf16" if bool(mixed_precision) else "fp32"
        if precision_mode is None:
            mode = legacy_mode
        elif legacy_mode != mode:
            raise RuntimeError(
                "mixed_precision and precision_mode conflict. "
                "Use precision_mode alone, or keep them aligned during migration."
            )

    return mode, mode != "fp32"
