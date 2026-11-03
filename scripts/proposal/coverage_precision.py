"""How precisely a test of size M states coverage; writes outputs/coverage_precision.json."""

import json
import pathlib

from scipy import stats

NOMINAL = 0.90
ALPHA = 0.05
TARGET_POWER = 0.80


def clopper_pearson(k: int, m: int) -> tuple[float, float]:
    low = 0.0 if k == 0 else stats.beta.ppf(ALPHA / 2, k, m - k + 1)
    high = 1.0 if k == m else stats.beta.ppf(1 - ALPHA / 2, k + 1, m - k)
    return float(low), float(high)


def power(m: int, true_coverage: float) -> float:
    """One-sided exact test of H0: coverage >= nominal, at level ALPHA."""
    critical = int(stats.binom.ppf(ALPHA, m, NOMINAL))
    if stats.binom.cdf(critical, m, NOMINAL) > ALPHA:
        critical -= 1
    return float(stats.binom.cdf(critical, m, true_coverage)) if critical >= 0 else 0.0


def smallest_m(true_coverage: float) -> int:
    """First M after which power stays above target: binomial power oscillates with M."""
    return next(
        m
        for m in range(5, 2000)
        if all(power(m + extra, true_coverage) >= TARGET_POWER for extra in range(30))
    )


def realized_coverage(n: int) -> dict[str, float]:
    """Mean and sd of split-conformal coverage over calibration draws, a Beta law for n units."""
    margin = int((n + 1) * (1 - NOMINAL))
    law = stats.beta(n + 1 - margin, margin)
    return {"mean": round(float(law.mean()), 3), "sd": round(float(law.std()), 3)}


def main() -> None:
    assert abs(clopper_pearson(45, 50)[0] - 0.782) < 0.001  # value from the published beta form
    report = {
        "ci95_at_picp_090": {
            m: [round(v, 3) for v in clopper_pearson(round(NOMINAL * m), m)]
            for m in (12, 24, 54, 75, 100, 200)
        },
        "m_for_80pct_power": {t: smallest_m(t) for t in (0.85, 0.80, 0.75, 0.70, 0.60)},
        "power_by_m": {
            m: {t: round(power(m, t), 3) for t in (0.85, 0.80, 0.70, 0.60)} for m in (12, 24, 75)
        },
        "realized_coverage_by_calibration_size": {n: realized_coverage(n) for n in (25, 50, 100)},
    }
    pathlib.Path("outputs").mkdir(exist_ok=True)
    text = json.dumps(report, indent=2)
    pathlib.Path("outputs/coverage_precision.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
