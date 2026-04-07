from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


PUBLIC_DATASET_REFERENCE = {
    "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
    "dataset_owner": "UCI Machine Learning Repository",
    "dataset_reference": "AI4I 2020 Predictive Maintenance Dataset",
    "dataset_url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
    "dataset_note": (
        "This project uses a compact local telemetry sample inspired by public industrial predictive-maintenance "
        "datasets, adapted to a frac pump failure forecasting framing for deterministic execution."
    ),
}


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".csv", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".json", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _generate_sample(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pumps = [f"FP-{index:02d}" for index in range(1, 9)]
    observed_cycle_plan = [58, 64, 71, 77, 83, 88, 94, 100]
    rows: list[dict[str, object]] = []

    for pump_id, observed_windows in zip(pumps, observed_cycle_plan, strict=True):
        base_discharge_pressure = rng.uniform(8200, 9600)
        base_suction_pressure = rng.uniform(65, 95)
        base_fluid_rate = rng.uniform(68, 82)
        base_vibration = rng.uniform(1.1, 2.1)
        base_power_end_temp = rng.uniform(72, 88)
        base_lube_pressure = rng.uniform(48, 62)
        base_stroke_rate = rng.uniform(11, 14)

        for window in range(1, observed_windows + 1):
            lifecycle = window / 100
            shock_event = rng.random() < (0.015 + 0.09 * lifecycle)

            discharge_pressure = base_discharge_pressure - lifecycle * rng.uniform(500, 1120) + rng.normal(0, 55)
            suction_pressure = base_suction_pressure - lifecycle * rng.uniform(6, 15) + rng.normal(0, 0.9)
            fluid_rate = base_fluid_rate - lifecycle * rng.uniform(11, 21) + rng.normal(0, 0.65)
            vibration = base_vibration + lifecycle * rng.uniform(1.4, 3.0) + rng.normal(0, 0.04)
            power_end_temperature = base_power_end_temp + lifecycle * rng.uniform(20, 38) + rng.normal(0, 0.9)
            lube_pressure = base_lube_pressure - lifecycle * rng.uniform(9, 19) + rng.normal(0, 0.55)
            stroke_rate = base_stroke_rate + lifecycle * rng.uniform(0.9, 2.8) + rng.normal(0, 0.10)
            valve_noise_index = rng.uniform(0.8, 1.5) + lifecycle * rng.uniform(1.1, 2.7) + rng.normal(0, 0.05)

            risk_score = (
                0.22 * max(8200 - discharge_pressure, 0) / 300
                + 0.10 * max(55 - suction_pressure, 0) / 5
                + 0.14 * max(66 - fluid_rate, 0) / 4
                + 0.18 * max(vibration - 2.6, 0)
                + 0.18 * max(power_end_temperature - 100, 0) / 8
                + 0.10 * max(46 - lube_pressure, 0) / 4
                + 0.08 * max(valve_noise_index - 2.6, 0)
            )
            failure_next_window = int(shock_event or risk_score > 0.38 or lifecycle > 0.81)

            rows.append(
                {
                    "pump_id": pump_id,
                    "window_id": f"{pump_id}-{window:03d}",
                    "window_number": window,
                    "discharge_pressure": round(float(discharge_pressure), 2),
                    "suction_pressure": round(float(suction_pressure), 2),
                    "fluid_rate": round(float(fluid_rate), 2),
                    "vibration": round(float(vibration), 3),
                    "power_end_temperature": round(float(power_end_temperature), 2),
                    "lube_pressure": round(float(lube_pressure), 2),
                    "stroke_rate": round(float(stroke_rate), 3),
                    "valve_noise_index": round(float(valve_noise_index), 3),
                    "failure_next_window": failure_next_window,
                }
            )

    return pd.DataFrame(rows)


def ensure_dataset(base_dir: str | Path) -> dict[str, str]:
    base_path = Path(base_dir)
    telemetry_path = base_path / "data" / "raw" / "frac_pump_telemetry_sample.csv"
    reference_path = base_path / "data" / "raw" / "public_dataset_reference.json"

    telemetry_df = _generate_sample()
    _atomic_write_csv(telemetry_df, telemetry_path)
    _atomic_write_json(PUBLIC_DATASET_REFERENCE, reference_path)

    return {
        "telemetry_path": str(telemetry_path),
        "reference_path": str(reference_path),
    }
