from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "Crop_production.csv"
DEFAULT_OUTPUT_DIR = "data/cleaned"
TARGET_CANDIDATES = [
    "Yield_ton_per_hec",
    "Production_in_tons",
    "yield",
    "production",
    "target",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze, clean, filter, and structure a CSV dataset for ML training."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV file path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for cleaned outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    unnamed_columns = [column for column in df.columns if str(column).lower().startswith("unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        normalized = (
            df[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
        df[column] = normalized
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce")
            continue

        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() and converted.notna().sum() >= len(df[column]) * 0.9:
            df[column] = converted
    return df


def detect_target_column(df: pd.DataFrame) -> str | None:
    columns = set(df.columns)
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def build_analysis_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    target_column: str | None,
) -> dict:
    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_summary = {}
    if numeric_columns:
        summary_frame = cleaned_df[numeric_columns].describe().transpose().round(4)
        numeric_summary = summary_frame.to_dict(orient="index")

    return {
        "original_shape": {
            "rows": int(original_df.shape[0]),
            "columns": int(original_df.shape[1]),
        },
        "cleaned_shape": {
            "rows": int(cleaned_df.shape[0]),
            "columns": int(cleaned_df.shape[1]),
        },
        "dropped_rows": int(original_df.shape[0] - cleaned_df.shape[0]),
        "dropped_columns": sorted(set(original_df.columns) - set(cleaned_df.columns)),
        "target_column": target_column,
        "missing_values_after_cleaning": {
            column: int(value) for column, value in cleaned_df.isna().sum().items()
        },
        "duplicate_rows_after_cleaning": int(cleaned_df.duplicated().sum()),
        "data_types": {column: str(dtype) for column, dtype in cleaned_df.dtypes.items()},
        "numeric_summary": numeric_summary,
    }


def fill_missing_values(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if column == target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode = df[column].mode(dropna=True)
            fallback = mode.iloc[0] if not mode.empty else "unknown"
            df[column] = df[column].fillna(fallback)
    return df


def filter_invalid_rows(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    df = df.copy()

    if target_column and target_column in df.columns:
        df = df[df[target_column].notna()]

    non_negative_keywords = ("rain", "area", "production", "yield", "n", "p", "k")
    for column in df.select_dtypes(include=["number"]).columns:
        lower_name = column.lower()
        if lower_name == "ph":
            df = df[df[column].between(0, 14, inclusive="both")]
            continue

        if "temperature" in lower_name:
            df = df[df[column].between(-20, 60, inclusive="both")]
            continue

        if any(keyword in lower_name for keyword in non_negative_keywords):
            df = df[df[column] >= 0]

    return df


def cap_outliers(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if column == target_column:
            continue

        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
        if pd.isna(lower_bound) or pd.isna(upper_bound) or lower_bound == upper_bound:
            continue
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df


def reorder_columns(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    if not target_column or target_column not in df.columns:
        return df

    feature_columns = [column for column in df.columns if column != target_column]
    return df[feature_columns + [target_column]]


def make_ml_ready(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    df = reorder_columns(df, target_column)

    if target_column and target_column in df.columns:
        feature_frame = pd.get_dummies(
            df.drop(columns=[target_column]),
            drop_first=False,
            dtype=int,
        )
        ml_ready = pd.concat([feature_frame, df[[target_column]].reset_index(drop=True)], axis=1)
        return ml_ready

    return pd.get_dummies(df, drop_first=False, dtype=int)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(input_path)
    cleaned_df = sanitize_columns(original_df)
    cleaned_df = normalize_text_columns(cleaned_df)
    cleaned_df = convert_numeric_columns(cleaned_df)
    target_column = detect_target_column(cleaned_df)

    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    cleaned_df = fill_missing_values(cleaned_df, target_column)
    cleaned_df = filter_invalid_rows(cleaned_df, target_column)
    cleaned_df = cap_outliers(cleaned_df, target_column)
    cleaned_df = cleaned_df.drop_duplicates()
    cleaned_df = reorder_columns(cleaned_df, target_column).reset_index(drop=True)

    ml_ready_df = make_ml_ready(cleaned_df, target_column)
    analysis_report = build_analysis_report(original_df, cleaned_df, target_column)

    cleaned_output_path = output_dir / "cleaned_data.csv"
    ml_ready_output_path = output_dir / "ml_ready_data.csv"
    report_output_path = output_dir / "data_analysis_report.json"

    cleaned_df.to_csv(cleaned_output_path, index=False)
    ml_ready_df.to_csv(ml_ready_output_path, index=False)
    report_output_path.write_text(json.dumps(analysis_report, indent=2), encoding="utf-8")

    print(f"Input file: {input_path}")
    print(f"Detected target column: {target_column or 'None'}")
    print(f"Cleaned dataset saved to: {cleaned_output_path}")
    print(f"ML-ready dataset saved to: {ml_ready_output_path}")
    print(f"Analysis report saved to: {report_output_path}")
    print(f"Final cleaned shape: {cleaned_df.shape}")
    print(f"Final ML-ready shape: {ml_ready_df.shape}")


if __name__ == "__main__":
    main()
