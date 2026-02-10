"""
Read data/附件.xlsx, convert 检测孕周 to decimal, save data/boy.csv and data/girl.csv.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def convert_weeks_to_decimal(week_str):
    """Convert gestational week string to decimal (weeks + days/7)."""
    if pd.isna(week_str):
        return np.nan
    week_str = str(week_str)
    match = re.match(r"(\d+)w\+(\d+)", week_str)
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2))
        return weeks + days / 7
    numbers = re.findall(r"\d+", week_str)
    if len(numbers) >= 2:
        return int(numbers[0]) + int(numbers[1]) / 7
    if len(numbers) == 1:
        return float(numbers[0])
    return np.nan


ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "data" / "附件.xlsx"
OUT_BOY = ROOT / "data" / "boy.csv"
OUT_GIRL = ROOT / "data" / "girl.csv"


def main():
    excel_data = pd.read_excel(EXCEL_PATH, sheet_name=None, engine="openpyxl")
    sheet_names = list(excel_data.keys())

    boy_df = excel_data[sheet_names[0]].copy()
    boy_df["检测孕周"] = boy_df["检测孕周"].apply(convert_weeks_to_decimal)
    boy_df.to_csv(OUT_BOY, index=False, encoding="utf-8-sig")

    girl_df = excel_data[sheet_names[1]].copy()
    girl_df["检测孕周"] = girl_df["检测孕周"].apply(convert_weeks_to_decimal)
    girl_df.to_csv(OUT_GIRL, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
