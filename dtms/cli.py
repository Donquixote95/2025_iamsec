# dtms/cli.py
import argparse
from .converter import parse_pdf_to_df, save_csv

def main():
    ap = argparse.ArgumentParser(description="DTMS-like PDF → CSV (IoU, 세로 병합셀, 여단 제외)")
    ap.add_argument("--input", required=True, help="PDF file path")
    ap.add_argument("--output", required=True, help="CSV output path")
    ap.add_argument("--brigade", type=str, default=None, help="브리게이드 명(선택)")
    args = ap.parse_args()

    df = parse_pdf_to_df(args.input, brigade=args.brigade)
    save_csv(df, args.output)
    print(f"Saved: {args.output} (rows={len(df)})")