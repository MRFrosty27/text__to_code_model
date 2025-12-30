#!/usr/bin/env python3
import os
import csv
import argparse
import sys


def add_txt_files_to_csv(csv_path, txt_files, headers=None, overwrite=False):
    """
    Add contents of multiple TXT files as a single row in a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        txt_files (list): List of paths to TXT files
        headers (list, optional): Custom column headers
        overwrite (bool): If True, overwrite existing CSV instead of appending
    """
    if len(txt_files) < 2:
        raise ValueError("At least 2 TXT files are required.")

    # Read contents of TXT files
    row = []
    for txt_path in txt_files:
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            row.append(content)

    file_exists = os.path.exists(csv_path)
    mode = 'w' if (not file_exists or overwrite) else 'a'

    with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # If we're writing a new file (or overwriting), write headers first
        if mode == 'w':
            if headers is None:
                headers = [f'feature{i+1}' for i in range(len(txt_files))]
            elif len(headers) != len(txt_files):
                raise ValueError(f"Number of headers ({len(headers)}) must match number of TXT files ({len(txt_files)}).")
            writer.writerow(headers)

        # If appending, validate column count matches existing CSV
        elif mode == 'a' and file_exists:
            # Rewind to check header
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            existing_header = next(reader, None)
            if existing_header is None:
                raise ValueError("Existing CSV is empty.")
            if len(existing_header) != len(txt_files):
                raise ValueError(f"Number of TXT files ({len(txt_files)}) does not match existing CSV columns ({len(existing_header)}).")

        # Write the data row
        writer.writerow(row)

    print(f"Successfully {'created' if mode == 'w' else 'appended to'} {csv_path} with {len(txt_files)} columns.")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple TXT files into a single row in a CSV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "txt_files",
        nargs="+",
        help="Paths to at least 2 TXT files (their contents will become columns)"
    )
    
    parser.add_argument(
        "-c", "--csv",
        default="dataset.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "-H", "--headers",
        nargs="+",
        default=None,
        help="Custom column headers (must match number of TXT files). Only used when creating a new CSV."
    )
    
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite the CSV file if it already exists (instead of appending)"
    )

    args = parser.parse_args()

    if len(args.txt_files) < 2:
        print("Error: At least 2 TXT files are required.", file=sys.stderr)
        sys.exit(1)

    try:
        add_txt_files_to_csv(
            csv_path=args.csv,
            txt_files=args.txt_files,
            headers=args.headers,
            overwrite=args.overwrite
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()