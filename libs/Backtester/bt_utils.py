import numpy as np
import pandas as pd
import os
import re

# a file for tidbits and useful code.

def filter_columns(col_names, columns=None, name_like=None, name_notlike=None):
    # Filter out columns named 'Date' or similar
    col_names = [col for col in col_names if col.lower() not in ['date', 'dates']]

    # Filter by specific column names if provided
    if columns:
        col_names = [col for col in col_names if col in columns]

    # Apply name_like patterns if provided
    if name_like:
        if isinstance(name_like, str):
            name_like = [name_like]
        for pattern in name_like:
            col_names = [col for col in col_names if re.search(pattern, col)]

    # Apply name_notlike patterns if provided
    if name_notlike:
        if isinstance(name_notlike, str):
            name_notlike = [name_notlike]
        for pattern in name_notlike:
            col_names = [col for col in col_names if not re.search(pattern, col)]

    return col_names

def getSigNames(directory, assets=None, columns=None, name_like=None, name_notlike=None):
    """
    Extract columns from a CSV file based on regex matching. If no assets are specified, finds the first CSV file
    matching '_signals.csv' in the directory, then lists all column names except "Date".

    Parameters:
    - directory (str): Path to the directory containing the CSV files.
    - assets (list or None): List of asset names or None to select the first matching file.
    - columns (list or None): Specific column names to extract from the CSV file.
    - name_like (list or str or None): Regex pattern(s) to match column names to include.
    - name_notlike (list or str or None): Regex pattern(s) to match column names to exclude.

    Returns:
    - Dictionary of the DataFrame or list of column names, excluding "Date".
    """

    results = {}

    def process_file(file_path):
        try:
            data = pd.read_csv(file_path)
            col_names = data.columns.tolist()
            filtered_columns = filter_columns(col_names, columns, name_like, name_notlike)
            return filtered_columns if not columns else data[filtered_columns]
        except Exception as e:
            print(f"Failed to read or process file {file_path}. Error: {e}")
            return []

    if assets is None:
        # Scan the directory for the first file matching '_signals.csv'
        for file in os.listdir(directory):
            if re.search(r'_signals\.csv$', file):
                file_path = os.path.join(directory, file)
                asset_name = os.path.basename(file_path).split('_')[0]
                results[asset_name] = process_file(file_path)
                return results[asset_name]

    if not assets:
        print("No suitable files found.")
        return results

    for asset_name in assets:
        file_path = os.path.join(directory, f"{asset_name}_signals.csv")
        if not os.path.exists(file_path):
            print(f"No file found for asset: {asset_name}")
            continue

        results[asset_name] = process_file(file_path)

    return results

def getSigNames1(directory, assets=None, columns=None, name_like=None, name_notlike=None):
    """
    Extract columns from a CSV file based on regex matching. If no assets are specified, finds the first CSV file
    matching '_signals.csv' in the directory, then lists all column names except "Date".

    Parameters:
    - directory (str): Path to the directory containing the CSV files.
    - assets (list or None): List of asset names or None to select the first matching file.
    - columns (list or None): Specific column names to extract from the CSV file.
    - name_like (str or None): Regex pattern to match column names to include.
    - name_notlike (str or None): Regex pattern to match column names to exclude.

    Returns:
    - Dictionary of the DataFrame or list of column names, excluding "Date".
    """

    results = {}
    if assets is None:
        # Scan the directory for the first file matching '_signals.csv'
        for file in os.listdir(directory):
            if re.search(r'_signals\.csv$', file):
                file_path = os.path.join(directory, file)
                asset_name = os.path.basename(file_path).split('_')[0]
                if os.path.exists(file_path):
                    try:
                        # Read the CSV file
                        data = pd.read_csv(file_path)
                        # Filter out columns named 'Date' or similar
                        col_names = [col for col in data.columns if 'date' not in col.lower()]
                        if columns:
                            col_names = [col for col in col_names if col in columns]
                        if name_like:
                            col_names = [col for col in col_names if re.search(name_like, col)]
                        if name_notlike:
                            col_names = [col for col in col_names if not re.search(name_notlike, col)]

                        results[asset_name] = data[col_names] if columns else col_names
                        return col_names
                    except Exception as e:
                        print(f"Failed to read file {file}. Error: {e}")
                        return []

    if not assets:
        print("No suitable files found.")
        return results

    for asset_name in assets:
        file_path = os.path.join(directory, f"{asset_name}_signals.csv")
        if not os.path.exists(file_path):
            print(f"No file found for asset: {asset_name}")
            continue

        try:
            data = pd.read_csv(file_path)
            col_names = data.columns.tolist()
            # Filter out columns named 'Date' or similar
            col_names = [col for col in col_names if col.lower() not in ['date', 'dates']]

            if columns:
                col_names = [col for col in col_names if col in columns]
            if name_like:
                col_names = [col for col in col_names if re.search(name_like, col)]
            if name_notlike:
                col_names = [col for col in col_names if not re.search(name_notlike, col)]

            results[asset_name] = data[col_names] if columns else col_names
        except Exception as e:
            print(f"Failed to read or process file for asset {asset_name}. Error: {e}")

    return results

def getSigs(directory, asset_names, columns):
    """
    Load specified columns from CSV files for given assets in a specified directory.

    Parameters:
    - directory (str): Path to the directory containing the CSV files.
    - asset_names (str or list): Name(s) of the asset(s) whose data is to be loaded.
    - columns (dict of aggnames and strats): dict of columns to extract from the asset's CSV file.

    Returns:
    - Dictionary with asset names as keys and DataFrames as values containing the specified columns.
    - If a file or columns do not exist for an asset, appropriate messages are printed.
    """

    if isinstance(asset_names, str):
        asset_names = [asset_names]  # Convert to list if only one asset name is provided
    if isinstance(columns, str):
        columns = [columns]  # Convert to list if only one asset name is provided
    results = {}
    for asset_name in asset_names:
        file_path = os.path.join(directory, f"{asset_name}_signals.csv")
        if not os.path.exists(file_path):
            print(f"No file found for asset: {asset_name}")
            continue  # Skip to the next asset
        try:
            data = pd.read_csv(file_path)
            # col = get the names of the strats from the dict, the name is the 1st element of the tuple value for each key value pair.
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                print(f"Missing columns {missing_columns} in file for asset: {asset_name}")
                continue  # Skip to the next asset

            query_cols = ['Date'] + [col for col in columns if col != 'Date']  # Avoid duplicating 'Date'
            data = data[query_cols]
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            results[asset_name] = data
        except Exception as e:
            print(f"Failed to read or process file for asset {asset_name}. Error: {e}")

    return results

