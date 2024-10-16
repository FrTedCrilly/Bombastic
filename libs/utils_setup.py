from datetime import datetime, timedelta
import os
import holidays
import pandas as pd

def get_last_business_day():
    """
    Returns the last business day before today considering weekends and UK public holidays,
    formatted as dd-mm-yyyy.
    """
    uk_holidays = holidays.UnitedKingdom()
    today = datetime.now()
    # Subtract one day at a time until a business day is found
    last_business_day = today - timedelta(days=1)

    while last_business_day.weekday() in (5, 6) or last_business_day in uk_holidays:
        last_business_day -= timedelta(days=1)

    return last_business_day.strftime('%Y-%m-%d')


def create_unique_date_folder(base_dir, folder_name, appednDate = True):
    # Append the current date to the folder name
    if appednDate:
        date_str = datetime.now().strftime('%Y-%m-%d')
        full_folder_name = f"{folder_name}_{date_str}"
    else:
        full_folder_name = f"{folder_name}"
    full_path = os.path.join(base_dir, folder_name, full_folder_name)

    # Initialize the suffix and check for existing folder
    suffix = 0
    original_full_path = full_path  # Keep the original path without suffix for checking

    while os.path.exists(full_path):
        # If the folder exists, increment the suffix and update the path
        suffix += 1
        full_path = f"{original_full_path}_{suffix}"

    # Create the folder with the unique name
    os.makedirs(full_path)
    print(f"Folder created: {full_path}")
    return full_path


def create_log_folder(directory):
    """
    Create a 'log' folder within the specified directory.

    Parameters:
    - directory: The path to the directory where the log folder will be created.

    Returns:
    - The path to the newly created log folder.
    """
    # Construct the path to the new log directory
    log_dir = os.path.join(directory, 'log')

    # Create the log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory at: {log_dir}")
    else:
        print(f"Log directory already exists at: {log_dir}")

    return log_dir
def encode_rle(data):
    """
    Encode a list of 0s and 1s using Run-Length Encoding (RLE).

    Parameters:
    - data: The input list of 0s and 1s

    Returns:
    - A list of tuples where the first element is the number (0 or 1) and the second element is the count of repetitions.
    """
    if not data:
        return []

    encoded_data = []
    current_value = data[0]
    count = 1

    for value in data[1:]:
        if value == current_value:
            count += 1
        else:
            encoded_data.append((current_value, count))
            current_value = value
            count = 1
    encoded_data.append((current_value, count))

    return encoded_data


def modDF(df1, df2, operation='div'):
    """
    Operates on two time series DataFrames or Series by aligning their indices and performing the specified arithmetic operation.

    Parameters:
        df1 (pd.DataFrame or pd.Series): First time series DataFrame or Series.
        df2 (pd.DataFrame or pd.Series): Second time series DataFrame or Series.
        operation (str): The operation to perform: 'add', 'subtract', 'multiply', 'divide'.

    Returns:
        pd.DataFrame: A DataFrame with the result of the operation.
    """
    # Normalize the operation input
    operation_map = {
        'add': ['add', 'a', 'plus', 'sum', '+'],
        'subtract': ['subtract', 'sub', 's', 'minus', '-'],
        'multiply': ['multiply', 'mult', 'm', 'times', '*'],
        'divide': ['divide', 'div', 'd', '/', 'division']
    }

    # Find the correct operation based on the input
    normalized_operation = next((key for key, values in operation_map.items() if operation.lower() in values), None)
    if normalized_operation is None:
        raise ValueError("Unsupported operation. Choose 'add', 'subtract', 'multiply', or 'divide'.")

    # Check if df1 or df2 is a Series and convert to DataFrame if necessary
    if isinstance(df1, pd.Series):
        df1 = df1.to_frame(name='Value')
    else:
        df1 = df1.rename(columns={df1.columns[0]: 'Value'})

    if isinstance(df2, pd.Series):
        df2 = df2.to_frame(name='Value')
    else:
        df2 = df2.rename(columns={df2.columns[0]: 'Value'})

    # Align the DataFrames on their index
    df1_aligned, df2_aligned = df1.align(df2, join='outer', axis=0)

    # Perform the operation
    if normalized_operation == 'add':
        result = df1_aligned['Value'] + df2_aligned['Value']
    elif normalized_operation == 'subtract':
        result = df1_aligned['Value'] - df2_aligned['Value']
    elif normalized_operation == 'multiply':
        result = df1_aligned['Value'] * df2_aligned['Value']
    elif normalized_operation == 'divide':
        result = df1_aligned['Value'] / df2_aligned['Value']

    return pd.DataFrame(result)
def saveData(encoded_data, file_name):
    """
    Save the encoded data to a CSV file.

    Parameters:
    - encoded_data: The encoded data to save.
    - file_name: The name of the file to save the data to.
    """
    import csv

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in encoded_data:
            writer.writerow(item)








