
import pandas as pd


def read_txt_file(filepath, num_cols, col_names=None):
    """
    Safely read space-separated text files with a specific number of columns.
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= num_cols:
                data.append(parts[:num_cols])
    if col_names:
        return pd.DataFrame(data, columns=col_names)
    return pd.DataFrame(data[1:len(data)], columns=data[0])
