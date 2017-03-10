
import csv

from .dataframe import DataFrame
from .na import NA


def _asnumeric(obj):
    try:
        return int(obj)
    except ValueError:
        pass

    try:
        return float(obj)
    except ValueError:
        pass

    if obj in ("True", "true", "TRUE"):
        return True
    elif obj in ("False", "false", "FALSE"):
        return False
    elif obj == "":
        return NA
    else:
        return obj


def read_csv(reader, header=True, skiprows=0, numeric=True):
    """
    Reads a file in as a DataFrame.

    :param reader: A file handle or filename.
    :param headers: True if headers are on the first line of data, false otherwise.
    :param skiprows: Skip this number of rows before reading data.
    :param numeric: True if data should be converted to numeric (if possible).
    :return: A DataFrame with the resulting data.
    """
    fname = None
    if "readline" not in dir(reader):
        # is a filename
        if not isinstance(reader, str):
            raise ValueError("Reader parameter is not an open file or a filename")
        fname = reader
        reader = open(fname, "r")

    csvreader = csv.reader(reader)
    records = []
    columns = None
    for line in csvreader:
        if skiprows > 0:
            skiprows -= 1
            continue
        if not records and not columns:
            # look for data
            if any([bool(c) for c in line]):
                if header:
                    columns = line
                else:
                    if numeric:
                        records.append([_asnumeric(c) for c in line])
                    else:
                        records.append(line)
            else:
                # no data
                continue
        else:
            # already a df
            if numeric:
                records.append([_asnumeric(c) for c in line])
            else:
                records.append(line)
    if fname:
        reader.close()

    return DataFrame(records) if columns is None else DataFrame(records, columns=columns)
