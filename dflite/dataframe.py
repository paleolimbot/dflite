
import numpy as np
import csv
import decimal


def _len(obj):
    if type(obj) in (list, tuple, np.ndarray, set):
        return len(obj)
    else:
        return 1


def _aslist(obj):
    if type(obj) in (list, tuple, np.ndarray, set):
        return obj
    else:
        return [obj, ]


def _asnumeric(obj):
    if type(obj) in (int, float, np.int, np.int64, np.int8, np.int16, np.int32, decimal.Decimal):
        return obj
    try:
        return int(obj)
    except ValueError:
        pass

    try:
        return float(obj)
    except ValueError:
        pass

    return obj


class _DFRow(dict):

    def __init__(self, columns, vals):
        super().__init__(zip(columns, vals))
        self._keys = tuple(columns)

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        else:
            try:
                return super().__getitem__(self._keys[item])
            except IndexError:
                raise KeyError("So such key in row")

    def __iter__(self):
        for key in self._keys:
            yield self[key]

    def _repr_html_(self):
        title = "".join(["<td><strong>%s</strong></td>" % key for key in self._keys])
        vals = "".join(["<td>%s</td>" % self[key] for key in self._keys])
        return "<table><tr>%s</tr><tr>%s</tr></table>" % (title, vals)

    def keys(self):
        return self._keys

    def items(self):
        for key in self._keys:
            yield key, self[key]


class _Iloc(object):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, item):
        if type(item) == int:
            return self.df._row(item)
        elif type(item) != tuple:
            return self.df._subset(item, self.df.columns)
        elif len(item) == 2 and type(item[0]) == int and type(item[1]) == int:
            return self.df[item[1]][item[0]]  # single value
        else:
            return self.df._subset(*item)

    def __setitem__(self, key, value):
        raise NotImplementedError("Setting is not available for this class (use pandas DataFrame instead)")


class DataFrame(object):
    """
    A lightweight copy of the Pandas DataFrame class (http://pandas.pydata.org/pandas-docs/stable/dsintro.html).
    All basic operations are supported such as selecting columns by indexing (e.g. df['column1']) or attribute
    (e.g. df.column1). Columns are stored as numpy.ndarray objects, so indexing of columns according to
    anything in numpy is supported.
    """

    def __init__(self, *args, columns=None, **kwargs):
        """
        Instantiate a DataFrame (also can do this from read_csv()).

        :param args: Iterables of values by column.
        :param columns Column names for args
        :param kwargs: Key/value parameters
        :return:
        """
        self.__rows = None
        if columns is not None:
            columns = list(columns)
            if len(args) == 0:
                # empty data frame
                args = [[] for i in range(len(columns))]
            if len(columns) != len(args):
                raise ValueError("Length of columns must be equal to list of names: %s, %s" % (columns,
                                                                                               len(args)))
        else:
            columns = [self.__default_arg_name(i) for i in range(len(args))]

        self.__keynames = []
        for i in range(len(args)):
            self.__setitem__(columns[i], args[i])
        for key, value in kwargs.items():
            self.__setitem__(key, value)

        self.loc = self.iloc = _Iloc(self)

    @staticmethod
    def __default_arg_name(index):
        return "V%02d" % index

    def copy(self):
        """
        :return: A shallow copy of the DataFrame (underlying ndarras are not copied)
        """
        return DataFrame(*[self[col] for col in self], columns=self.__keynames)

    def ncol(self):
        """
        :return: The number of columns. Probably shouldn't use this as it isn't a part of pandas.DataFrame
        """
        return len(self.__keynames)

    def _subset(self, rows, cols):
        if type(cols) == slice:
            icols = [self. __internal_key(col) for col in list(range(len(self.__keynames)))[cols]]
        elif type(cols) == int:
            icols = [self.__internal_key(cols), ]
        else:
            icols = [self.__internal_key(col) for col in cols]
        if None in icols:
            raise KeyError("One of the following is not a valid column: %s" % (cols, ))
        if type(rows) == tuple:
            rows = list(rows)
        return DataFrame(*[_aslist(self[col][rows]) for col in icols], columns=icols)

    def _row(self, i):
        return _DFRow(self.__keynames, [self[col][i] for col in self])

    def itertuples(self, header=False, rownames=True):
        """
        An iterator that iterates over rows.

        :param header: True if the header should be inclued in the iterations.
        :param rownames: True if rownames should be inclued in the rows returned.
        :return: Each row is basically an ordered dictionary that can be indexed by name or position.
        """
        if header:
            if rownames:
                yield [-1, ] + self.__keynames
            else:
                yield list(self.__keynames)
        if rownames:
            for i in range(len(self)):
                yield _DFRow(["_index", ] + self.__keynames, [i, ] + [self[col][i] for col in self])
        else:
            for i in range(len(self)):
                yield _DFRow(self.__keynames, [self[col][i] for col in self])

    def __iter__(self):
        return iter(self.__keynames)

    def __reversed__(self):
        return reversed(self.__keynames)

    def __contains__(self, item):
        return item in self.__keynames

    def __len__(self):
        return 0 if self.__rows is None else self.__rows

    def __internal_key(self, keyin):
        if keyin in self.__keynames:
            return keyin
        else:
            try:
                if keyin < len(self.__keynames):
                    return self.__keynames[keyin]
            except (TypeError, IndexError):
                return None

    def __delitem__(self, orig):
        key = self.__internal_key(orig)
        if key is None:
            raise KeyError("No such column: %s" % orig)
        del self.__dict__[key]
        if key in self.__keynames:
            self.__keynames.remove(key)

    def __setitem__(self, orig, value, check=True):
        key = self.__internal_key(orig)
        if key is None:
            key = orig
        if self.__rows is None:
            self.__rows = len(value)
        else:
            if _len(value) == 1:
                value = np.repeat(value, self.__rows)
            if self.__rows != len(value) and check:
                raise ValueError("Number of observations is not consistent (%s, %s) for arg %s" %
                                 (self.__rows, len(value), key))

        self.__dict__[key] = np.array(value)
        if key not in self.__keynames:
            self.__keynames.append(key)

    def __setattr__(self, key, value):
        if key == "columns":
            if len(value) != len(self.__keynames):
                raise ValueError("Dimension mismatch: cannot set names of more columns than exist.")
            # rename columns
            for oldcol, newcol in zip(list(self.__keynames), value):
                if oldcol != newcol:
                    self[newcol] = self[oldcol]
                    del self[oldcol]
        else:
            super().__setattr__(key, value)

    def __getattr__(self, item):
        if item == "columns":
            return list(self.__keynames)
        if item in self.__keynames:
            key = self.__internal_key(item)
            if key is None:
                raise KeyError("No such column: %s" % item)
            return self.__dict__[key]
        else:
            return super().__getattribute__(item)

    def __getitem__(self, item):
        key = self.__internal_key(item)
        if key is None:
            raise KeyError("No such column: %s" % item)
        return self.__dict__[key]

    def append(self, *args, **kwargs):
        """
        Append a row to the data frame. Must have same number of cols as the DataFrame. Technically
        more than one value can be appended at a time, but this works poorly if multidimentional types
        are to be appended (e.g. tuple objects).

        :param args: A list of values added by column name.
        :param kwargs: Values to be added by column name.
        """

        if (len(args) + len(kwargs)) != self.ncol():
            raise ValueError("Dimension mismatch: %s cols found and %s expected" % (len(args)+len(kwargs), self.ncol()))

        # check input (just warning for now because lists are added sometimes in dbinterface)
        lengths = [_len(arg) for arg in args] + [_len(value) for value in kwargs.values()]
        newrows = min(lengths)
        if any([length != newrows for length in lengths]):
            raise ValueError("possible dimension mismatch in DataFrame.append()")

        # going to use
        for i in range(len(args)):
            key = self.__internal_key(i)
            if key is None:
                raise KeyError("No such column: ", i)
            self.__dict__[key] = np.append(self[key], args[i])
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError("No such column: ", key)
            self.__dict__[key] = np.append(self[key], value)
        self.__rows += newrows

    def __bool__(self):
        return self.__rows is not None and self.__rows > 0

    def __repr__(self, sep="\t"):
        return "\n".join(sep.join(str(cell) for cell in row) for row in self.itertuples(header=True, rownames=False))

    def head(self, nrow=6):
        """
        Return the first nrows of the dataframe as a new DataFrame.

        :param nrow: The number of rows to return.
        :return: The first nrows of the dataframe as a new DataFrame.
        """
        return self.iloc[:nrow, :]

    def tail(self, nrow=6):
        """
        Return the last nrows of the dataframe as a new DataFrame.

        :param nrow: The number of rows to return.
        :return: The last nrows of the dataframe as a new DataFrame.
        """
        return self.iloc[(len(self)-nrow):len(self), :]

    def _repr_html_(self):
        """
        Jupyter Notebook magic repr function.
        """
        head = '<tr><td></td>%s</tr>\n' % ''.join(['<td><strong>%s</strong></td>' % c for c in self.__keynames])
        rows = ['<td><strong>%d</strong></td>' % i + ''.join(['<td>%s</td>' % c for c in row])
                for i, row in enumerate(self.itertuples(rownames=False, header=False))]
        html = '<table>{}</table>'.format(head + '\n'.join(['<tr>%s</tr>' % row for row in rows]))
        return html

    def to_csv(self, writer, driver=None, mode="w"):
        """
        Write this object to writer using driver.

        :param writer: A file handle or filename.
        :param driver: Either 'csv' or 'tsv', automatically determined by extension if None
        :param mode: 'w' for write, 'a' for append. using 'a' will omit printing headers.
        """
        fname = None
        if "write" in dir(writer):
            # is an open file object
            if driver is None:
                raise ValueError("Must specify driver of 'csv', 'tsv', or 'json'")
        else:
            # is a filename
            if not isinstance(writer, str):
                raise ValueError("Writer paramter is not an open file or a filename")
            fname = writer
            if driver is None:
                # autodetect from ext
                ext = writer[fname.rfind("."):]
                if len(ext) > 1:
                    driver = ext[1:]
                else:
                    raise ValueError("Invalid extension provided, cannot autodetect out driver")
            writer = open(fname, mode)

        headers = mode == "w"
        if driver == "csv":
            for row in self.itertuples(header=headers, rownames=False):
                cells = [str(cell) for cell in row]
                for i, cell in enumerate(cells):
                    if "," in cell:
                        cells[i] = '"%s"' % cell
                writer.write(",".join(cells))
                writer.write("\n")
        elif driver == "tsv":
            for row in self.itertuples(header=headers, rownames=False):
                writer.write("\t".join([str(cell) for cell in row]))
                writer.write("\n")
        elif driver == "json":
            if fname:
                writer.close()
            raise NotImplementedError("JSON driver not yet implemented")
        else:
            if fname:
                writer.close()
            raise ValueError("Unrecgonized driver: %s" % driver)

        if fname:
            writer.close()

    @staticmethod
    def from_records(data, columns=None):
        """
        Creates a DataFrame from an iterable grouped by row.

        :param data: An interable grouped by row.
        :param columns: Column names for data.
        :return: A DataFrame with the requested data.
        """
        return DataFrame(*zip(*data)) if columns is None else DataFrame(*zip(*data), columns=columns)

    @staticmethod
    def from_dict_list(list_of_dicts, no_value=float('nan'), keys=None):
        """
        Flattens a list of dict objects into a DataFrame.

        :param list_of_dicts: What it sounds like.
        :param no_value: The value to use if not all dicts have the same keys.
        :param keys: Use only these keys to create the data frame.
        :return: A DataFrame with the resulting data.
        """
        if keys is None:
            keys = set()
            for d in list_of_dicts:
                keys = keys.union(set(d.keys()))
        df = DataFrame()
        for key in keys:
            df[key] = [d[key] if key in d else no_value for d in list_of_dicts]
        return df


def read_csv(reader, driver=None, headers=True, skiprows=0, numeric=True):
    """
    Reads a file in as a DataFrame.

    :param reader: A file handle or filename.
    :param driver: The driver with which to read the file, or None for auto by extension. Currently only csv works.
    :param headers: True if headers should be written, false otherwise.
    :param skiprows: Skip this number of rows before reading data.
    :param numeric: True if data should be converted to numeric (if possible).
    :return: A DataFrame with the resulting data.
    """
    fname = None
    if "readline" in dir(reader):
        # is an open file object
        if driver is None:
            raise ValueError("Must specify driver of 'csv', 'tsv', or 'json'")
    else:
        # is a filename
        if not isinstance(reader, str):
            raise ValueError("Reader paramter is not an open file or a filename")
        fname = reader
        if driver is None:
            # autodetect from ext
            ext = reader[fname.rfind("."):]
            if len(ext) > 1:
                driver = ext[1:]
            else:
                raise ValueError("Invalid extension provided, cannot autodetect out driver")
        reader = open(fname, "r")

    if driver != "csv":
        if fname:
            reader.close()
        raise NotImplementedError("Driver other than csv is not yet supported.")

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
                if headers:
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

    return DataFrame.from_records(records) if columns is None else DataFrame.from_records(records, columns=columns)
