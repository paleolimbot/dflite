
import itertools

import numpy as np
from .na import NA


class DataFrame(object):

    def __init__(self, data, index=None, columns=None, copy=False, dtype=None):
        if index is not None:
            raise ValueError("Row index for DataFrame objects is not supported")
        if index is not None:
            raise ValueError("Argument 'dtype' is not supported")

        # save copying preference
        if copy not in (True, False):
            raise ValueError("Argument 'copy' must be True or False")
        self._copy = copy

        # process input data
        self._data = {}

        if isinstance(data, dict) or type(data).__name__ == "DataFrame":
            # process by iterating through columns, names by iteration through the object
            if columns is None:
                columns = [x for x in data]

            self.columns = tuple(columns)

            # check column lengths
            lens = set(len(data[col]) if np.ndim(data[col]) != 0 else -1 for col in data)
            lens = list(lens.difference((-1, )))
            if len(lens) == 0:
                raise ValueError("Passing all scalars isn't allowed in Pandas, so it's not allowed here either...")
            elif len(lens) != 1:
                raise ValueError("Arguments imply differing numbers of rows: " +
                                 ", ".join([str(l) for l in lens]))
            self._nrow = lens[0]

            for col in self.columns:
                val = data[col]
                if np.ndim(val) == 0:
                    self._data[col] = np.repeat(val, self._nrow)
                elif not isinstance(val, np.ndarray) or copy:
                    self._data[col] = np.array(val)
                else:
                    self._data[col] = val

        elif isinstance(data, list) or isinstance(data, tuple) or isinstance(data, np.ndarray):
            # process by iterating as rows, names are integers or from columns input
            self._nrow = len(data)

            lens = list(set(len(x) for x in data))
            if len(lens) != 1 or lens[0] == 0:
                raise ValueError("Data argument implies differing numbers of columns: " +
                                 ", ".join([str(l) for l in lens]))

            self.columns = tuple(range(lens[0])) if columns is None else columns
            if len(self.columns) != lens[0]:
                raise ValueError("Length of 'columns' differs from number of columns")
            for col, val in zip(self.columns, zip(*data)):
                if not isinstance(val, np.ndarray) or copy:
                    self._data[col] = np.array(val)
                else:
                    self._data[col] = val
        else:
            raise ValueError("Don't know how to create a DataFrame from type '%s'" % type(data).__name__)

        # define loc and iloc
        self.loc = _Loc(self)
        self.iloc = _ILoc(self)

    # ---------------- define the internal accessor functions -------------------

    def _row(self, i, columns=None):
        if i >= self._nrow:
            raise IndexError("Row index out of range: %s" % i)
        if columns is None:
            columns = self.columns
        return _DFRow(columns, [self._data[col][i] if col in self.columns else NA for col in columns])

    def _col(self, item, rows=None):
        if rows is None:
            return self._data[item]
        else:
            return self._data[item][rows]

    def _subset_loc(self, rows, columns=None):
        if columns is None:
            columns = self.columns
        elif isinstance(columns, slice):
            # slice of columns...need to turn into column names
            if columns.start is None:
                colstart = 0
            elif columns.start not in self.columns:
                raise KeyError("Column not found: %s" % columns.start)
            else:
                colstart = self.columns.index(columns.start)

            if columns.stop is None:
                colstop = len(self.columns)
            elif columns.stop not in self.columns:
                raise KeyError("Column not found: %s" % columns.stop)
            else:
                colstop = self.columns.index(columns.stop)

            columns = self.columns[colstart:colstop:columns.step]

        # if rows are a list, they need to be an array
        if np.ndim(rows) == 1:
            rows = np.array(rows)

        if np.ndim(columns) == 1:
            # list of columns
            if isinstance(rows, int):
                # single row
                return self._row(rows, columns)
            elif isinstance(rows, slice) or np.ndim(rows) == 1:
                # data frame subset
                if all(col not in self.columns for col in columns):
                    raise ValueError("None of %s were found in columns" % ", ".join(columns))

                newdata = {}
                for col in columns:
                    newdata[col] = self._data[col][rows]
                return DataFrame(newdata, columns=columns, copy=False)
            else:
                raise ValueError("Don't know how to subset with rows of type %s" % type(rows).__name__)

        elif columns in self.columns:
            # single column
            if isinstance(rows, int):
                # single value
                return self._data[columns][rows]
            elif isinstance(rows, slice) or np.ndim(rows) == 1:
                # subset of column
                return self._data[columns][rows]
            else:
                raise ValueError("Don't know how to subset with rows of type %s" % type(rows).__name__)

        else:
            raise ValueError("Don't know how to subset with columns of type %s" % type(columns).__name__)

    def _subset_iloc(self, rows, columns=None):
        if columns is None:
            return self._subset_loc(rows, self.columns)
        elif isinstance(columns, int) or isinstance(columns, slice):
            return self._subset_loc(rows, self.columns[columns])
        elif np.ndim(columns) == 1:
            # can be an array booleans or ints
            columns = np.array(columns)
            if columns.dtype.kind in ("i", "b"):
                return self._subset_loc(rows, np.array(self.columns)[columns])
            else:
                raise ValueError("Don't know how to subset with column array of type %s" % columns.dtype)

    # ----------------- define the interface -------------------

    def iterrows(self):
        for i in range(self._nrow):
            yield i, self._row(i)

    def iteritems(self):
        for col in self.columns:
            yield col, self._data[col]

    def items(self):
        return self.iteritems()

    def values(self):
        for col in self.columns:
            yield self._data[col]

    def head(self, n=6):
        return self._subset_loc(slice(0, n, None), None)

    def tail(self, n=6):
        return self._subset_loc(slice(-n, None, None), None)

    def groupby(self, by):
        if np.ndim(by) == 0:
            return _GroupBy(self, (by, ))
        else:
            return _GroupBy(self, by)

    # -------------------- define special methods ----------------------------------

    def __delitem__(self, key):
        del self._data[key]

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value, copy=None):
        if copy is None:
            copy = self._copy
        if np.ndim(value) == 0:
            self._data[key] = np.repeat(value, self._nrow)
        elif len(value) != self._nrow:
            raise ValueError("Value implies differing number of rows: %s, %s" % (self._nrow, len(value)))
        elif not isinstance(value, np.ndarray) or copy:
            self._data[key] = np.array(value)
        else:
            self._data[key] = value

        if key not in self.columns:
            self.columns = self.columns + (key, )

    def __len__(self):
        return self._nrow

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return self.columns.__iter__()

    def _repr_html_(self):
        """
        Jupyter Notebook magic repr function.
        """
        head = '<tr><td></td>%s</tr>\n' % ''.join(['<td><strong>%s</strong></td>' % c for c in self.columns])
        rows = ['<td><strong>%d</strong></td>' % i + ''.join(['<td>%s</td>' % c for c in row])
                for i, row in self.iterrows()]
        html = '<table>{}</table>'.format(head + '\n'.join(['<tr>%s</tr>' % row for row in rows]))
        return html

    def __repr__(self):
        strcols = [" ", " --"] + [(" " + str(i)) for i in range(self._nrow)]
        strcols = [strcols, ] + [[str(col), "----"] + [str(val) for val in self._data[col]] for col in self.columns]
        nchars = [max(len(val) for val in col) + 2 for col in strcols]

        rows = []
        for row in zip(*strcols):
            row = list(row)
            rows.append("".join(row[j] + " " * (nchars[j] - len(row[j])) for j in range(len(row))))
        return "\n" + "\n".join(rows) + "\n"


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
                raise KeyError("No such key in row")

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


class _Loc(object):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return self.df._subset_loc(item)
        else:
            return self.df._subset_loc(*item)


class _ILoc(object):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return self.df._subset_iloc(item)
        else:
            return self.df._subset_iloc(*item)


class _GroupBy(object):

    def __init__(self, df, groups):
        if any(col not in df for col in groups):
            raise ValueError("Grouping columns must all be in DataFrame")
        self.df = df
        self.groups = groups

    def apply(self, fun):
        results = [fun(piece) for piece in self]
        return results

    def __iter__(self):
        groupcols = self.groups
        df = self.df

        ids = [tuple(df._subset_loc(i, groupcols)) for i in range(len(df))]
        # sorting/determining unique values by string value
        ids = sorted(enumerate(ids), key=lambda x: "".join([str(val) for val in x[1]]))

        # groupby the tuples
        for group in itertools.groupby(ids, key=lambda x: x[1]):
            yield group[0], df._subset_loc([x[0] for x in group[1]])