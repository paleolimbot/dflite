
import numpy as np
import dflite.dataframe as dfl

# creating a data frame
data = {"cola": [1, 2, 3, 4], "colb": ["one", "two", "three", "four"],
        "colc": ["some", "random", "values", "stuff"]}

# using a dict
df1 = dfl.DataFrame(data)

# using a dict, specifying column order
df2 = dfl.DataFrame(data, columns=("colb", "cola", "colc"))

# using a dict, specifying not all the columns
df3 = dfl.DataFrame(data, columns=("colb", "cola"))

# using a list (assumes by row, can't be an iterator)
df4 = dfl.DataFrame(list(data.values()))
print(df4)

# using a list (force columns), assign names (must be same length as cols)
df5 = dfl.DataFrame(list(zip(*data.values())), columns=("colb", "cola", "colc"))

# using an ndarray
arr = np.array(range(9)).reshape((3, 3))
df6 = dfl.DataFrame(arr)

print(repr(df1))

data["newcol"] = 6
df7 = dfl.DataFrame(data)

print(df7)


print(repr(df7.loc[[True, True, False, True]]))
print(df7.loc[[0, 1, 3]])

a = dfl.DataFrame({"c1":[1, 2, 3, 4], "c2":["one", "two", "three", "four"], "c3":["data1", "data2", "data3", "data4"]})
a["idvar1"] = ["a", "b", "a", "b"]
a["idvar2"] = [1, 1, 1, 2]

print(a.groupby("idvar1").apply(lambda x: x))
print(a.groupby("idvar2").apply(lambda x: x))
print(a.groupby(("idvar1", "idvar2")).apply(lambda x: x))