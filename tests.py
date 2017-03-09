

import dflite as df
import os

def test_dataframe():
    a = df.DataFrame([1, 2, 3], ["one", "two", "three"], ["data1", "data2", "data3"])
    b = df.DataFrame([], [], [], columns=["Column1", "Column2", "Column3"])
    print(a[0])
    print(a[1])
    print(b[0])
    print(b[1])
    print(b.Column1)
    print(b.Column2)
    print(len(b))
    print(b.ncol())
    # a.newcol = [1.23, 4.44, 9.19] #this does not add 'newcol' to columns, and does not work. use [] for assigning
    a["newcol"] = [1.23, 4.44, 9.19]
    b["newcol"] = [] #like this
    b.append(1,2,3, newcol="bananas")
    a.append([4, 5], ["four", "five"], ["data4", "data5"], newcol=[13, 10])
    print(a.columns)
    a.columns = ("col1", "col2", "col3", "col4")
    print(a.columns)
    a.columns = ("col2", "col3", "col4", "col1")
    print(a._repr_html_())
    print(a.head(2))
    print(a.tail(2))

    print(a)
    print(b)
    with open("fish.csv", "w") as f:
        a.to_csv(f, driver="csv")
    a.to_csv("fish.tsv", driver="tsv")
    b.to_csv("fish.csv")
    b.to_csv("fish.csv", mode="a")
    c = df.read_csv("fish.csv")
    print(c)
    d = df.read_csv("fish.csv", headers=False)
    print(d)

    with open("fish.csv") as f:
        e = df.read_csv(f, driver="csv")
        print(e)

    a = df.DataFrame([1,2], columns=["fish",])
    a["fish"] = ["one", "two"]
    print(a)
    print("-----")
    # print(b.rowasdict(0))
    print(a.iloc[0])  # should be the same
    print(a.iloc[0, :])  # should be the same except as data frame

    print(b.copy())

    os.unlink("fish.tsv")
    os.unlink("fish.csv")

def test_groupby():
    a = df.DataFrame([1, 2, 3, 4], ["one", "two", "three", "four"], ["data1", "data2", "data3", "data4"])
    a["idvar1"] = ["a", "b", "a", "b"]
    a["idvar2"] = [1, 1, 1, 2]

    print(a.groupby("idvar1").apply(lambda x: x))
    print(a.groupby("idvar2").apply(lambda x: x))
    print(a.groupby(("idvar1", "idvar2")).apply(lambda x: x))



if __name__ == "__main__":
    # test_dataframe()
    test_groupby()