import pandas as pd
import csv

def main():
    data = pd.read_csv("data/HRBlockIntuitReviewsTrainDev_vLast7.csv")

    # Formo los dataframes que vamos a utilizar en cada prueba
    HRNeg = pd.DataFrame(columns=["reviewText", "summary"])
    HRPos = pd.DataFrame(columns=["reviewText", "summary"])
    IntuitNeg = pd.DataFrame(columns=["reviewText", "summary"])
    IntuitPos = pd.DataFrame(columns=["reviewText", "summary"])

    for idx in data.index:
        row = data.loc[idx]
        brand = row["brand"]
        overall = row["overall"]
        row = row[["reviewText", "summary"]]

        # print("Brand: " + str(brand) + ", " + "Overall: " + str(overall))
        # print(row)

        if "Intuit" in brand:
            if overall > 3:
                IntuitPos = pd.concat([IntuitPos, row], axis=1, join="inner")
            elif overall < 2:
                IntuitNeg = pd.concat([IntuitNeg, row], axis=1, join="inner")
        elif "HR" in brand:
            if overall > 3:
                HRPos = pd.concat([HRPos, row], axis=1, join="inner")
            elif overall < 2:
                HRNeg = pd.concat([HRNeg, row], axis=1, join="inner")

    # print los dataframes nuevos
    print("#########################################################")
    print(HRNeg.head(5))
    print("#########################################################")
    print(HRPos.head(5))
    print("#########################################################")
    print(IntuitNeg.head(5))
    print("#########################################################")
    print(IntuitPos.head(5))
    print("#########################################################")

    i = 1

    for i in [1, 4]:
        if i == 1:
            nombre = "HRNeg.csv"
            df = HRNeg
        elif i == 2:
            nombre = "HRPos.csv"
            df = HRPos
        elif i == 3:
            nombre = "IntuitNeg.csv"
            df = IntuitNeg
        elif i ==4:
            nombre = "IntuitPos.csv"
            df = IntuitPos

        f = open("data/" + nombre)
        writer = csv.writer(f)

        writer.writerow(["reviewText", "summary"])

        for idx in df.index:
            fila = df.iloc[idx]
            writer.writerow(fila)


if __name__ == "__main__":
    main()
