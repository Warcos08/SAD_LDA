import pandas as pd
import csv

def main():
    data = pd.read_csv("data/VG-Reviews5AndMetaElecNintSonyMic_v2.csv")

    # Formo los dataframes que vamos a utilizar en cada prueba
    Nint = data.loc[data["brand"].isin(["by\n    \n    Nintendo", "Nintendo", "Nintendo of America", "Super Nintendo Super Castlevania IV"])]
    NintNeg = Nint.loc[data["overall"] < 3][["brand", "overall", "reviewText", "summary"]]
    NintPos = Nint.loc[data["overall"] > 3][["brand", "overall", "reviewText", "summary"]]
    Sony = data.loc[data["brand"].isin(["Sony", "by\n    \n    Sony", "by\n    \n    Sony Computer Entertainment", "by\n    \n    Sony Computer Entertainment America", "by\n    \n    Sony Online Entertainment", "Sony Computer Entertainment", "Sony Entertainment"])]
    SonyNeg = Sony.loc[data["overall"] < 3][["brand", "overall", "reviewText", "summary"]]
    SonyPos = Sony.loc[data["overall"] > 3][["brand", "overall", "reviewText", "summary"]]
    Micro = data.loc[data["brand"].isin(["Microsoft", "by\n    \n    Electronic Arts", "Electronic Arts", "Electronic Arts, Inc.", "Electronc Arts", "Microsoft Corporation", ])]
    MicroNeg = Micro.loc[data["overall"] < 3][["brand", "overall", "reviewText", "summary"]]
    MicroPos = Micro.loc[data["overall"] > 3][["brand", "overall", "reviewText", "summary"]]

    # print los dataframes nuevos
    print("#########################################################")
    print(NintNeg.head(5))
    print("#########################################################")
    print(NintPos.head(5))
    print("#########################################################")
    print(SonyNeg.head(5))
    print("#########################################################")
    print(SonyPos.head(5))
    print("#########################################################")
    print(MicroNeg.head(5))
    print("#########################################################")
    print(MicroPos.head(5))
    print("#########################################################")

    NintNeg.to_csv("data/NintNeg.csv")
    NintPos.to_csv("data/NintPos.csv")
    SonyNeg.to_csv("data/SonyNeg.csv")
    SonyPos.to_csv("data/SonyPos.csv")
    MicroNeg.to_csv("data/MicroNeg.csv")
    MicroPos.to_csv("data/MicroPos.csv")


if __name__ == "__main__":
    main()
