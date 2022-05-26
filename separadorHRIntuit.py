import pandas as pd
import csv

def main():
    data = pd.read_csv("data/HRBlockIntuitReviewsTrainDev_vLast7.csv")
    data = data[['overall', 'brand', 'reviewText', 'summary']]

    # Formo los dataframes que vamos a utilizar en cada prueba
    HRNeg = pd.DataFrame(columns=["reviewText", "summary"])
    HRPos = pd.DataFrame(columns=["reviewText", "summary"])
    IntuitNeg = pd.DataFrame(columns=["reviewText", "summary"])
    IntuitPos = pd.DataFrame(columns=["reviewText", "summary"])

    Neg = data.loc[(data['overall'] <3)]
    HRNeg = Neg.loc[(data['brand'].isin(['Administaff HRTools', 'H & R Block', 'H&amp;R Block', 'H&R', 'H&R BLCOK', 'H&R Block', "H&R BLOCK", "HRBB9",'by\n    \n    H&R Block']))]
    IntuitNeg = Neg.loc[(data['brand'].isin(['by\n    \n    Intuit','TurboTax', 'Intuit', 'Intuit Inc.', 'Intuit Inc./BlueHippo', 'Intuit, Inc.']))]
    Pos = data.loc[(data['overall'] >3)]
    HRPos = Pos.loc[(data['brand'].isin(['Administaff HRTools', 'H & R Block', 'H&amp;R Block', 'H&R', 'H&R BLCOK', 'H&R Block', "H&R BLOCK", "HRBB9",'by\n    \n    H&R Block']))]
    IntuitPos = Pos.loc[(data['brand'].isin(['by\n    \n    Intuit','TurboTax', 'Intuit', 'Intuit Inc.', 'Intuit Inc./BlueHippo', 'Intuit, Inc.']))]
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

    HRNeg.to_csv('data/HRNEG.csv')
    HRPos.to_csv('data/HRPOS.csv')
    IntuitNeg.to_csv('data/IntuitNEG.csv')
    IntuitPos.to_csv('data/IntuitPos.csv')

if __name__ == "__main__":
    main()