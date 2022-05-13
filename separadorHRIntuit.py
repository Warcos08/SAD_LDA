import pandas as pd
import csv

def main():
    data = pd.read_csv("HRBlockIntuitReviewsTrainDev_vLast7.csv")
    data = data[['overall', 'brand', 'reviewText', 'summary']]

    # Formo los dataframes que vamos a utilizar en cada prueba
    HRNeg = pd.DataFrame(columns=["reviewText", "summary"])
    HRPos = pd.DataFrame(columns=["reviewText", "summary"])
    IntuitNeg = pd.DataFrame(columns=["reviewText", "summary"])
    IntuitPos = pd.DataFrame(columns=["reviewText", "summary"])

    Neg = data.loc[(data['overall'] <3)]
    HRNeg = Neg.loc[(data['brand'].isin(['Administaff HRTools', 'H&R Block', 'HRBB9', 'by\n    \n    H&R Block']))]
    IntuitNeg = Neg.loc[(data['brand'].isin(['Intuit', 'by\n    \n    Intuit','TurboTax Premier 2014 Fed + State + Fed Efile Tax Software [Old Version]']))]
    Pos = data.loc[(data['overall'] >3)]
    HRPos = Pos.loc[(data['brand'].isin(['Administaff HRTools', 'H&R Block', 'HRBB9', 'by\n    \n    H&R Block']))]
    IntuitPos = Pos.loc[(data['brand'].isin(['Intuit', 'by\n    \n    Intuit','TurboTax Premier 2014 Fed + State + Fed Efile Tax Software [Old Version]']))]
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