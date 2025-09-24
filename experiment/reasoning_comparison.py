import pandas as pd
import numpy as np
import argparse


def compare_token_count(df_a, df_b):
    """
    Compare the token counts of the two usage dataframes. Go step by step and print out whether the A datframe or the B
    dataframe performs better (or the same). Finally report the overall token counts for both.

    :param df_a: usage dataframe marked as A
    :param df_b: usage dataframe marked as B
    :return: None
    """

    sum_a = 0
    sum_b = 0
    for n in range(60):
        cell_a = np.array(df_a[str(n)])[0] # get the cell for this block
        cell_b = np.array(df_b[str(n)])[0]
        strings_a = cell_a.strip("[]").split(", ") # turn it into a list of strings
        strings_b = cell_b.strip("[]").split(", ")
        for m, (a, b) in enumerate(zip(strings_a, strings_b)):
            sum_a += int(a) # add the token count to the counter
            sum_b += int(b)
            if int(a) > int(b): # figure out which used more tokens
                print(a, b)
                print("A greater than B for Block", n + 1, "Step", m + 1)
            elif int(b) > int(a):
                print(a, b)
                print("B greater than A for Block", n + 1, "Step", m + 1)
            else:
                print(a, b)
                print("A equal to B for Block", n + 1, "Step", m + 1)
        print("Sum A:", sum_a)
        print("Sum B:", sum_b)


def main(deshuffled_a: str, deshuffled_b: str):
    """
    Take two usage .csv files and compare the reasoning token usage between the two step by step, block by block.
    :param deshuffled_a: str pathname for usage dataframe marked as A
    :param deshuffled_b: str pathname for usage dataframe marked as B
    :return:
    """
    df_a = pd.read_csv(deshuffled_a) # build dataframes from the paths
    df_b = pd.read_csv(deshuffled_b)
    compare_token_count(df_a, df_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="Take two usage .csv files and compare the reasoning token usage between the two step by step, block by block.")
    parser.add_argument("deshuffled_a", help="First usage .csv file path (A).", type=str)
    parser.add_argument("deshuffled_b", help="Second usage .csv file path (A).", type=str)

    args = parser.parse_args()

    main(**vars(args))
