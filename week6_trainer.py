import sys
import pandas as pd

def readfile(file):
    df = pd.read_csv(file)
    return df

def quantize_to_half(value):
    return round(value * 2) / 2

def rounding_data(df):
    column_names = df.columns
    # print(column_names)
    for column in column_names:
        if column =="Age" or column =="Ht":
            df[column] = df[column].round()
        elif column == "TailLn" or column=="HairLn" or column == "BangLn" or column == "Reach":
            df[column] = df[column].apply(quantize_to_half)
    return df

def creating_decision_tree(df):
    pass
    
def main():
    # Check the number of command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    # Retrieve the file path from the command-line arguments
    file_path = sys.argv[1]

    # Now you can use the 'file_path' variable in your script
    print(f"Processing file: {file_path}")
    df = readfile(file_path)
    df = (rounding_data(df))
    creating_decision_tree(df)

if __name__ == "__main__":
    main()