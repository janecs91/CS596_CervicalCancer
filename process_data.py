import numpy as np
import pandas as pd
 

def load_data():
    # Flags
    drop_missing_rows = True
    show_missing = False
    show_mean_std = False
    show_correlation = False
    show_feature_frequency = False

    # Load data
    file_name = "data/kag_risk_factors_cervical_cancer.csv"
    data = pd.read_csv(file_name)
    cols = data.columns
    # binary columns
    bin_cols = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
                'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 
                'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN', 
                'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
    # numerical columns
    num_cols = data.columns.difference(bin_cols)
    #print(num_cols)
    label_col = 'Biopsy'

    # Clean missing values
    data = data.replace('?', np.NaN)
    # Convert data to numeric
    data = data.apply(pd.to_numeric)
    #print(data.dtypes)
    bad_columns = []

    # Replace missing values with mean (so missing data won't affect training too much)
    for col in cols:
        col_data = data[col].astype(float)
        col_mean = col_data.mean()
        col_std = col_data.std()
        if col_std == 0.0:
            # no variance in column
            bad_columns.append(col)
        if col in bin_cols:
            #print("{} is binary".format(col))
            if drop_missing_rows is False:
                col_mean = int(col_mean)  
        if col_mean is not None:
            data[col] = col_data.fillna(round(col_mean,1))
        col_mean = None
    # Drop rows with missing values
    if drop_missing_rows:
        data.dropna(inplace=True)

    # Analyze missing values
    if show_missing:
        for col in cols:
            missing_values = data[col].isnull().sum()
            print("{}: {}".format(col, missing_values))

    # Columns with too much missing data
    # These have 787 missing values out of 858 samples
    bad_columns.extend(['STDs: Time since last diagnosis','STDs: Time since first diagnosis'])

    # Mean, Standard Deviation
    if show_mean_std is True:
        for col in cols:
            col_data = data[col].astype(float)
            mean = col_data.mean()
            std = col_data.std()
            print("{}: {}, {}".format(col, mean, std))
        # plot

    # Drop bad columns
    #print(bad_columns)
    data.drop(bad_columns, axis=1, inplace=True)
    num_cols = num_cols.difference(bad_columns)

    # Correlation Matrix
    corr = data.corr()
    if show_correlation:    
        print(corr)
        # plot
        size=10
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar(plt.pcolor(corr))

    corrl = corr[label_col]
    corrl = corrl.abs().drop(label_col)
    corrl_sorted = corrl.sort_values(kind="quicksort", ascending=False)
    feature_sequence = list(corrl_sorted.index)

    features_data = data[feature_sequence] #data.drop(columns=label_col)
    label_data = data[label_col]
    print(features_data.columns)

    if show_feature_frequency:
        for i in range(11):
            tfn = features_data.columns[i]
            tfd = data[tfn]
            tt = len(data)
            #print(tt)
            if tfn in bin_cols:
                # bar graph
                fd00 = len(data[(tfd<1)&(label_data<1)])
                fd01 = len(data[(tfd<1)&(label_data>0)])
                fd10 = len(data[(tfd>0)&(label_data<1)])
                fd11 = len(data[(tfd>0)&(label_data>0)])
                x= ['{}:0,\n {}:0'.format(tfn,label_col), '{}:0,\n {}:1'.format(tfn,label_col), '{}:1,\n {}:0'.format(tfn,label_col),
                    '{}:1,\n {}:1'.format(tfn,label_col)]
                y= [fd00, fd01, fd10, fd11]
                for i, v in enumerate(y):
                    plt.text(i, v, str(v))
                plt.ylabel('Frequency')
                plt.bar(x,y)
                plt.show()
            else:
                # scatter
                x= tfd
                y= label_data
                plt.scatter(x,y)
                plt.xlabel(tfn)
                plt.ylabel(label_col)
                plt.show()
    return features_data, label_data
