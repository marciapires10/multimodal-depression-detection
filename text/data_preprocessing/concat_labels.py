import pandas as pd

train_file = '../train_split_Depression_AVEC2017.csv'
dev_file = '../dev_split_Depression_AVEC2017.csv'
test_file = '../full_test_split.csv'

train = pd.read_csv(train_file)
dev = pd.read_csv(dev_file)
test = pd.read_csv(test_file)

drop_columns = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure',
                'PHQ8_Concentrating', 'PHQ8_Moving']

train = train.drop(drop_columns, axis=1)
dev = dev.drop(drop_columns, axis=1)

test.columns = train.columns

full_labels = pd.concat([train, dev, test])

full_labels = full_labels.sort_values(['Participant_ID'], ascending=True)
full_labels = full_labels.reset_index(drop=True)
full_labels.to_csv('full_labels.csv')
