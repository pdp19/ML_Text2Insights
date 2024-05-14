import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class CricketDataAnalysis():

    def read_csv_file(self, filename):
        df = pd.read_csv('./uploads/' + filename)
        df.head()
        return df

    def drop_null_columns(self, df):
        total_rows = df.shape[0]
        for col in df.columns:
            if df[col].isnull().sum() == total_rows:
                df.drop(col, axis=1, inplace=True)
        return df

    def rename_column_name(self, df):
        rename_dict = {'Unnamed: 0': 'id', 'Player' : 'Player Name', 'Mat':'Matches', 'Inns':'Innings', 'NO':'Not Out', 'HS':'Highest',
            'Ave':'Average', 'BF':'Best Score', 'SR':'Strike Rate'}
        df.rename(columns=rename_dict, inplace=True)
        return df

    def add_extra_columns(self, df):
        df['Country'] = df['Player Name'].apply(lambda x: x.split('(')[1].replace(')',''))
        df['Player Name'] = df['Player Name'].apply(lambda x: x.split('(')[0].strip())
        df['Highest Not Out'] = df['Highest'].apply(lambda x: 1 if '*' in x else 0)
        df['Highest'] = df['Highest'].apply(lambda x: x.replace('*',''))
        df['Start Year'] = df['Span'].apply(lambda x: x.split('-')[0])
        df['End Year'] = df['Span'].apply(lambda x: x.split('-')[1])
        df['Start Year'] = pd.to_datetime(df['Start Year'])
        df['End Year'] = pd.to_datetime(df['End Year'])
        df['Total Days'] = df['End Year'] - df['Start Year']
        df['Player Name'] = df['id'].astype('str') + ' ' + df['Player Name']
        return df

    def impute_null_values(self, df):
        df.replace('-', '0', inplace=True)
        return df

    def convert_data_types(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    try:
                        df[col] = df[col].astype(int)
                    except:
                        try:
                            df[col] = df[col].astype(float)
                        except:
                            df[col] = df[col].astype(str)
        return df

    def drop_column(self, df):
        df.drop('id', axis=1, inplace=True)
        return df

    def save_dataframe(self, df, file_name='output.csv'):
        filename = 'uploads/' + file_name
        df.to_csv(filename, index=False)
        return filename
    
    def change_player_name(self, df):
        df['player_name'] = df['player_name'].apply(lambda x: ' '.join(x.split(" ")[1:]) if ' ' in x else x)
        return df

    def data_analysis_on_cricket(self, filename='./Datasets/ODI_data.csv'):
        df = self.read_csv_file(filename)
        df = self.drop_null_columns(df)
        df = self.rename_column_name(df)
        df = self.add_extra_columns(df)
        df = self.impute_null_values(df)
        df = self.convert_data_types(df)
        df = self.drop_column(df)
        # df = self.change_player_name(df)
        # output_file = self.save_dataframe(df)
        return df

