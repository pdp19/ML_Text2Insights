import pandas as pd
from num2words import num2words
from utils.cricket_data_analysis import CricketDataAnalysis

class AnyDataAnalysis():
    
    def read_csv_file(self, filename):
        df = pd.read_csv('./uploads/' + filename)
        return df
    
    def read_any_file(self, filename):
        df = pd.read_csv(filename)
        return df
    
    def save_any_file(self, df, filename):
        df.to_csv(filename, index=False)
        return df
    
    def save_schema(self, schema, filename):
        with open(filename, 'w') as file:
            file.write(schema)
            
    def read_schema(self, filename):
        with open(filename, 'r') as file:
            schema = file.read()
        return schema

    def drop_null_columns(self, df):
        row_size = df.shape[0]
        for col in df.columns:
            if row_size == df[col].isnull().sum():
                df.drop(col, axis=1, inplace=True)
        return df

    def drop_null_rows(self, df):
        col_size = df.shape[1]
        for index,row in df.iterrows():
            if col_size-1 <= row.isnull().sum():
                df.drop(index, axis=0, inplace=True)
        return df

    def is_numeric(self, val):
        try:
            int(val)
            return True
        except:
            return False

    def convert_digit_columns_to_word(self, df):
        for col in df.columns:
            if self.is_numeric(col):
                df.rename(columns={col:num2words(col)}, inplace=True)
        return df

    def convert_data_types(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                # try:
                    # df[col] = pd.to_datetime(df[col])
                # except:
                try:
                    df[col] = df[col].astype(int)
                except ValueError:
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        df[col] = df[col].astype(str)   
        return df

    def update_column_name(self, df):
        for col in df.columns:
            df.rename(columns={col: col.lower().replace(' ', '_').strip()}, inplace=True)
        return df

    def get_primary_key(self, df):
        primary_key = ''
        total_rows = df.shape[0]
        distinct_values = {col: df[col].nunique for col in df.columns}
        for col,count in distinct_values.items():
            if total_rows == count:
                primary_key = col
                break
        return primary_key

    def data_analysis(self, filename):
        existing_file_names = ['ODI_data.csv', 't20.csv', 'test.csv', 'Bowling_ODI.csv', 'Bowling_t20.csv', 'Bowling_test.csv', 'Fielding_ODI.csv', 'Fielding_t20.csv', 'Fielding_test.csv']
        if filename in existing_file_names:
            cricketData = CricketDataAnalysis()
            df = cricketData.data_analysis_on_cricket(filename)
        else:
            df = self.read_csv_file(filename)
        df = self.drop_null_columns(df)
        df = self.drop_null_rows(df)
        df = self.convert_digit_columns_to_word(df)
        df = self.convert_data_types(df)
        df = self.update_column_name(df)
        return df
    
    def get_schema(self, df):
        schema = ""
        primary_key = self.get_primary_key(df)
        start_text = "<schema> create table " + schema + " ( "
        table_text = ""
        end_text = " );</schema> "
        for col in df.columns:
            data_type_text = ''
            primary_text = ''
            data_type = str(df[col].dtypes)
            
            if data_type.find('int') != -1:
                data_type_text = " INT "
            elif data_type.find('datetime') != -1:
                data_type_text = " DATE "
            elif data_type.find('float') != -1:
                data_type_text = " DECIMAL(10, 2) "
            else:
                data_type_text = " VARCHAR "
            if col == primary_key:
                primary_text = " PRIMARY KEY "
            
            table_text += col + data_type_text + primary_text + ",\n"
            
        schema = start_text + table_text + end_text
        return schema
    
    def format_dataframe(self, df):
        max_lengths = [max(len(str(header)), max(len(str(cell)) for cell in df[col])) for col, header in zip(df.columns, df.columns.astype(str))]
        table_str = ''
        top_line = '=' * (sum(max_lengths) + 3 * len(max_lengths))
        table_str += top_line + '\n'
        header_row = ''
        for i, col in enumerate(df.columns):
            header_row += f"{col:<{max_lengths[i]}} | "
        table_str += header_row.rstrip(" | ") + '\n'
        bold_line = '-' * (sum(max_lengths) + 3 * len(max_lengths))
        table_str += bold_line + '\n'
        for index, row in df.iterrows():
            row_str = ''
            for i, (col, max_length) in enumerate(zip(row, max_lengths)):
                row_str += f"{str(col):<{max_length}} | "
            table_str += row_str.rstrip(" | ") + '\n'
        table_str += bold_line
        return table_str


                    