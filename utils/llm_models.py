from groq import Groq
import os
from dotenv import load_dotenv
import json
from pandasql import sqldf


class LlmModels():
    
    def __init__(self):
        self.model_name = "llama3-70b-8192"
        load_dotenv()
        self.api_key_value = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key_value,)
    
    def generate_output_from_llm(self, prompt, max_tokens=200):
        chat_completion = self.client.chat.completions.create(
            messages=prompt,
            model=self.model_name,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1,
            stop=None,
            stream=False,)
        return chat_completion.choices[0].message.content
    
    ### Query to SQL generation
    def get_prompt_sql(self, schema, input_query):
        success_json = self.get_success_json_sql()
        failure_json = self.get_failure_json_sql()
        prompt_sql = [
            {"role": "system", "content": "You are a helpful digital assistant which provide SQL query about the asked question. Please provide correct, pricise and accurate SQL query."},
            {"role": "system", "content": "As a smart assistant, you will receive a user's question, and database schema from where data is fetched"},
            {"role": "system", "content": "Your objective is to thoroughly analyze the database schema and provide SQL query for the user's question from that schema."},
            {"role": "system", "content": "Give table name as 'table_name' always in the resulted query"},
            {"role": "system", "content": "If the database schema or question is not provided, return 'I do not know.'"},
            {"role": "system", "content": "If the question is not related with the provided schema, return 'Wrong question for the data.'"},
            {"role": "system", "content": f"Return the success results in this format: '{success_json}'"},
            {"role": "system", "content": f"Return the failure results in this format: '{failure_json}'"},
            {"role": "user", "content": f"Can you provide only correct and accurate SQL Query for: Database schema '{schema}' user's query '{input_query}'."},
          ]
        return prompt_sql
    
    def get_success_json_sql(self):
        json = "{'sql':''}"
        return json
    
    def get_failure_json_sql(self):
        json = "{'error':''}"
        return json
        
    def generate_sql_query(self, schema, input_query):
        prompt = self.get_prompt_sql(schema, input_query)
        try:
            output = self.generate_output_from_llm(prompt)
            print("SQL Output: " + output)
            if "{'error':" in output:
                print("Error occurecd while generating SQL query!" + output)
                return "Error occurecd while generating SQL query!" + output
        except:
            print("Exception occurecd while generating SQL query!")
            return "Exception occurecd while generating SQL query!"
        output = "{" + output.split("{")[1].split("}")[0] + "}"
        output = output.replace("\n", " ")
        try:
            dict_result = json.loads(output.replace("'", "\""))
        except:
            return "Exception occurecd while converting SQL query to json!"
        return dict_result
    
    def get_required_data(self, df, schema, input_text):
        sql_query = self.generate_sql_query(schema, input_text)
        if 'Error occurecd' in sql_query:
            return None
        if 'Exception occurecd' in sql_query:
            return None
        final_sql_query = sql_query.get('sql').replace('table_name', 'df')
        try:
            result_df = sqldf(final_sql_query)
        except:
            print("Exception occurecd while generating dataset from SQL query!")
            return None
        return result_df
    
    ### Data to insight generator
    def get_prompt_description(self, df, schema, input_query):
        success_json = self.get_success_json_description()
        failure_json = self.get_failure_json_description()
        prompt_description = [
            {"role": "system", "content": "You are a helpful digital assistant which provide insightes about the provided data. Please provide correct, pricise and accurate information to the user."},
            {"role": "system", "content": "As a smart assistant, you will receive a subset of user question, data which answer's that question and schema of the dataset from where data is fetched"},
            {"role": "system", "content": "Your objective is to thoroughly analyze the data fetched from the user's question and offer precise insights."},
            {"role": "system", "content": f"Return the success results in this format: {success_json}"},
            {"role": "system", "content": f"Return the failure results in this format: {failure_json}"},
            {"role": "system", "content": "If the dataset or question is not provided, return 'I do not know.'"},
            {"role": "user", "content": f"Database schema {schema} user's question {input_query} and data returned for this {df}. Can you provide some insights in bullet point."},
          ]
        return prompt_description
    
    def get_success_json_description(self):
        json = "{'description':''}"
        return json
    
    def get_failure_json_description(self):
        json = "{'error':''}"
        return json
    
    def generate_description(self, df, schema, input_query):
        prompt = self.get_prompt_description(df, schema, input_query)
        try:
            output = self.generate_output_from_llm(prompt, max_tokens=500)
            print("Insights Output: " + output)
            if "{'error':" in output:
                print("Error occurecd while generating insights for query!" + output)
                return None
        except:
            print("Exception occurecd while generating insights for query!")
            return None
        output = "{" + output.split("{")[1].split("}")[0] + "}"
        description = output.split("'description':")[1].split("}")[0]
        return description
    
    ### Predict Chart Type from the given data
    def get_prompt_chart_type(self, df, input_query):
        success_json = self.get_success_json_chart_type()
        failure_json = self.get_failure_json_chart_type()
        possible_chart_type = '''Possible chart types:
                        1. Box Plot
                        2. Line Chart
                        3. Pie Chart
                        4. Scatter Plot
                        5. Histogram
                        6. Bar Chart
                        7. Area Chart
                        8. Bubble Chart
                        9. Radar Chart
                        10. Heatmap'''
        prompt_description = [
            {"role": "system", "content": "You are a helpful digital assistant which provide 'chart type' from the asked query. Please provide correct, pricise and accurate chart type."},
            {"role": "system", "content": "As a smart assistant, You will receive a user's query, database schema and the dataset for which chart is required."},
            {"role": "system", "content": "Your objective is to thoroughly analyze the database schema, provided dataset and user's query and provide accurate chart type."},
            {"role": "system", "content": "Give preference to user's question first, if query ask for any specific chart type then return with that chart type."},
            {"role": "system", "content": "If query does not specify for any specific chart type then analyse the dataset and return with accurate chart type."},
            {"role": "system", "content": "Provide x-axis and y-axis as well for that chart type from the provided dataset"},
            {"role": "system", "content": "Provide correct x-axis and y-axis from dataset, like box plot is made of 2 numeric value or bar chart is made of one categorical and one numeric value, like so for other chart type as well"},
            {"role": "system", "content": possible_chart_type},
            {"role": "system", "content": f"Return the success results in this format: {success_json}"},
            {"role": "system", "content": f"Return the failure results in this format: {failure_json}"},
            {"role": "system", "content": "If there are multiple data and one chart would not be enough then provide muliple chart type with x-axis and y-axis."},
            {"role": "system", "content": "If the dataset or query is not provided, return with blank 'chart_type'"},
            {"role": "user", "content": f"dataset {df} user's query {input_query}. Can you provide only correct and accurate chart type."},
          ]
        return prompt_description
    
    def get_success_json_chart_type(self):
        json = "{'chart_type':'','x_axis':'','y_axis':''}"
        return json
    
    def get_failure_json_chart_type(self):
        json = "{'error':''}"
        return json
    
    def predict_chart_type(self, df, input_query):
        prompt = self.get_prompt_chart_type(df, input_query)
        try:
            output = self.generate_output_from_llm(prompt, max_tokens=100)
            print("Chart Output: " + output)
            result = "{" + output.split("{")[1].split("}")[0] + "}"
            if "{'error':" in output:
                print("Error occurecd while predicting chart type!" + output)
                return "Error occurecd while predicting chart type!" + output
        except:
            print("Exception occurecd while predicting chart type!")
            return "Exception occurecd while predicting chart type!"
        return result