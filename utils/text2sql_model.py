from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sqlparse
from pandasql import sqldf

class Text2SqlModel:
    # model_name = 'PipableAI/pip-sql-1.3b'
    def __init__(self):
        model_name = 'defog/sqlcoder-7b-2'
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        #   trust_remote_code=True, 
                                                        #   torch_dtype=torch.float16, 
                                                        #   load_in_8bit=True, 
                                                        #   device_map='auto', 
                                                        #   use_cache=True,
                                                          )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate_sql_query_from_text(self, input_text, max_length=200):
        promt = promt.format(query=input_text)
        inputs = self.tokenizer(promt, return_tensors='pt', padding=True)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        
        results = self.tokenizer.decode(outputs[0])
        
        return results
    
    def generate_sql_query(self, df, primary_key, input_text, max_length=200):
        prompt = self.get_final_promt(df, primary_key)
        updated_promt = prompt.format(query=input_text)
        inputs = self.tokenizer(updated_promt, return_tensors='pt')#.to('cuda')
        generated_ids = self.model.generate(**inputs, 
                                            # num_return_sequences=1,
                                            # eos_token_id=self.tokenizer.eos_token_id,
                                            # pad_token_id=self.tokenizer.eos_token_id,
                                            max_new_tokens=max_length,
                                            # do_sample=False,
                                            # num_beams=1,
                                            )
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)
    
    
    def get_final_promt(self, df, primary_key):
        schema = self.get_schema(df, primary_key)
        prompt = """### Task
                Generate a SQL query to answer [QUESTION]{query}[/QUESTION]

                ### Instructions
                - If you cannot answer the question with the available database schema, return 'I do not know'

                ### Database Schema
                This query will run on a database whose schema is represented in this string:
                

                ### Answer
                Given the database schema, here is the SQL query that answers [QUESTION]{query}[/QUESTION]
                [SQL]
                """
        return prompt
    
    def get_required_data(self, df, primary_key, input_text):
        sql_query = self.generate_sql_query(df, primary_key, input_text)
        result_df = sqldf(sql_query)
        return result_df
    
        
        
        