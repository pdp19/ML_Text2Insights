from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Text2ChartTypeModel:
    
    def __init__(self):
        model_name = 'google/flan-t5-base'
        self.model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                          trust_remote_code=True, 
                                                          torch_dtype=torch.float16, 
                                                        #   load_in_8bit=True, 
                                                          device_map='auto', 
                                                          use_cache=True,)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def predict_chart_type(self, input_text, df):
        prompt = self.get_prompt()
        updated_prompt = prompt.format(query=input_text, data=df)
        inputs = self.tokenizer(updated_prompt, return_tensors='pt').input_ids
        outputs = self.model.generate(inputs)
        chart_type = self.tokenizer.decode(outputs[0]).split("<pad>")[-1].replace("</s>","")
        return chart_type
    
    def get_prompt():
        prompt = """Question: "What chart type would best represent {query} and {data}?"
        
                    Possible chart types:
                    1. Box plot
                    2. Line chart
                    3. Pie chart
                    4. Scatter plot
                    5. Histogram
                    6. Bar chart
                    7. Area chart
                    8. Bubble chart
                    9. Radar chart
                    10. Heatmap

                    Choose the most suitable chart type based on the context and characteristics of the data."""
        return prompt