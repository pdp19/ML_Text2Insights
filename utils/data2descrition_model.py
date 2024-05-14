from transformers import pipeline

class Data2DescriptionModel:
    
    def __init__(self) -> None:
        model_name = 'facebook/bart-large-cnn'
        self.summarizer = pipeline("summarization", model=model_name)
        
    def get_description_for_data(self, df):
        df_string = df.to_string(index=False)
        result = self.summarizer(df_string, max_length=500, min_length=50, do_sample=False, temperature = 0.0)
        return result[0]['summary_text']
    