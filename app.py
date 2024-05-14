import pandas as pd
from flask import Flask, render_template, request, jsonify
from utils.text2sql_model import Text2SqlModel
from utils.data2descrition_model import Data2DescriptionModel
from utils.text2chart_type_model import Text2ChartTypeModel
from utils.any_data_analysis import AnyDataAnalysis
from utils.data_visualization import DataVisualization

app = Flask(__name__)
data_analysis = AnyDataAnalysis()
sql_model = Text2SqlModel()
description_model = Data2DescriptionModel()
chart_type_model = Text2ChartTypeModel()
data_visualization = DataVisualization()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file found!'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file is selected, please select CSV file'
    if '.csv' not in file.filename:
        return 'Please select a valid CSV file!'
    
    file.save('uploads/' + file.filename)
    
    return 'File uploaded successfully!'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    file_name = data.get('file_name')

    answer = process_question(question, file_name)
    return answer

    # return jsonify({'answer': answer})

def process_question(question, file_name):
    if question is None:
        return jsonify({'question': 'No question provided!'})
    else:
        df = data_analysis.data_analysis(file_name)
        primary_key = data_analysis.get_primary_key(df)
        subset_df = sql_model.get_required_data(df, primary_key, question)
        subset_insight = 'Insights for your data:\n' + description_model.get_description_for_data(subset_df)
        # subset_insight = 'Insights for your data:\n' + description_model.get_description_for_data(df.iloc[0:5,1:3])
        chart = data_visualization
        # subset_data = 'Your output data:\n' + data_analysis.format_dataframe(df.iloc[0:5,1:3])
        subset_data = 'Your output data:\n' + data_analysis.format_dataframe(subset_df)
            
    return jsonify({'question': 'Your Question:\n' + question, 'data': subset_data, 'description': subset_insight, 'chart': df.head(5).to_string(index=False)})
    

if __name__ == '__main__':
    app.run(debug=True)
