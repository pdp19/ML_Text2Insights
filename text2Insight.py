import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from utils.any_data_analysis import AnyDataAnalysis
from utils.data_visualization import DataVisualization
from utils.llm_models import LlmModels

app = Flask(__name__)
data_analysis = AnyDataAnalysis()
llama = LlmModels()
data_visualization = DataVisualization()
FILE_NAME = "./CSV/Full_df.csv"
SUBSET_FILE_NAME = "./CSV/Subset_df.csv"
SCHEMA_FILE_NAME = "./CSV/Schema.txt"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/charts/<path:filename>')
def serve_image(filename):
    # Assuming the image files are stored in the 'charts' directory
    image_path = f'./charts/{filename}'
    return send_file(image_path)

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

    df = data_analysis.data_analysis(file.filename)
    data_analysis.save_any_file(df, FILE_NAME)
    schema = data_analysis.get_schema(df)
    data_analysis.save_schema(schema, SCHEMA_FILE_NAME)
    return 'File uploaded successfully!'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    file_name = data.get('file_name')

    answer = process_question(question, file_name)
    return answer

def process_question(question, file_name):
    if question is None:
        return jsonify({'question': 'No question provided!'})
    else:
        df = data_analysis.read_any_file(FILE_NAME)
        schema = data_analysis.get_schema(df)
        subset_df = llama.get_required_data(df, schema, question)
        subset_data_str = 'Your output data:\n' + data_analysis.format_dataframe(subset_df)
        subset_insight = 'Insights for your data:\n' + llama.generate_description(subset_df, schema, question)
        chart_type = llama.predict_chart_type(subset_df, question)
        filename = data_visualization.create_chart(subset_df, chart_type)
            
    return jsonify({'question': 'Your Question:\n' + question, 'data': subset_data_str, 'description': subset_insight, 'chart': filename})

@app.route('/get_data', methods=['POST'])
def get_required_data():
    data = request.json
    question = data.get('question')
    
    df = data_analysis.read_any_file(FILE_NAME)
    if df.shape[0] <= 1 or df.shape[1] <= 1:
        return jsonify({'question': 'Your Question:\n' + question, 'data': 'Empty dataset!'})
    else:
        schema = data_analysis.read_schema(SCHEMA_FILE_NAME)
        subset_df = llama.get_required_data(df, schema, question)
        if subset_df is None:
            return jsonify({'question': 'Your Question:\n' + question, 'data': 'Error Getting SQL Query!'})
        else:
            data_analysis.save_any_file(subset_df, SUBSET_FILE_NAME)
            subset_data_str = 'Your output data:\n' + data_analysis.format_dataframe(subset_df)
            return jsonify({'question': 'Your Question:\n' + question, 'data': subset_data_str})
    
@app.route('/get_insights', methods=['POST'])
def get_insights_for_query():
    data = request.json
    question = data.get('question')
    
    subset_df = data_analysis.read_any_file(SUBSET_FILE_NAME)
    schema = data_analysis.read_schema(SCHEMA_FILE_NAME)
    subset_insight = 'Insights for your data:\n' + llama.generate_description(subset_df, schema, question)
    chart_type = llama.predict_chart_type(subset_df, question)
    filename = data_visualization.create_chart(subset_df, chart_type)
    if subset_insight is None:
        return jsonify({'description': 'Error Getting Insights!', 'chart': filename})
    if filename is None:
        return jsonify({'description': subset_insight, 'chart': 'Error Getting Charts!'})
    
    return jsonify({'description': subset_insight, 'chart': filename})

if __name__ == '__main__':
    app.run(debug=True)
