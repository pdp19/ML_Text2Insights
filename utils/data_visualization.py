import json
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import random
from sklearn.preprocessing import MinMaxScaler
from math import pi
import pandas as pd

class DataVisualization():
    
    def __init__(self):
        pass
    
    def create_chart(self, df, chart_type_json):
        dict_chart_type = json.loads(chart_type_json.replace("'", "\""))
        chart_type = dict_chart_type.get('chart_type')
        x_axis = dict_chart_type.get('x_axis')
        y_axis = dict_chart_type.get('y_axis')
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
        if 'Box' in chart_type:
            filename = self.create_box_chart(df, x_axis, y_axis)
        elif 'Line' in chart_type:
            filename = self.create_line_chart(df, x_axis, y_axis)
        elif 'Pie' in chart_type:
            filename = self.create_pie_chart(df, x_axis, y_axis)
        elif 'Scatter' in chart_type:
            filename = self.create_scatter_chart(df, x_axis, y_axis)
        elif 'Histogram' in chart_type:
            filename = self.create_histogram_chart(df, x_axis, y_axis)
        elif 'Bar' in chart_type:
            filename = self.create_bar_chart(df, x_axis, y_axis)
        elif 'Area' in chart_type:
            filename = self.create_area_chart(df, x_axis, y_axis)
        elif 'Bubble' in chart_type:
            filename = self.create_bubble_chart(df, x_axis, y_axis)
        elif 'Radar' in chart_type:
            filename = self.create_radar_chart(df, x_axis, y_axis)
        elif 'Heatmap' in chart_type:
            filename = self.create_heatmap_chart(df)
        else:
            filename = self.create_bar_chart(df, x_axis, y_axis)
            
        return filename
    
    def create_box_chart(self, df, x_axis, y_axis):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 2:
                return f"Box plot is not suitable for this dataset, as dataframe does not have required numeric value columns!"
            df.boxplot(column=list(numeric_cols))
            plt.title('Box Plot for your Question', fontsize=16, fontweight='bold')
            plt.xticks(range(1, len(numeric_cols) + 1), numeric_cols)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "box_plot_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Box Plot!"
        
        return filename
    
    def create_line_chart(self, df, x_axis, y_axis):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 1:
                return f"Line plot is not suitable for this dataset, as dataframe does not have required numeric value columns!"
            df = df[numeric_cols]
            df.plot(kind='line', marker='o', linestyle='-')
            plt.title('Line Chart for your Question', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "line_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Line Chart!"
        
        return filename
    
    def create_pie_chart(self, df, x_axis, y_axis):
        try:
            if df[x_axis].dtype in ('int64', 'float64'):
                if df[y_axis].dtype in ('int64', 'float64'):
                    return f"Pie chart is not suitable for this dataset, as both {x_axis} or {y_axis} is numeric value"
            if df[x_axis].dtype not in ('int64', 'float64'):
                if df[y_axis].dtype not in ('int64', 'float64'):
                    return f"Pie chart is not suitable for this dataset, as neither {x_axis} nor {y_axis} is numeric value"
            
            if df[x_axis].dtype in ('int64', 'float64'):
                numeric = df[x_axis]
                categorical = df[y_axis]
            else:
                numeric = df[y_axis]
                categorical = df[x_axis]
            fig, ax = plt.subplots()
            ax.pie(numeric, labels=categorical, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Pie Chart for your Question', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "pie_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)
            
            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Pie Chart!"
        
        return filename
    
    def create_scatter_chart(self, df, x_axis, y_axis):
        try:
            if df[x_axis].dtype not in ('int64', 'float64'):
                return f"Scatter chart is not suitable for this dataset, as {x_axis} is not numeric value"
            if df[y_axis].dtype not in ('int64', 'float64'):
                return f"Scatter chart is not suitable for this dataset, as {y_axis} is not numeric value"

            plt.scatter(df[x_axis], df[y_axis])
            
            plt.title('Scatter Chart for your Question', fontsize=16, fontweight='bold')
            x_label = x_axis[0].upper() + x_axis[1:].replace("_"," ")
            y_label = y_axis[0].upper() + y_axis[1:].replace("_"," ")
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "scatter_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Scatter Chart!"
        
        return filename
    
    def create_histogram_chart(self, df, x_axis, y_axis):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 1:
                return f"Hostogram plot is not suitable for this dataset, as dataframe does not have required numeric value columns!"
            # df = df[numeric_cols]
            if df[x_axis].dtype in ('int64', 'float64'):
                numeric = df[x_axis]
                label = x_axis
            else:
                numeric = df[y_axis]
                label = y_axis

            iqr = np.percentile(numeric, 75) - np.percentile(numeric, 25)
            bin_width = 2 * iqr * len(numeric) ** (-1/3)
            num_bins = int((max(numeric) - min(numeric)) / bin_width)
            
            plt.hist(numeric, bins=num_bins)
            
            plt.title('Histogram for your Question', fontsize=16, fontweight='bold')
            x_label = label[0].upper() + label[1:].replace("_"," ")
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "histogram_plot_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Histogram!"
        
        return filename
        
    def create_bar_chart(self, df, x_axis, y_axis):
        try:
            if df[x_axis].dtype in ('int64', 'float64'):
                if df[y_axis].dtype in ('int64', 'float64'):
                    return f"Bar chart is not suitable for this dataset, as both {x_axis} or {y_axis} is numeric value"
            if df[x_axis].dtype not in ('int64', 'float64'):
                if df[y_axis].dtype not in ('int64', 'float64'):
                    return f"Bar chart is not suitable for this dataset, as neither {x_axis} nor {y_axis} is numeric value"

            x_axis_length = len(df[x_axis])
            title_text = 'Bar Chart for your Question'
            title_fontsize = 16
            title_width_inches = len(title_text) * title_fontsize * 0.01
            fig_size = (x_axis_length * 1.5 + title_width_inches, 6)

            fig, ax = plt.subplots(figsize=fig_size)
            bars = ax.bar(df[x_axis], df[y_axis], color='skyblue', alpha=0.7)
            ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold')
            x_label = x_axis[0].upper() + x_axis[1:].replace("_"," ")
            y_label = y_axis[0].upper() + y_axis[1:].replace("_"," ")
            ax.set_xlabel(x_label, fontsize=14)
            ax.set_ylabel(y_label, fontsize=14)
            ax.tick_params(axis='x', labelrotation=90, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)  
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            def add_value_lables(bars):
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)
            add_value_lables(bars)
            
            fig.set_facecolor('none')
            ax.set_facecolor('none') 
            plt.tight_layout()
            
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "bar_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)
            
            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Bar Chart!"
        
        return filename
    
    def create_area_chart(self, df, x_axis, y_axis):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 2:
                return f"Box plot is not suitable for this dataset, as dataframe does not have required numeric value columns!"
            for col in numeric_cols:
                plt.fill_between(range(len(df[col])), df[col].values, alpha=0.4, label=col)
            
            plt.title('Area Chart for your Question', fontsize=16, fontweight='bold', loc='center')
            plt.xlabel('Index', fontsize=12)
            plt.ylabel('Values', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "area_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Area Chart!"
        
        return filename
    
    def create_bubble_chart(self, df, x_axis, y_axis):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 2:
                return f"Box plot is not suitable for this dataset, as dataframe does not have required numeric value columns!"
            x_axis = numeric_cols[0]
            y_axis = numeric_cols[1]
            min_size=10
            max_size=300
            distances = np.sqrt((np.diff(df[x_axis])**2) + (np.diff(df[y_axis])**2))
            normalized_distances = (distances - min(distances)) / (max(distances) - min(distances))
            sizes = (normalized_distances * (max_size - min_size)) + min_size
            sizes = sizes.tolist()
            
            plt.scatter(df[x_axis], df[y_axis], s=sizes)
            
            plt.title('Bubble Chart for your Question', fontsize=16, fontweight='bold')
            x_label = x_axis[0].upper() + x_axis[1:].replace("_"," ")
            y_label = y_axis[0].upper() + y_axis[1:].replace("_"," ")
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "bubble_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Bubble Chart!"
        
        return filename
    
    def create_radar_chart(self, df, x_axis, y_axis):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 2:
                return f"Radar chart is not suitable for this dataset, as dataframe does not have required numeric value columns!"
                    
            scaler = MinMaxScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
            fig_size = (min(df_normalized.shape[0] * 1.5, 12), 6)
            categories = list(df_normalized.columns)
            values = df_normalized.iloc[0].tolist()
            values += values[:1]

            num_vars = len(categories)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

            ax.plot(angles, values, color='skyblue', linewidth=2, linestyle='solid')
            ax.fill(angles, values, color='skyblue', alpha=0.4)

            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)

            plt.title('Radar Chart for your Question', fontsize=16, fontweight='bold')
            plt.tight_layout()
                
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "radar_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)
                
            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Radar Chart!"

        return filename
    
    def create_heatmap_chart(self, df):
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 2:
                return f"Heatmap plot is not suitable for this dataset, as dataframe does not have required numeric value columns!"
            
            df = df[numeric_cols]
            sns.heatmap(df.corr(), cmap='viridis', annot=True, fmt=".2f", linewidths=.5)
            
            plt.title('Heatmap for your Question', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            random_value = str(random.uniform(1000, 10000000))
            filename = "./charts/" + "heatmap_chart_" + random_value + ".png"
            plt.savefig(filename, format='png', transparent=True)

            plt.clf()
            plt.close()
        except:
            return "Provided dataset is not suitable for Heatmap!"
        
        return filename
    
    def get_max_fig_size(self, fig_size):
        static_fig_size = (12, 6)
        static_area = static_fig_size[0] * static_fig_size[1]
        dynamic_area = fig_size[0] * fig_size[1]
        return dynamic_area if dynamic_area > static_area else static_area
    
    # def create_bar_chart(self, df, x_axis, y_axis):
    #     fig_size = (len(df[x_axis]) * 1.5, 6)
    #     plt.figure(figsize=fig_size)
    #     bars = plt.bar(df[x_axis], df[y_axis], color='skyblue', alpha=0.7)
    #     plt.title('Bar Chart for your Question', fontsize=16, fontweight='bold')
    #     x_label = x_axis[0].upper() + x_axis[1:].replace("_"," ")
    #     y_label = y_axis[0].upper() + y_axis[1:].replace("_"," ")
    #     plt.xlabel(x_label, fontsize=14)
    #     plt.ylabel(y_label, fontsize=14)
    #     plt.xticks(fontsize=12, rotation=45, ha='right')
    #     plt.yticks(fontsize=12)
    #     plt.grid(axis='y', linestyle='--', alpha=0.5)
    #     plt.gca().set_facecolor('none')
        
    #     def add_value_lables(bars):
    #         for bar in bars:
    #             yval = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)
    #     add_value_lables(bars)
    #     plt.tight_layout()
        
    #     random_value = str(random.uniform(1000, 10000000))
    #     filename = "./charts/" + "bar_chart_" + random_value + ".png"
    #     plt.savefig(filename, format='png')
        
    #     plt.clf()
    #     plt.close()
        
    #     return filename
    
    #   def create_radar_chart(self, df, x_axis, y_axis):
    #     try:
    #     # try:
            # if df[x_axis].dtype in ('int64', 'float64'):
            #     if df[y_axis].dtype in ('int64', 'float64'):
            #         return f"Radar chart is not suitable for this dataset, as both {x_axis} or {y_axis} is numeric value"
            # if df[x_axis].dtype not in ('int64', 'float64'):
            #     if df[y_axis].dtype not in ('int64', 'float64'):
            #         return f"Radar chart is not suitable for this dataset, as neither {x_axis} nor {y_axis} is numeric value"
            
            # for col in df.columns:
            #     if df[col].dtype not in ('int64', 'float64'):
            #         labels=np.array(df[col])
            #         stats=df.drop(col, axis=1)
            
            # angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            # stats=np.concatenate((stats,stats[:,[0]]),axis=1)
            # angles=np.concatenate((angles,[angles[0]]))
            
            # fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
            # for i in range(len(df)):
            #     ax.plot(angles, stats[i], linewidth=2, label=labels[i])
            # ax.fill(angles, stats[i], alpha=0.4)
            # ax.set_yticklabels([])
            # ax.set_title('Radar Chart for your Question', size=16, color='green', y=1.1)
            # ax.grid(True)
        
        # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # if len(numeric_cols) < 2:
        #     return f"Radar chart is not suitable for this dataset, as dataframe does not have required numeric value columns!"
        #     # if df[x_axis].dtype not in ('int64', 'float64'):
        #     #     # numeric = df[x_axis]
        #     #     categorical_x = df[x_axis]
        #     # if df[y_axis].dtype not in ('int64', 'float64'):
        #     #     # numeric = df[x_axis]
        #     #     categorical_y = df[y_axis]
                
        # scaler = MinMaxScaler()
        # df_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
        # # fig_size = (df_normalized.shape[0] * 1.5, 6)
        # fig_size = (min(df_normalized.shape[0] * 1.5, 12), 6)

        #     # Prepare data for radar chart
        # categories = list(df_normalized.columns)
        # values = df_normalized.iloc[0].tolist()  # Using the first row for demonstration
        # values += values[:1]  # Close the radar chart

        # num_vars = len(categories)
        # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # angles += angles[:1]

        #     # Create radar chart
        # fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

        # ax.plot(angles, values, color='skyblue', linewidth=2, linestyle='solid')
        # ax.fill(angles, values, color='skyblue', alpha=0.4)

        # ax.set_yticklabels([])
        # ax.set_xticks(angles[:-1])
        # ax.set_xticklabels(categories, fontsize=12)

        # plt.title('Radar Chart for your Question', fontsize=16, fontweight='bold')
        # plt.tight_layout()
 
            # numeric = df[numeric_cols] 
            # num_vars = len(categorical)
            # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            # numeric += numeric[:1]
            # angles += angles[:1]

            # fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
            # ax.fill(angles, numeric, color='skyblue', alpha=0.4)

            # ax.set_yticklabels([])
            # ax.set_xticks(angles[:-1])
            # ax.set_xticklabels(categorical, fontsize=12) 

            # plt.title('Radar Chart for your Question', fontsize=16, fontweight='bold')
            # plt.tight_layout()
            
        # random_value = str(random.uniform(1000, 10000000))
        # filename = "./charts/" + "radar_chart_" + random_value + ".png"
        # plt.savefig(filename, format='png', transparent=True)
            
        # plt.clf()
        # plt.close()
        # # except:
        # #     return "Provided dataset is not suitable for Radar Chart!"

        # return filename
    