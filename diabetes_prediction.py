# Import các thư viện và modules cần thiết
from dash import Dash, Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dash.exceptions import PreventUpdate
import plotly.express as px


# Load dataset
url = "https://raw.githubusercontent.com/dipkimyen/Diabetes_prediction/master/diabetes_prediction_dataset.csv"
df = pd.read_csv(url)
df_show = df

#Loại bỏ dữ liệu trùng lắp :
df = df.drop_duplicates()
#Loại bỏ dữ liệu bị null :
df = df.dropna()
#Khởi tạo đối tượng :
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

#Lọc ra dữ liệu dạng chuỗi và lấy tên các cột trong có dữ liệu dạng chuỗi :
data_string_cols = df.select_dtypes(exclude=[np.number]).columns
#Dùng phương thức fit_transform của đối tượng label_encoder đã khai báo ở trên :
#Để thực hiện việc chuẩn hoá dữ liệu dạng chuỗi thành số :
for col in data_string_cols:
  df[col] = label_encoder.fit_transform(df[col])
  
#Phân vị thứ nhất tương ứng với giá trị ở vị trí thứ 25% của mẫu dữ liệu :
q1 = df.quantile(0.25)
#Phân vị thứ ba tương ứng với giá trị ở vị trí thứ 75% của mẫu dữ liệu :
q3 = df.quantile(0.75)
#Tính IQR (Khoảng tứ phân vị) :
iqr = q3 - q1

df_hypertension	= df['hypertension'].unique()
df_heart	= df['heart_disease'].unique()
df_diabetes	= df['diabetes'].unique()

iqr['heart_disease'] = iqr['heart_disease'] + 1
iqr['hypertension'] = iqr['hypertension'] + 1
iqr['diabetes'] = iqr['diabetes'] + 1
#Tính biên dưới và biên trên :
bien_duoi = q1 - 1.5*iqr
bien_tren = q3 + 1.5*iqr

#Loại bỏ các giá trị nằm phía dưới biên dưới hoặc nằm phía trên biên trên :
#Dùng mệnh đề phũ đinh ~ any để loại bỏ :
df_no_outlier = df[~((df < bien_duoi) |
                              (df > bien_tren))
                              .any(axis = 1)]

from sklearn.model_selection import train_test_split
#Tách biến phụ thuộc ra khỏi tập dữ liệu :
X = df_no_outlier.drop('diabetes', axis=1)
y = df_no_outlier['diabetes']

#Chuẩn hóa dữ liệu tập (Dùng StandardScaler) :
scaler = StandardScaler()
for col in X.columns :
  X[[col]] = scaler.fit_transform(X[[col]])

# Huấn luyện mô hình
from sklearn.ensemble import RandomForestClassifier
#Xây dựng mô hình Random Forest (Với n_estimators là số cây quyết định trong rừng cây (mặc định là 100 cây quyết định)) :
#Mặc định Random Forest sẽ sử dụng kỹ thuật Bagging (Bootstrap Sampling) để chia tập dữ liệu huấn luyện thành các tập con có hoàn lại (With Replacement)
#With Replacement là chia tập con và có hoàn lại mục đinh là làm tăng mức đa dạng của từng cây trong rừng
#Tức là mỗi mẫu có thể xuất hiện nhiều lần và xuất trong nhiều tập con (Để không hoàn lại thì sử dụng bootstrap=False) :
#max_depth là độ sâu của cây quyết định (Mặc định là sẽ phân loại hoàn toàn được các trường hợp của tập huấn luyện) :
randomfs_model = RandomForestClassifier(n_estimators=150,max_depth = 20,class_weight='balanced',bootstrap=False,random_state=101)
#Huấn luyện mô hình trên tập huấn luyện :
randomfs_model.fit(X, y)

# Tạo Dash app
app = Dash(__name__)
server=app.server

# Định nghĩa layout
app.layout = html.Div([
    html.H1("Diabetes Prediction Dashboard", style={'text-align': 'center'}),
    
    html.H3("Show Data", style={'text-align': 'center'}),
    # Dropdown to filter data by 'diabetes' column
    dcc.Dropdown(
        id='diabetes-filter',
        options=[{'label': x, 'value': x} for x in df_show['diabetes'].unique()],
        value=df_show['diabetes'].unique()[0],
        multi=False
    ),
    
    # Scatter plot to show the relationship between 'blood_glucose_level' and 'bmi'
    dcc.Graph(id='scatter-plot'),
    
    html.H3("Prediction", style={'text-align': 'center'}),
    html.Div([
        html.Div([
            html.Label('Gender', style={'margin-bottom': '5px'}),
            dcc.Dropdown(
                id='gender',
                options=[
                    {'label': 'Female', 'value': 'Female'},
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Other', 'value': 'Other'}
                ],
                value='Female',
                style={'width': '100%'}
            ),
        ], className='row'),

        html.Div([
            html.Label('Age', style={'margin-bottom': '5px'}),
            dcc.Input(id='age', type='number', value=80, style={'width': '100%'}),
        ], className='row'),

        html.Div([
            html.Label('Hypertension', style={'margin-bottom': '5px'}),
            dcc.Input(id='hypertension', type='number', value=0, style={'width': '100%'}),
        ], className='row'),

        html.Div([
            html.Label('Heart Disease', style={'margin-bottom': '5px'}),
            dcc.Input(id='heart-disease', type='number', value=1, style={'width': '100%'}),
        ], className='row'),

        html.Div([
            html.Label('Smoking History', style={'margin-bottom': '5px'}),
            dcc.Dropdown(
                id='smoking-history',
                options=[
                    {'label': 'never', 'value': 'never'},
                    {'label': 'No Info', 'value': 'No Info'},
                    {'label': 'current', 'value': 'current'},
                    {'label': 'former', 'value': 'former'},
                    {'label': 'ever', 'value': 'ever'},
                    {'label': 'not current', 'value': 'not current'},
                ],
                value='never',
                style={'width': '100%'}
            ),
        ], className='row'),

        html.Div([
            html.Label('BMI', style={'margin-bottom': '5px'}),
            dcc.Input(id='bmi', type='number', value=25.19, style={'width': '100%'}),
        ], className='row'),

        html.Div([
            html.Label('HbA1c Level', style={'margin-bottom': '5px'}),
            dcc.Input(id='hba1c-level', type='number', value=6.6, style={'width': '100%'}),
        ], className='row'),

        html.Div([
            html.Label('Blood Glucose Level', style={'margin-bottom': '5px'}),
            dcc.Input(id='blood-glucose-level', type='number', value=140, style={'width': '100%'}),
        ], className='row'),

        html.Button('Predict', id='predict-button', n_clicks=0, style={'margin-top': '20px', 'width': '100%', 'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '4px'}),
        
        html.Div(id='prediction-output', style={'text-align': 'center', 'padding-top': '20px'}),
    ], style={'width': '50%', 'margin': 'auto'}),
])


# Callback để thực hiện dự đoán khi nhấn nút
@app.callback(
    [
        Output('prediction-output', 'children'),
        Output('scatter-plot', 'figure')
    ],
    [
        Input('predict-button', 'n_clicks')
    ],
    [
        Input('gender', 'value'),
        Input('age', 'value'),
        Input('hypertension', 'value'),
        Input('heart-disease', 'value'),
        Input('smoking-history', 'value'),
        Input('bmi', 'value'),
        Input('hba1c-level', 'value'),
        Input('blood-glucose-level', 'value'),
        Input('diabetes-filter', 'value')
    ]
)
def update_prediction_and_plot(n_clicks, gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, selected_diabetes):
    if n_clicks > 0:
        prediction_result = perform_prediction(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level)
        scatter_plot = update_scatter_plot(selected_diabetes)
        
        return f"Predicted Result: {prediction_result}" if prediction_result else "No prediction available", scatter_plot
    raise PreventUpdate



# Hàm perform_prediction để thực hiện dự đoán dựa trên thông tin đầu vào từ người dùng
def perform_prediction(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level):
    try:
        # Tạo DataFrame từ thông tin nhập vào
        data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c_level],  # Thay 'hba1c_level' bằng 'HbA1c_level'
            'blood_glucose_level': [blood_glucose_level]
        }
        df = pd.DataFrame(data)

        # Nếu muốn chuyển đổi các cột số về dạng numeric
        data_string_cols = df.select_dtypes(exclude=[np.number]).columns
        #Dùng phương thức fit_transform của đối tượng label_encoder đã khai báo ở trên :
        #Để thực hiện việc chuẩn hoá dữ liệu dạng chuỗi thành số :
        for col in data_string_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Dự đoán
        prediction = randomfs_model.predict(df)

        # Trả về kết quả dự đoán
        return prediction[0] if prediction else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def update_scatter_plot(selected_diabetes):
    filtered_df = df_show[df_show['diabetes'] == selected_diabetes]
    
    fig = px.scatter(filtered_df, x='bmi', y='blood_glucose_level', color='age',
                     title=f"Scatter plot for {selected_diabetes} diabetes")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
