import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Tạo một đối tượng LabelEncoder
label_encoder = LabelEncoder()
# Load the model
model_path = './Model/linear_regression_model.pkl'
pipe = joblib.load(model_path)

# Load the dataframe
df = pd.read_csv("data/laptop_cleaned2.csv")

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Brand'].unique())

# Processor_name
Processor_name = st.text_input('Enter Processor_name')

# Processor_variant
Processor_variant = st.text_input('Enter Processor_variant')

# Processor_gen
Processor_gen = st.number_input("Enter Processor_gen", min_value=0, max_value=20, step=1)

# Core_per_processor
Core_per_processor = st.number_input("Core_per_processor", min_value=1, max_value=64, step=1)

# Threads
Threads = st.selectbox("Threads", [4, 6, 8, 12, 16, 24])

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64, 128])

# RAM type
ram_type = st.selectbox("RAM type", df["RAM_type"].unique())

# Disk
disk = st.selectbox('Disk (in GB)', [0, 64, 128, 256, 512, 1024, 2048])

# Disk type
disk_type = st.selectbox("Type of disk", df["Storage_type"].unique())

# Graphics_name
Graphics_name = st.text_input('Graphics_name of the Laptop')

# Graphics_GB
Graphics_GB = st.number_input("Graphics_GB of the laptop", min_value=0, max_value=16, step=1)

# Graphics_integreted
Graphics_integreted = st.selectbox('Graphics integreted', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size', min_value=10.0, max_value=20.0, step=0.1)

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])


def isDetactString(data):
    # Tạo một mảng mới để lưu trữ dữ liệu đã chuyển đổi
    converted_data = []

    # Lặp qua từng phần tử trong mảng
    for item in data:
        # Nếu phần tử là một chuỗi số, chuyển đổi nó thành số nguyên hoặc số thực
        if isinstance(item, str) and item.replace('.', '', 1).isdigit():  # Kiểm tra xem chuỗi có phải là số không
            if '.' in item:  # Nếu là số thực
                converted_data.append(float(item))
            else:  # Nếu là số nguyên
                converted_data.append(int(item))
        # Nếu không, giữ nguyên phần tử và thêm vào mảng mới
        else:
            converted_data.append(item)

    return converted_data

def covertStringtoNumber(converted_data):
    # Find the indices of string elements in the array
    str_indices = [i for i, item in enumerate(converted_data) if isinstance(item, str)]

    # Extract the string elements
    str_data = [converted_data[i] for i in str_indices]

    # Initialize LabelEncoder
    encoder = LabelEncoder()

    # Fit and transform the string data to numerical labels
    encoded_data = encoder.fit_transform(str_data)

    # Replace the original string elements with the encoded numerical labels
    for i, index in enumerate(str_indices):
        converted_data[index] = encoded_data[i]

    return converted_data


if st.button('Predict Price'):


    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Convert categorical inputs to appropriate numerical values
    touchscreen = 1 if touchscreen == 'Yes' else 0
    Graphics_integreted = 1 if Graphics_integreted == 'Yes' else 0

    # Create the feature array
    query = np.array([company, Processor_name, Processor_variant, Processor_gen,
                      Core_per_processor, Threads, ram, ram_type, disk, disk_type,
                      Graphics_name, Graphics_GB, Graphics_integreted, screen_size, X_res, Y_res, ppi, touchscreen])

    # Reshape query to a 2D array
    # query = query.reshape(1, -1)
    query = covertStringtoNumber(isDetactString(query))
    st.title(query)

    # # Convert to DataFrame to match the pipeline's expected input format
    query_df = pd.DataFrame([query], columns=['Brand', 'Processor_name', 'Processor_variant', 'Processor_gen',
                                              'Core_per_processor', 'Threads', 'RAM_GB', 'RAM_type', 'Storage_capacity_GB', 'Storage_type',
                                              'Graphics_name', 'Graphics_GB', 'Graphics_integreted', 'Display_size_inches',
                                              'Horizontal_pixel', 'Vertical_pixel', 'ppi', 'Touch_screen'])

    predicted_price = ""
    # Predict the price
    try:
        st.title("Hello")
        st.title(query_df)
        predicted_price = pipe.predict(query_df)
        # st.title(f"The predicted price of this configuration is ${int(np.exp(predicted_price))}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

