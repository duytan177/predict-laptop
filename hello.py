import numpy as np
from sklearn.preprocessing import LabelEncoder

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

data = np.array(['MSI', 'Ryzen 5 5500U', '5500U', 5, 6, 12, 8, 'DDR4', 512, 'SSD',
                 'AMD Radeon Graphics', 0, 'True', 15.6, 1920, 1080, 141.21199808219862, 'False'])

converted_data = isDetactString(data)

print(covertStringtoNumber(converted_data))