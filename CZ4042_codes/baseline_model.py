import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from tensorflow.keras.models import Model

# Retrieving of the data from Adience dataset
fold_0=pd.read_table('./Folds/original_txt_files/fold_0_data.txt')
fold_1=pd.read_table('./Folds/original_txt_files/fold_1_data.txt')
fold_2=pd.read_table('./Folds/original_txt_files/fold_2_data.txt')
fold_3=pd.read_table('./Folds/original_txt_files/fold_3_data.txt')
fold_4=pd.read_table('./Folds/original_txt_files/fold_4_data.txt')
total_data = pd.concat([fold_0, fold_1, fold_2, fold_3, fold_4], ignore_index=True)
total_data = total_data.dropna()

# Renaming image paths to match our folder structure
for i in total_data.index:
    total_data.loc[i, 'image_path'] = "./Adience/aligned/" + \
                                       total_data.loc[i, 'user_id'] + \
                                       "/landmark_aligned_face." + \
                                       str(total_data.loc[i, 'face_id']) + \
                                       "." + \
                                       total_data.loc[i, 'original_image']
    
# Obtaining subset of dataset that contains the records with the valid gender
df = total_data[total_data['gender'] != 'u'][['age', 'gender', 'x', 'y', 'dx', 'dy','image_path']]

# Binary encoding of gender field
df['gender'] = df['gender'].apply(lambda x : 0 if x == 'm' else 1)

# Helper function to parse age strings into numeric values or ranges
def parse_age(age_str):
    if ',' in age_str:  # it's a range
        return tuple(map(int, age_str.strip("()").split(',')))
    else:  # it's a single age
        return int(age_str)

age_ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 50), (51, 65), (65, 80), (81, 100)]
# Function to map an age to our desired age range
def map_age_to_range(age):
    for age_range in age_ranges:
        if isinstance(age, int):  # If age is a specific number
            average_age = age
        elif isinstance(age, tuple):  # If age is a range
            average_age = sum(age) / len(age)  # Calculate the average of the range
        
        # Now compare the average age with the age range
        if age_range[0] <= average_age <= age_range[1]:
            return age_range

    return age_ranges[4] # Default case if no other range fits

# Apply the function to parse age strings
df['age'] = df['age'].apply(parse_age)

# Map each age or age range in the DataFrame to the broader age range categories
df['age'] = df['age'].apply(map_age_to_range)

# Convert the age range to categorical data
df['age'] = pd.Categorical(df['age'], categories=age_ranges, ordered=True)

# Get the categorical mapping
age_categories = df['age'].cat.categories

# Convert these categories into unique integer codes for classification
df['age'] = df['age'].cat.codes

# Create a dictionary mapping from codes to categories
code_to_range_mapping = {code: category for code, category in enumerate(age_categories)}

# Print the mapping
print("Category code to age range mapping:")
for code, age_range in code_to_range_mapping.items():
    print(f"{code}: {age_range}")

# Minimising dataset to only required fields
df = df[['image_path','age', 'gender']]

# Helper function to extract the features in each image
def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = extract_features(df['image_path'])

# normalize the images
X = X/255.0
y_gender = np.array(df['gender'])
y_age = np.array(df['age'])
input_shape = (128, 128, 1)

inputs = Input(shape=input_shape)
# convolutional layers
conv_1 = Conv2D(96, kernel_size=(7, 7), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(3, 3)) (conv_1)
conv_2 = Conv2D(256, kernel_size=(5, 5), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(384, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(3, 3)) (conv_3)

flatten = Flatten() (maxp_3)
num_age_categories = 10
# Fully connected layers
dense_1 = Dense(512, activation='relu') (flatten)
dense_2 = Dense(512, activation='relu') (flatten)

dropout_1 = Dropout(0.5) (dense_1)
dropout_2 = Dropout(0.5) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(num_age_categories, activation='softmax', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
              optimizer='adam',
              metrics={'gender_out': 'accuracy', 'age_out': 'accuracy'})

history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=200, validation_split=0.2)