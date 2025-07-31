import pandas as pd
import json
import numpy as np
import os

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# read the parquet file
# Benhao: download from here: https://huggingface.co/datasets/BUAADreamer/llava-en-zh-2k
parquet_path = "TODO"
df = pd.read_parquet(parquet_path)

# Shuffle the DataFrame for random splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define split sizes
train_size = 800
val_size = 200

# Adjust split sizes if the dataframe is smaller than 1000
if len(df) < train_size + val_size:
    print(f"Warning: Dataset has only {len(df)} rows. Using an 80/20 split instead.")
    train_size = int(len(df) * 0.8)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

print(f"Splitting into {len(train_df)} training samples and {len(val_df)} validation samples.")

# Create a directory to store images
output_image_dir = 'llava_images'
os.makedirs(output_image_dir, exist_ok=True)

def save_images_and_update_path(row):
    images = row['images']
    if images is not None:
        processed_images = []
        for i, img_dict in enumerate(images):
            if isinstance(img_dict, dict) and 'bytes' in img_dict and isinstance(img_dict['bytes'], bytes):
                # copy dict to avoid modifying original
                new_img_dict = img_dict.copy()
                image_bytes = new_img_dict.pop('bytes')

                # Define image path
                image_filename = f"image_{row.name}_{i}.png"
                image_path = os.path.join(output_image_dir, image_filename)
                
                # Save image
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Add path to dict
                new_img_dict['path'] = image_path
                processed_images.append(new_img_dict)
            else:
                processed_images.append(img_dict)
        row['images'] = processed_images
    return row

df = df.apply(save_images_and_update_path, axis=1)

# Process and save the training set
train_data = df.iloc[:train_size]
train_dumped = json.dumps(train_data.to_dict(orient='records'), indent=4, ensure_ascii=False, cls=NumpyEncoder)
with open('train_llava.json', 'w', encoding='utf-8') as f:
    f.write(train_dumped)

# Process and save the validation set
val_data = df.iloc[train_size:]
val_dumped = json.dumps(val_data.to_dict(orient='records'), indent=4, ensure_ascii=False, cls=NumpyEncoder)
with open('val_llava.json', 'w', encoding='utf-8') as f:
    f.write(val_dumped)

print(f"Successfully created train_llava.json and val_llava.json")