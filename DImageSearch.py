import streamlit as s
import DeepImageSearch.config as config
from DImageSearch import Load_Data, Search_Setup
image_list = Load_Data().from_folder(["/Users/adnanahmed/Downloads/archive"])
s.header("Image Recommendation App")
# Set up the search engine
st = Search_Setup(image_list=image_list,model_name='vgg19',pretrained=True,image_count= None)
st.run_index()

metadata = st.get_image_metadata_file()

st.add_images_to_index(image_list[101:110])

# Update metadata
metadata = st.get_image_metadata_file()

# Get similar images
st.get_similar_images(image_path=image_list[11],number_of_images=10)
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
import time
from PIL import ImageOps
import math
import faiss

#Getting rid of the streamlit warnings: 
s.set_option('deprecation.showPyplotGlobalUse', False)

loaded_index = faiss.read_index("/Users/adnanahmed/Downloads/ImageSearch/image_features_vectors.idx")
image_data = pd.read_pickle("/Users/adnanahmed/Downloads/ImageSearch/image_data_features.pkl")
def _extract(img):
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        # Extract features
        feature = st.model(x)
        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)

def _get_query_vector(image_path: str):
        img = Image.open(image_path)
        query_vector = _extract(img)
        return query_vector

def _search_by_vector(v, n: int):
        #self.v = v
        #self.n = n

        D, I = loaded_index.search(np.array([v], dtype=np.float32), n)
        image_paths = [os.path.abspath(path) for path in image_data.iloc[I[0]]['images_paths'].to_list()]
        image_paths = [path.replace('/ImageSearch/drive/MyDrive/', '/') for path in image_paths] 
        return image_paths

def plot_similar_images_new(image_path: str, number_of_images: int = 6):
        """
        Plots a given image and its most similar images according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image to be plotted.
        number_of_images : int, optional (default=6)
            The number of most similar images to the query image to be plotted.
        """
        input_img = Image.open(image_path)
        input_img_resized = ImageOps.fit(input_img, (896, 896), Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.title('Input Image', fontsize=18)
        plt.imshow(input_img_resized)
        s.pyplot()

        query_vector = _get_query_vector(image_path)
        img_list = _search_by_vector(query_vector, number_of_images)

        grid_size = math.ceil(math.sqrt(number_of_images))
        axes = []
        fig = plt.figure(figsize=(20, 15))
        for a in range(number_of_images):
            axes.append(fig.add_subplot(grid_size, grid_size, a + 1))
            plt.axis('off')
            img = Image.open(img_list[a])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle('Similar Result Found', fontsize=22)
s.pyplot(plot_similar_images_new(image_path = image_list[350],number_of_images=16))
