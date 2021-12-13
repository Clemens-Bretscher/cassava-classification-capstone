from six import print_
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center;'>Classava</h1>", unsafe_allow_html=True)
#st.title('Classava')
#st.header('Classava: Capstone Project')








# class Model:
#     def __init__(self,image) -> None:
#         self.image = image
#         pass

def about():
    about_button = st.expander('About 👉')
    with about_button:
        st.markdown("<p style='text-align: center; font-size: 22px;'>Diseases of Cassava</p>", unsafe_allow_html=True)
        st.markdown('''
        As the second-largest provider of carbohydrates in Africa, 
        cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. 
        At least 80 percent of household farms in Sub-Saharan Africa grow this starchy root, 
        but viral diseases are major sources of poor yields. With the help of data science, 
        it may be possible to identify common diseases so they can be treated.
        ''')
        if st.checkbox('Images'):
            st.image('images/leafs.png')
        data = pd.read_csv('data/train.csv')
        if st.checkbox('Dataset'):
            fig = plt.figure()
            sns.countplot(data = data, y=data.label_name)
            plt.title('Cassava Disease Category')
            plt.xlabel('Count')
            plt.ylabel(' ')
            st.pyplot(fig)

    return 

def predict(image):
    model = load_model(r'saved_models/my_model')

    test_img = image.resize((380,380))
    test_img = preprocessing.image.img_to_array(test_img).astype(np.float32)/255
    test_img = np.expand_dims(test_img,axis=0)

    probabilities = model.predict(test_img)
    prediction = np.argmax(probabilities,axis=1)
    
    return
  
#st.selectbox('Select a file', "images/Coat_of_arms_of_Uganda.png")
menu = ["Image","Dataset","DocumentFiles","About"]
#choice = st.sidebar.selectbox("Menu",menu)
#choice = st.sidebar("Menu",menu)

def upload():
    
    upload_file = st.file_uploader('Upload Your Image File',type=['jpg','png','jpeg','bmp','gif'])
    if upload_file is not None:
        #lets use PIL to open the uploaded file
        col1,col2,col3 = st.columns([1,4,1])
        with col1:
            st.write("")
        with col2:   
            image = Image.open(upload_file)
        #lets show the image:
            st.image(image)

            # results = predict(image)

            # if results == 0:
            #     st.write(f'Uploaded image is class: CBB')
            # elif results == 1:
            #     st.write(f'Uploaded image is class: CBSD')
            # elif results == 2:
            #     st.write(f'Uploaded image is class: CGM')
            # elif results == 3:
            #     st.write(f'Uploaded image is class: CMD')
            # else:
            #     st.write(f'Uploaded image is class: Healthy')

            st.markdown("<p style='text-align: center; font-size: 22px;'>The prediction is <b>...CMD...</b></p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 22px;'>The certainty is <b>...58%...</b></p>", unsafe_allow_html=True)

            

        
        with col3:
            st.write("")



        #lets call the predict function



        



    return



#st.image("images/Coat_of_arms_of_Uganda.png", width=200)
#image2 = Image.open('images/Coat_of_arms_of_Uganda.png')
#st.image(image2, caption=None, width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")        
#st.markdown("<img src="pictures/Create_new-branch.png"
#     alt="Create_branch]" width=400>"
#st.markdown('<img src="pictures/Create_new-branch.png">', unsafe_allow_html=True)


def space():
    st.write('')
def uganda():
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write("")
        st.write("")
        #st.markdown("<h1 style='text-align: center;'>dljsdfjn </h1>", unsafe_allow_html=True)

    with col2:
        st.image("images/Coat_of_arms_of_Uganda.png", width=200)
    with col3:
        st.write("")
        #st.markdown("<h1 style='text-align: center;'> dljsdfjn</h1>", unsafe_allow_html=True)

# if st.button(label='Upload your image here'):
#     upload()



if __name__=='__main__':
    upload()
    space()
    space()
    space()
    uganda()
    about()
