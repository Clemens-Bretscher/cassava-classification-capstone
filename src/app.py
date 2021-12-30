from pandas.core import indexing
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import base64

path = '/Users/paulosgidyelew/Desktop/Capstone_Project/data/soup/'
data_1 = pd.read_csv('/Users/paulosgidyelew/Desktop/cassava-classification-capstone/data/train.csv')
data_2 = pd.read_csv(path+'soup_3.csv')
data_info = pd.read_csv('/Users/paulosgidyelew/Desktop/cassava-classification-capstone/data/data_info.csv')

def about():

    st.markdown('''<h1 style='text-align: justify; font-size: 25px;color:green'><b>About Classava.io</b></h1>''', unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify; font-size: 20px;color:black'>
        This app was developed in association with the
        <mark>National Crops Resources Research Institute (NaCRRI) and 
        the AI lab in Makarere University, Kampala</mark>.
        As the second-largest provider of carbohydrates in Africa, 
        cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. 
        At least 80 percent of household farms in Sub-Saharan Africa grow this starchy root, 
        but viral diseases are major sources of poor yields. With the help of data science, 
        it may be possible to identify common diseases so they can be treated.</p>''', unsafe_allow_html=True)
    return
def statistics():

    st.markdown('''<h1 style='font-size:25px;color: green'><b>Cassava Leaf Classes and Data Distribution</b></h1>''', unsafe_allow_html=True)

    st.markdown('''<p style='text-align: justify; font-size: 20px;'>
        As the second-largest provider of carbohydrates in Africa, 
        cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. 
        At least 80 percent of household farms in Sub-Saharan Africa grow this starchy root, 
        but viral diseases are major sources of poor yields. 
        About 5,656 images of cassava plants were collected from Ugandan farmers.
        Experts grouped the images in 5 categories: 
        Healthy, Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD),
        Cassava Green Mottle (CGM), and Cassava Mosaic Disease (CMD) With this data an artificial neural network was created that automatically
        classifies an image of cassava in these five different categories.</p>''', unsafe_allow_html=True)
    
    with st.expander('Data Count'):
        st.table(data_info.assign(index='Count').set_index('index'))
    
    st.markdown("<h2 style='text-align: left; font-size: 20px; color: green'>Cassava Leaf Classification Images</h2>", unsafe_allow_html=True)

    if st.checkbox('Show Images'):
        st.image('images/leafs.png')
        
    st.markdown("<h2 style='text-align: left; font-size: 20px; color: green'>Data Distribution of the Classes</h2>", unsafe_allow_html=True)
    if st.checkbox('Show Plot'):
        fig = plt.figure()
        sns.countplot(data = data_1, x=data_1.label_name)
        plt.title('Cassava Disease Category')
        plt.xlabel('Count')
        plt.ylabel(' ')
        st.pyplot(fig)  
    return 

def customer():
    
    st.markdown('''<p style='text-align: justify; font-size: 25px;color:green'><b>Overview: Uganda</b></p>''', unsafe_allow_html=True)
  
    st.markdown('''<p style='text-align: justify; font-size: 20px;'>Uganda is a landlocked nation 
        located in East Africa with population about 20 million.
        Over 25 percent of the land is considered suitable for agriculture, which is much higher than the 
        average for sub-Saharan Africa (6.4 percent).
        Agriculture accounts for more than 60 percent, 98 percent of export earnings and over 40 percent of 
        government revenue. Farming is labour intensive, with women and children providing 60–80 percent of 
        the labour and crops are cultivated both as cash and food security crops.</p>''', unsafe_allow_html=True)
  
    with st.expander('Show More'):
        st.markdown('''<h2 style='text-align: justify; font-size: 25px;color:green'><b>Cassava Plant</b></h2>''', unsafe_allow_html=True)
        st.markdown('''<p style='text-align: justify; font-size: 18px;'><b>Cassava</b> was introduced to Uganda 
        through Tanzania by Arab traders between 1862 and 1875 (Langlands. 1972). 
        Following its initial introduction, cassava quickly spread to other areas of Uganda. 
        It is one of the <mark>most important food crops in Uganda</mark>. 
        It ranks second to <b>bananas</b> in terms of <b>area occupied</b>, <b>total production</b> and <b>per capita consumption</b>, 
        respectively (Otim-Nape, 1990). It is regarded as the most important staple crop.</p>''', unsafe_allow_html=True)
    st.markdown('''<h2 style='text-align: justify; font-size: 18px;color:green'><b>Quantitative food demand trends for cassava (1981–1994)</b></h2>''', unsafe_allow_html=True)
    with st.expander('Show Table'):
        st.dataframe(data_2)
    if st.checkbox('Show Plot'):
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0] =sns.pointplot(data=data_2, x=data_2['Year'], y=data_2['Total demand (million tonnes)'], color='r', ax=ax[0])
        ax[1]=sns.pointplot(data=data_2,y=data_2['Industry use (tonnes)'][:6],x=data_2['Year'],color='y',ax=ax[1],markers='D')
        ax[1]=sns.pointplot(data=data_2,y=data_2['Industry use (tonnes)'][5:],x=data_2['Year'],color='b',ax=ax[1], linestyles='--')
        ax[0].set_title('Cassava Food Demand')
        ax[1].set_title('Cassava Industry Use')
        ax[0].set_xticklabels([1980,'',1982,'',1984,'',1986,'',1988,'',1990,'',1992,'',1994])
        ax[1].set_xticklabels([1980,'',1982,'',1984,'',1986,'',1988,'',1990,'',1992,'',1994])
        ax[0].set_xlabel('Year')
        ax[1].set_xlabel('Year')
        st.pyplot(fig)
    return


# class Model:
#     def __init__(self,image) -> None:
#         self.image = image
#         pass

def predict(image,label):
    model = load_model(r'/Users/paulosgidyelew/Desktop/cassava-classification-capstone/saved_model/model_cv')
    test_img = image.resize((224,224))
    test_img = preprocessing.image.img_to_array(test_img).astype(np.float32)/255
    test_img = np.expand_dims(test_img,axis=0)
    probabilities = model.predict(test_img)
    results = np.argmax(probabilities,axis=1)
    
    if st.button('Process'):

        latest_iteration = st.empty()
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1)
            latest_iteration.text(f'{0 + percent_complete + 1}%')
        latest_iteration.empty()
        my_bar.empty()
        st.image(image)
        if results == 0:
            st.markdown("<p style='text-align: left; font-size: 18px; color:red'>Image Predicted As Class: <b>cbb</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left; font-size: 18px; color: green'>True Label of Image Is: <b>{label}</b></p>", unsafe_allow_html=True)     
        elif results == 1:
            st.markdown("<p style='text-align: left; font-size: 18px; color:red'>Image Predicted As Class: <b>cbsd</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left; font-size: 18px; color: green'>True Label of Image Is: <b>{label}</b></p>", unsafe_allow_html=True)
        elif results == 2:
            st.markdown("<p style='text-align: left; font-size: 18px; color:red'>Image Predicted As Class: <b>cgm</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left; font-size: 18px; color: green'>True Label of Image Is: <b>{label}</b></p>", unsafe_allow_html=True)
        elif results == 3:
            st.markdown("<p style='text-align: left; font-size: 18px; color:red'>Image Predicted As Class: <b>cmd</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left; font-size: 18px; color: green'>True Label of Image Is: <b>{label}</b></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: left; font-size: 18px; color:red'>Image Predicted As Class: <b>healthy</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left; font-size: 18px; color: green'>True Label of Image Is: <b>{label}</b></p>", unsafe_allow_html=True)
     
    return

def upload():
    options = ['About','Stakeholder','Stats','Model']
    choice = st.sidebar.radio('Select Options', options)
    if choice == 'Model':
        upload_file = st.file_uploader('Upload Image',type=["png", "jpeg", "bmp", "pdf", "ppm", "gif", "tif", "jpg"])
        try:
            if upload_file is not None:
                image = Image.open(upload_file)
                try:
                    true_label = upload_file.name.split('-')[1]
                    predict(image,true_label)
                except IndexError as e:
                    st.write(f'{e}, please save your img as " xxx-class_name-xxx.jpg"')
        except ValueError:
            st.write('Error: invalid entry!')
        return
    elif choice == 'Stakeholder':
        customer()
    elif choice == 'Stats':
        statistics()
    else:
        about()

if __name__=='__main__':
    upload()
