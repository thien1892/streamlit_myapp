import numpy as np
# from scipy.io import wavfile
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from apps.covid_cough import *
# from configs.example_config import Config
import os
from pydub import AudioSegment
import streamlit as st
import joblib


@st.cache
def load_vgg():
	return hub.load('https://tfhub.dev/google/vggish/1')

@st.cache
def load_yamnet():
	return hub.load('https://tfhub.dev/google/yamnet/1')

@st.cache
def load_covid():
	return joblib.load("data/example_model_gg.h5")


modelvgg = load_vgg()
yamnet_model = load_yamnet()

def mask_acoustic_feat(filename):
    # mask_feat = []
    # for filename in X.file_path.values:
    y,sr = librosa.load(filename)
    x = librosa.resample(y,sr, 16000)
    if len(x.shape)>1:
        x = np.mean(x,axis=1)
    xt, index = librosa.effects.trim(x)
    scores, embeddings, spectrogram = yamnet_model(xt)
    class_scores = tf.reduce_mean(scores, axis=0)
    f1 = np.array(class_scores)[0]
    f2 = np.array(class_scores)[42]
    if f2 >  0.2 or (f2 < 0.2 and f1 > 0.2):
        mask_values = True
    else:
        mask_values = False
    # mask_feat.append(mask_values)
    # mask_feat = np.array(mask_feat)

    return mask_values
    

def make_acoustic_feat(filename):
    feat = []
    # for filename in X.file_path.values:
    y,sr = librosa.load(filename)
    feature_values_vec = total_732_feature(y,sr, model = modelvgg)
    feature_values_vec = np.array(feature_values_vec)
    feat.append(feature_values_vec)
    feat = np.array(feat)
    feat = np.nan_to_num(feat, nan = np.nan)
    feat = np.clip(feat, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    return feat

def convert_to_wav(file_path):
    """
    This function is to convert an audio file to .wav file
    Args:
        file_path (str): paths of audio file needed to be convert to .wav file
    Returns:
        new path of .wav file
    """
    ext = file_path.split(".")[-1]
    assert ext in [
        "mp4", "mp3", "acc"], "The current API does not support handling {} files".format(ext)

    sound = AudioSegment.from_file(file_path, ext)
    wav_file_path = ".".join(file_path.split(".")[:-1]) + ".wav"
    sound.export(wav_file_path, format="wav")

    os.remove(file_path)
    return wav_file_path

def app():
    st.title('Dự đoán covid qua tiếng ho')
    st.write('''
    Dự đoán covid qua tiếng ho, ứng dụng demo chỉ mang tính chất tham khảo. dự liệu thu thập gần 6000 tiếng ho, độ đặc hiệu: 98.95%, độ nhạy: 58.33%
    ''')
    uploaded_file = st.file_uploader('Tải file âm thanh của bạn lên', type= ['wav', 'mp3','mp4', 'aac'])

    # class VideoProcessor:
    #     def recv(self, frame):
    if uploaded_file is not None:
        if uploaded_file.split(".")[-1] != 'wav':
            uploaded_file = convert_to_wav(uploaded_file)
        # file is cough or no_cough
        mask_X = mask_acoustic_feat(uploaded_file)
        if mask_X == False:
            st.text(f'Up lại âm thanh, đây không phải tiếng ho hoặc tiếng ho không rõ !!!')
        # extract features
        X = make_acoustic_feat(uploaded_file)
        
        model = load_covid()
        y_predict = model.predict_proba(X)
        y_predict = np.where(mask_X == True, y_predict, 0)
        y_predict[:,1]
        st.text(f'Khả năng bị covid là: {y_predict[:,1][0] * 100:.2f} %')
        