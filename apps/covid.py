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
# import os
from datetime import datetime

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

# def convert_to_wav(file_path):
#     """
#     This function is to convert an audio file to .wav file
#     Args:
#         file_path (str): paths of audio file needed to be convert to .wav file
#     Returns:
#         new path of .wav file
#     """
#     ext = file_path.split(".")[-1]
#     assert ext in [
#         "mp4", "mp3", "acc"], "The current API does not support handling {} files".format(ext)

#     sound = AudioSegment.from_file(file_path, ext)
#     wav_file_path = ".".join(file_path.split(".")[:-1]) + ".wav"
#     sound.export(wav_file_path, format="wav")

#     os.remove(file_path)
#     return wav_file_path

# @st.cache
def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0

def app():
    st.title('Dự đoán covid qua tiếng ho')
    st.write('''
    Dự đoán covid qua tiếng ho, ứng dụng demo chỉ mang tính chất tham khảo. dự liệu thu thập gần 6000 tiếng ho, độ đặc hiệu: 98.95%, độ nhạy: 58.33%
    ''')
    # uploaded_file = st.file_uploader('Tải file âm thanh của bạn lên', type= ['wav'])
    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
    if audio_file is not None:
    # if audio_file is not None:
        if not os.path.exists("audio"):
            os.makedirs("audio")
        path = os.path.join("audio", audio_file.name)
        if_save_audio = save_audio(audio_file)
        if if_save_audio == 1:
            st.warning("File size is too large. Try another file.")
        elif if_save_audio == 0:
            # extract features
            # display audio
            st.audio(audio_file, format='audio/wav', start_time=0)
            # try:
            #     wav, sr = librosa.load(path, sr=44100)
            #     # Xdb = get_melspec(path)[1]
            #     # mfccs = librosa.feature.mfcc(wav, sr=sr)
            #     # # display audio
            #     # st.audio(audio_file, format='audio/wav', start_time=0)
            # except Exception as e:
            #     audio_file = None
            #     st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
        else:
            st.error("Unknown error")

# if uploaded_file is not None:

        mask_X = mask_acoustic_feat(path)
        if mask_X == False:
            st.text('Up lại âm thanh, đây không phải tiếng ho hoặc tiếng ho không rõ !!!')
        # extract features
        X = make_acoustic_feat(path)
        
        model = load_covid()
        y_predict = model.predict_proba(X)
        y_predict = np.where(mask_X == True, y_predict, 0)
        y_predict[:,1]
        st.text(f'Khả năng bị covid là: {y_predict[:,1][0] * 100:.2f} %')
        