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

@st.cache(ttl=24*3600)
def load_vgg():
	return hub.load('https://tfhub.dev/google/vggish/1')

@st.cache(ttl=24*3600)
def load_yamnet():
	return hub.load('https://tfhub.dev/google/yamnet/1')

# @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
# def load_covid():
# 	return joblib.load("data/example_model.h5")


modelvgg = load_vgg()
yamnet_model = load_yamnet()

@st.cache(ttl=20*60)
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
    # if f2 >  0.4 or (f2 < 0.2 and f1 > 0.2):
    if f2 >  0.1:
        mask_values = True
    else:
        mask_values = False
    # mask_feat.append(mask_values)
    # mask_feat = np.array(mask_feat)

    return mask_values
    
@st.cache(ttl=20*60)
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


@st.cache(ttl=10*60)
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
    st.title('D??? ??o??n covid qua ti???ng ho')
    st.write('''
    D??? ??o??n covid qua ti???ng ho, ???ng d???ng demo ch??? mang t??nh ch???t tham kh???o. D??? li???u ????o t???o thu th???p g???n 6000 ti???ng ho cho ????? ?????c hi???u: 98.95%, ????? nh???y: 58.33%.
    ''')
    # uploaded_file = st.file_uploader('T???i file ??m thanh c???a b???n l??n', type= ['wav'])
    audio_file = st.file_uploader("T???i file ??m thanh l??n:", type=['wav', 'mp3', 'ogg', 'm4a'])
    if audio_file is not None:
    # if audio_file is not None:
        if not os.path.exists("audio"):
            os.makedirs("audio")
        path = os.path.join("audio", audio_file.name)
        if_save_audio = save_audio(audio_file)
        if if_save_audio == 1:
            st.warning("File qu?? l???n, h??y th??? up l???i file kh??c")
        elif if_save_audio == 0:
            # extract features
            # display audio
            st.audio(audio_file, format='audio/wav', start_time=0)
            try:
                mask_X = mask_acoustic_feat(path)
                if mask_X == False:
                    st.text('Up l???i ??m thanh, ????y kh??ng ph???i ti???ng ho ho???c ti???ng ho kh??ng r?? !!!')
                # extract features
                else:
                    X = make_acoustic_feat(path)
                    
                    model = joblib.load("data/example_model.h5")
                    # model = load_covid()
                    y_predict = model.predict_proba(X)
                    y_predict = np.where(mask_X == True, y_predict, 0)
                    y_predict[:,1]
                    st.text(f'Kh??? n??ng b??? covid l??: {y_predict[:,1][0] * 100:.2f} %')
            except Exception as e:
                audio_file = None
                st.error(f"Error {e} - Kh??ng h??? tr??? ?????nh d???nh file, h??y up file .wav, .mp3, .ogg, .m4a")
        else:
            st.error("L???i kh??ng x??c ?????nh")

# if uploaded_file is not None:

        # mask_X = mask_acoustic_feat(path)
        # if mask_X == False:
        #     st.text('Up l???i ??m thanh, ????y kh??ng ph???i ti???ng ho ho???c ti???ng ho kh??ng r?? !!!')
        # # extract features
        # X = make_acoustic_feat(path)
        
        # model = joblib.load("data/example_model.h5")
        # y_predict = model.predict_proba(X)
        # y_predict = np.where(mask_X == True, y_predict, 0)
        # y_predict[:,1]
        # st.text(f'Kh??? n??ng b??? covid l??: {y_predict[:,1][0] * 100:.2f} %')
        