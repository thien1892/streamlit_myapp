# import streamlit as st
# import pandas as pd
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# def app():
#     st.title('Khám phá data')
#     st.write('''
#     Ứng dụng này giúp bạn trực quan dữ liệu của bạn. Khai phá các mặt tổng quan của dữ liệu.
#     ''')
#     uploaded_file = st.file_uploader('Up load data của bạn', type= ['csv', 'xlsx'])
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         pr = df.profile_report()
#         st_profile_report(pr)

