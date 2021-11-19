import streamlit as st
from multiapp import MultiApp
from apps import home, mask, mask_camera # import your app modules here

st.markdown("""
# Ứng dụng Teck with Thiện

Đây là trang web demo một số ứng dụng của **thien1892**.

""")
app = MultiApp()



# Add all your application here
app.add_app("Khám phá data", home.app)
# app.add_app("Nhận diện đeo khẩu trang", mask.app)
app.add_app("Nhận diện đeo khẩu trang qua camera", mask_camera.app)
# The main app
app.run()
