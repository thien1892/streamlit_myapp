import streamlit as st
from multiapp import MultiApp
from apps import home, mask_v2, yolo_img, covid # import your app modules here

st.markdown("""
# Ứng dụng Demo của **Thien1892**
""")
app = MultiApp()



# Add all your application here

# app.add_app("Nhận diện đeo khẩu trang", mask.app)
app.add_app("Nhận diện đeo khẩu trang qua camera", mask_v2.app)
app.add_app("Phát hiện vật thể với YOLO", yolo_img.app)
app.add_app("Dự đoán covid qua tiếng ho", covid.app)
app.add_app("Khám phá data", home.app)
# The main app
app.run()
