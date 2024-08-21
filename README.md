# chatbot

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization
 - Để thực hiện mã nguồn trên đầu tiên cần tải về theo lệnh:
```
git clone https://github.com/NguyenHuy31072002/Chatbot_with_Data.git
```
- Truy cập vào Chatbot_with_Data tạo một môi trường để chạy
```
python -m venv .venv
```
- Truy cập vào môi trường
```
.venv\Scripts\activate
```
- Đi đến dường dẫn chứa file requirements.txt thực hiện chạy lệnh
```
pip install -r requirements.txt
```
- Truy cập vào folder VectorDB và vào file prepare_vector_db.py
```
cd src/VectorDB/
```
- Sửa đường dẫn 
```
txt_data_path = "C:/Users/PC/Desktop/Chatbot/chatbot/data" thành đường dẫn đến folder data chứa các file txt.
```
- Thực hiện chạy file prepare_vector_db.py
```
python prepare_vector_db.py
```
- Sau khi chạy xong quay chuyển đường dẫn trong terminal sang đường dẫn chứa file chatbot.py
- Vào file chatbot.py sửa lại đường dẫn đến vectorstores/db_faiss ở dòng 24 của mã nguồn được tạo ra từ quá trình chạy file prepare_vector_db.py
- Rồi chạy chương trình
```
streamlit run chatbot.py
```
Sau khi chạy nó sẽ tạo ra giao diện như sau và chúng ta tiến hành hỏi đáp
![image](https://github.com/user-attachments/assets/017addac-dfc3-4882-914d-f40ef06621ba)

Video demo hỏi đáp 
[![Watch the video](https://img.youtube.com/vi/sNtAHX77O5g/0.jpg)](https://www.youtube.com/watch?v=sNtAHX77O5g)



