# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv


# # Load biến môi trường từ file .env
# load_dotenv()

# # Lấy API key từ biến môi trường
# api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# # Cấu hình API key cho Google Generative AI
# genai.configure(api_key=api_key)

# # Khởi tạo mô hình GenerativeModel
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Cấu hình đường dẫn tới VectorDB
# vector_db_path = "C:/Users/PC/Desktop/Chatbot/chatbot/src/VectorDB/vectorstores/db_faiss"

# # Đọc từ VectorDB
# def read_vectors_db():
#     # Embedding
#     embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code": True})
#     db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
#     return db

# # Tạo prompt template
# def create_prompt(template):
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     return prompt

# # Tạo simple chain
# def create_qa_chain(prompt, db):
#     def custom_llm(query, context):
#         full_prompt = prompt.format(context=context, question=query)
#         response = model.generate_content(full_prompt)
#         if "Tôi không biết" in response.text:
#             # Gọi trực tiếp mô hình Gemini để trả lời
#             response = model.generate_content(query)
#         return response.text

#     class CustomRetrievalQA:
#         def __init__(self, retriever, prompt):
#             self.retriever = retriever
#             self.prompt = prompt
        
#         def invoke(self, inputs):
#             query = inputs["query"]
#             docs = self.retriever.get_relevant_documents(query)
#             context = " ".join([doc.page_content for doc in docs])
#             answer = custom_llm(query, context)
#             links = [doc.metadata['source'] for doc in docs]
#             return {"answer": answer, "links": links}
    
#     retriever = db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=6000)
#     return CustomRetrievalQA(retriever, prompt)

# # Khởi tạo Streamlit app
# st.title('Hệ Thống Hỏi Đáp')

# # Đọc VectorDB và tạo prompt
# db = read_vectors_db()
# template = """Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn. Không thêm bớt, chỉnh sửa hoặc diễn giải lại thông tin. Nếu bạn không biết câu trả lời, hãy nói 'Tôi không biết'.

# Thông tin:
# {context}

# Câu hỏi:
# {question}
# Trả lời giống hệt dữ liệu mà không thêm bớt:"""
# prompt = create_prompt(template)
# llm_chain = create_qa_chain(prompt, db)

# # Tạo giao diện người dùng
# question = st.text_input("Nhập câu hỏi của bạn:")
# if question:
#     response = llm_chain.invoke({"query": question})
#     st.write("Trả lời:", response["answer"])
#     st.write("Liên kết tới phần liên quan trong dữ liệu:")
#     for link in response["links"]:
#         st.write(link)



import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import SentenceTransformerEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv


# Load biến môi trường từ file .env
load_dotenv()

# Lấy API key từ biến môi trường
api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# Cấu hình API key cho Google Generative AI
genai.configure(api_key=api_key)

# Khởi tạo mô hình GenerativeModel
model = genai.GenerativeModel('gemini-1.5-flash')

# Cấu hình đường dẫn tới VectorDB
collection_name = "langchain_example"
uri = "http://localhost:19530"  # Đảm bảo Milvus server đang chạy tại địa chỉ này

# Đọc từ VectorDB
def read_vectors_db():
    # Embedding
    embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code": True})
    
    # Tạo kết nối với Milvus
    db = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
        collection_name=collection_name
    )
    return db

# Tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Tạo simple chain
def create_qa_chain(prompt, db):
    def custom_llm(query, context):
        full_prompt = prompt.format(context=context, question=query)
        response = model.generate_content(full_prompt)
        if "Tôi không biết" in response.text:
            # Gọi trực tiếp mô hình Gemini để trả lời
            response = model.generate_content(query)
        return response.text

    class CustomRetrievalQA:
        def __init__(self, retriever, prompt):
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, inputs):
            query = inputs["query"]
            docs = self.retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])
            answer = custom_llm(query, context)
            links = [doc.metadata['source'] for doc in docs]
            return {"answer": answer, "links": links}
    
    retriever = db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=6000)
    return CustomRetrievalQA(retriever, prompt)

# Khởi tạo Streamlit app
st.title('Hệ Thống Hỏi Đáp')

# Đọc VectorDB và tạo prompt
db = read_vectors_db()
template = """Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn. Không thêm bớt, chỉnh sửa hoặc diễn giải lại thông tin. Nếu bạn không biết câu trả lời, hãy nói 'Tôi không biết'.

Thông tin:
{context}

Câu hỏi:
{question}
Trả lời giống hệt dữ liệu mà không thêm bớt:"""
prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, db)

# Tạo giao diện người dùng
question = st.text_input("Nhập câu hỏi của bạn:")
if question:
    response = llm_chain.invoke({"query": question})
    st.write("Trả lời:", response["answer"])
    st.write("Liên kết tới phần liên quan trong dữ liệu:")
    for link in response["links"]:
        st.write(link)
