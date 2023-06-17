from flask import Flask, render_template, request
from textblob import TextBlob
import os
import io
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
app = Flask(__name__)

#dữ liệu câu trả lời
data=[]
data_tmp=[]
data_kq=[]
with io.open('mysite/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.readlines()
    stopwords = stopwords[-1].split()
f.close()
def data_pre(cau_tra_loi):
    cau_co_1_stopword = cau_tra_loi
    for w in stopwords:
        if w in cau_tra_loi:
            check=w
            word = word_tokenize(cau_tra_loi.lower())
            cau_bo_stopword=[k for k in word if not k in stopwords]
            cau_bo_stopword= " ".join(str(k) for k in cau_bo_stopword)
            cau_co_1_stopword=cau_bo_stopword +" "+ check
    return cau_co_1_stopword

def getData(folder_path, doc, doc_tmp):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename) #lấy đường dẫn các file trong thư mục
        if os.path.isfile(file_path):
            with io.open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                file_content = file_content.replace('\n','')
                file_content = file_content.split('- - ') #chia tách từng câu trả lời thành 1 phần tử của mảng
                file_content = [str.strip('- ') for str in file_content]

                doc.extend(file_content)
                c= [data_pre(str) for str in file_content]
                doc_tmp.extend(c)

folder_path=os.getcwd()+"/mysite/data/BoCauTraLoi"
getData(folder_path, data, data_tmp)
d2=[]
d2_tmp=[]
folder_path=os.getcwd()+"/mysite/data/KhaiNiem"
getData(folder_path, d2, d2_tmp)
#tiền xử lý câu hỏi input
def preprocess(text):
    stop_words = set(stopwords)
    word_tokens = word_tokenize(text.lower()) #chia tách keyword
    filtered_tokens = [w for w in word_tokens if not w in stop_words and w not in string.punctuation] #bỏ các từ khóa có trong stopword và các ký tự đặc biệt, dấu câu
    return " ".join(filtered_tokens)

def search3_1(query, documents, doc_tmp):
    vectorizer = TfidfVectorizer() #tạo một ma trận tf-idf
    tfidf_matrix = vectorizer.fit_transform(doc_tmp) #cho dữ liệu đã qua xử lý vào ma trận
    query_vec = vectorizer.transform([query]) #cho câu hỏi vào ma trận
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten() #tính độ tương đồng giữa câu hỏi và câu trả lời, hàm flatten trả về mảng 1 chiều
    top=[]
    i=1
    while i<=15:
        match_index = cosine_similarities.argsort()[-i]
        top.append(documents[match_index])
        i+=1
    return top
def search3_2(query, documents):
    vectorizer = TfidfVectorizer() #tạo một ma trận tf-idf
    tfidf_matrix = vectorizer.fit_transform(documents) #cho dữ liệu vào ma trận
    query_vec = vectorizer.transform([query]) #cho câu hỏi vào ma trận
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten() #tính độ tương đồng giữa câu hỏi và câu trả lời, hàm flatten trả về mảng 1 chiều
    top=[]
    i=1
    while i<=7:
        match_index = cosine_similarities.argsort()[-i]
        if cosine_similarities[match_index]<0.3:
            break
        top.append(documents[match_index])
        i+=1
    return top
def search3(query, documents, doc_tmp):
    t=preprocess(query) #tiền xử lý câu hỏi input
    r=search3_1(t, documents, doc_tmp) #lọc câu trả lời lần 1, top15
    r2=search3_2(query, r) #lọc top 7 bằng fulltext search
    global data_kq
    data_kq = r2[:]
    if len(r2) == 0:
        return "Xin lỗi Chatbot không đủ dữ liệu để trả lời câu hỏi này."
    return r2[0]

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get2")
def get_bot_response():
    while True:
        question = request.args.get('smsg')
        # Sử dụng TextBlob để phân tích câu hỏi và loại bỏ các từ không cần thiết
        blob = TextBlob(question)
        # nếu như trong câu có "DT", "IN", hay "Belo" thì textBlob sẽ bỏ ra, rồi nếu mà câu đó == 0 thì nó xuất ra Xin lỗi....
        words = [word for word, tag in blob.tags if tag != 'DT' and tag != 'IN' and tag != 'BELO']
        if "là gì" in question.lower():
            return search3(question, d2, d2_tmp)
        # Nếu câu hỏi không hợp lệ hoặc vô nghĩa
        elif "chào" in question.lower():
            return search3(question, data, data_tmp)
        elif len(words) < 2:
            return str("Xin lỗi, tôi không hiểu bạn muốn hỏi gì")
        else:
            return search3(question, data, data_tmp)
@app.route("/get3")
def get_bot_reresponse():
    while True:
        if data_kq:
            data_kq.pop(0)
            return data_kq[0]
        else:
            return "Xin lỗi Chatbot không đủ dữ liệu để trả lời câu hỏi này."
