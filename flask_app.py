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

def getData(folder_path):
    global data, data_tmp
    data=[]
    data_tmp=[]
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename) #lấy đường dẫn các file trong thư mục
        if os.path.isfile(file_path):
            with io.open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                file_content = file_content.replace('\n','')

                file_content = file_content.split('- - ') #chia tách từng câu trả lời thành 1 phần tử của mảng
                file_content = [str.strip('- ') for str in file_content]

                data.extend(file_content)
                c= [data_pre(str) for str in file_content]
                data_tmp.extend(c)

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
    print(cosine_similarities)
    check=0 #kiểm tra có dữ liệu có độ tương đồng > 0.1
    for k in cosine_similarities:
        if k> 0.1:
            check=1
            break
    if check == 0: #nếu tất cả dữ liệu đều <0.1
        return "xin lỗi ko có dữ liệu phù hợp"
    else:
        top=[]
        if 15 < len(cosine_similarities):
            i=1
            while i<=15:
                match_index = cosine_similarities.argsort()[-i]
                if cosine_similarities[match_index]<0.1:
                    break

                top.append(documents[match_index])
                i+=1
        else:
            top= documents[:]
        return top

def search3_2(query, documents):
    vectorizer = TfidfVectorizer() #tạo một ma trận tf-idf
    tfidf_matrix = vectorizer.fit_transform(documents) #cho dữ liệu vào ma trận
    query_vec = vectorizer.transform([query]) #cho câu hỏi vào ma trận
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten() #tính độ tương đồng giữa câu hỏi và câu trả lời, hàm flatten trả về mảng 1 chiều
    best_match_index = cosine_similarities.argsort()[-1] #sắp xếp kết quả theo độ tương đồng tăng dần, lấy vị trí giá trị có độ tương đồng cao nhất (tại vị trí -1)
    top=[]
    if 7 < len(cosine_similarities):
        i=1
        while i<=7:
            match_index = cosine_similarities.argsort()[-i]
            if cosine_similarities[match_index]<0.1:
                break

            top.append(documents[match_index])
            i+=1
    else:
        top= documents[:]
    return top
def search3_3(query, documents):
    d_tmp=[]
    c= [data_pre(str) for str in documents]
    d_tmp.extend(c)

    vectorizer = TfidfVectorizer() #tạo một ma trận tf-idf
    tfidf_matrix = vectorizer.fit_transform(d_tmp) #cho dữ liệu đã qua xử lý vào ma trận
    query_vec = vectorizer.transform([query]) #cho câu hỏi vào ma trận
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten() #tính độ tương đồng giữa câu hỏi và câu trả lời, hàm flatten trả về mảng 1 chiều
    top=[]
    i=1
    while i <= len(documents):
        match_index = cosine_similarities.argsort()[-i]
        top.append(documents[match_index])
        i+=1
    return top

def search3(query, documents, doc_tmp):
    t=preprocess(query) #tiền xử lý câu hỏi input
    r=search3_1(t, data, data_tmp) #lọc câu trả lời lần 1, top15
    if type(r)!= list:
        return r
    r2=search3_2(query, r) #lọc top 7 bằng fulltext search
    r3= search3_3(t, r2)
    global data_kq
    data_kq = r3[:]
    return r3[0]
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

        folder_path=os.getcwd()+"/mysite/data"
        getData(folder_path)
        # if "là" and "gì" in question.lower():
        #     folder_path=os.getcwd()+"/data/KhaiNiem"
        #     getData(folder_path)
        #     return search3(question, data, data_tmp)
        # Nếu câu hỏi không hợp lệ hoặc vô nghĩa
        if "chào" in question.lower():
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
