import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from utils import TextProcessing as tpr
from utils import evaluation
from utils import RecommendationSystem as rs
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from wordcloud import WordCloud as wc

import surprise
from surprise.model_selection import cross_validate
from surprise.model_selection.validation import cross_validate
from surprise import Reader, Dataset, SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, CoClustering, BaselineOnly

from gensim import corpora, models, similarities

label_encoder = LabelEncoder()

# Load các emoji biểu cảm thường gặp
emoji_dict = tpr.load_emojicon(file_path='files/emojicon.txt')
teen_dict = tpr.load_teencode(file_path='files/teencode.txt')
translate_dict = tpr.load_translation(file_path='files/english-vnmese.txt')
stopwords_lst = tpr.load_stopwords(file_path='files/vietnamese-stopwords.txt')
wrong_lst = tpr.load_words(file_path='files/wrong-word.txt')
# Nạp các từ ngữ tiêu cực sau khi đã xử lý bằng tay
positive_words_lst: list = tpr.load_words(file_path='files/hasaki_positive_words.txt')
# Nạp các từ ngữ tiêu cực sau khi đã xử lý bằng tay
negative_words_lst: list = tpr.load_words(file_path='files/hasaki_negative_words.txt')

# Load model và TF-IDF vectorizer
@st.cache_resource
def load_model_and_tfidf():
    model = evaluation.Load_Object('models/SVD_algo.pkl')
    tfidf_vectorizer = evaluation.Load_Object('models/tfidf_content_base.pkl')
    index = evaluation.Load_Object('models/index_content_base.pkl')
    dictionary = evaluation.Load_Object('models/dictionary_content_base.pkl')
    return model, tfidf_vectorizer, index, dictionary

# Load model và tfidf
surprise_model, gensim_tfidf, gensim_index, gensim_dictionary = load_model_and_tfidf()

# Giao diện phần 'Tải dữ liệu lên hệ thống'
st.sidebar.write('# :briefcase: Đồ án tốt nghiệp K299')
st.sidebar.write('### :scroll: Project 2: Recommender system')
# st.sidebar.write('### Hệ thống gợi ý sản phẩm')
st.sidebar.write('### Menu:')
info_options = st.sidebar.radio(
    ':gear: Các chức năng:', 
    options=['Tổng quan về hệ thống', 'Tải dữ liệu lên hệ thống', 'Tổng quan về dataset', 'Thông tin về sản phẩm', 'Hệ thống gợi ý sản phẩm']
)
st.sidebar.write('-'*3)
st.sidebar.write('### :left_speech_bubble: Giảng viên hướng dẫn:')
st.sidebar.write('### :female-teacher: Thạc Sỹ Khuất Thùy Phương')

st.sidebar.write('-'*3)
st.sidebar.write('#### Nhóm cùng thực hiện:')
st.sidebar.write(' :boy: Nguyễn Minh Trí')
st.sidebar.write(' :boy: Võ Huy Quốc')
st.sidebar.write(' :boy: Phan Trần Minh Khuê')
st.sidebar.write('-'*3)
st.sidebar.write('#### :clock830: Thời gian báo cáo:')
st.sidebar.write(':spiral_calendar_pad: 14/12/2024')

## Kiểm tra dữ liệu đã upload trước đó
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None  # Khởi tạo nếu chưa có dữ liệu
    
## Các bước thực hiện
if info_options == 'Tổng quan về hệ thống':
    st.image('img/hasaki_logo.png', use_column_width=True)
    general_info_tabs = st.tabs(['Business Objective', 'Triển khai hệ thống'])
    with general_info_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.image('img/think_man.png', use_column_width=True)
        with col2:
            st.write('- HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn...”')
            st.write('- Khách hàng có thể vào website hasaki.vn để tìm kiếm, lựa chọn sản phẩm, xem các đánh giá/nhận xét và đặt mua sản phẩm.')
            st.write('- Giả sử rằng HASAKI.VN chưa triển khai hệ thống Recommender System để giúp đề xuất sản phẩm phù hợp với khách hàng/users.')
    with general_info_tabs[1]:
        st.header('Quy trình triển khai hệ thống')
        st.header('Giải pháp đề xuất')
        st.write('- Recommendation System: là hệ thống đề xuất/khuyến nghị được thiết kế để tự động đưa ra gợi ý cho người dùng về các mục hoặc thông tin mà họ có thể quan tâm. Sự đề xuất này dựa trên việc phân tích và đánh giá dữ liệu lịch sử của người dùng, như lịch sử mua sắm, đánh giá, hay hành vi trực tuyến.')
        st.write('- Recommendation System sử dụng các thuật toán và mô hình học máy để tối ưu hóa quá trình đề xuất, tạo ra trải nghiệm cá nhân hóa cho từng người dùng.')
        st.write('''Lợi ích của Recommendation System:
- Tăng hiệu suất bán hàng: hệ thống đề xuất giúp tăng cơ hội bán hàng bằng cách đưa ra các sản phẩm phù hợp với sở thích và nhu cầu của người dùng.
- Cải thiện trải nghiệm người dùng: Người dùng nhận được đề xuất cá nhân hóa, giúp họ khám phá nội dung mới một cách thuận lợi.
- Giữ được sự tương tác lâu hơn: Cung cấp nội dung hoặc sản phẩm phù hợp có thể giữ chân người dùng và thúc đẩy tương tác liên tục trên các nền tảng trực tuyến.''')
        st.image('img/Gioi_thieu_proj2.PNG', use_column_width=True)

## Xem dữ liệu đã upload lên, đưa dữ liệu vào session để sử dụng lại được
if info_options == 'Tải dữ liệu lên hệ thống':
    st.image('img/hasaki_logo.png', use_column_width=True)
    st.header('Tải dữ liệu đầu vào')

    # Chỉ hiện nút tải file nếu chưa có dữ liệu
    if st.session_state['uploaded_data'] is None:
        uploaded_file = st.file_uploader('Upload file CSV chứa dữ liệu:', type='csv')

        if uploaded_file is not None:
            # Đọc file CSV và lưu vào session_state
            data = pd.read_csv(uploaded_file)
            data = data.drop(data[data['ngay_binh_luan'] == '30/11/-0001'].index)
            data['ngay_binh_luan'] = pd.to_datetime(data['ngay_binh_luan'], format='%Y-%m-%d')
            data['quarter'] = data['ngay_binh_luan'].dt.to_period('Q').astype(str)
            st.session_state['uploaded_data'] = data
            st.write('-'*3)
            st.success('Dữ liệu đã được tải lên!')
            st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].head(5))
            st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].tail(5))  
    else:
        st.info('Dữ liệu đã được tải lên trước đó.')
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].head(5))
        st.dataframe(data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'so_sao', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gia_ban']].tail(5))
# Giao diện phần 'Tổng quan về dataset'
if info_options == 'Tổng quan về dataset':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset chưa được tải lên')
    else:
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        # Tạo đồ thị đếm số sao
        st.write('### Phân bổ số sao đánh giá')
        fig, ax = plt.subplots(figsize=(10, 6))  # Khởi tạo figure và axis cho Seaborn
        sns.countplot(data=data, x='so_sao', hue='so_sao', palette='tab10', ax=ax)  # Tạo biểu đồ trên ax
        # Thêm số count trên cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Tùy chỉnh đồ thị
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Xoay nhãn trục X
        ax.set_title('Số sao phân bổ')  # Tiêu đề
        plt.tight_layout()  # Đảm bảo bố cục không bị cắt
        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)
        
        # Số lượng các từ tích cực
        st.write('### Số lượng các từ tích cực đã nhận xét')
        # Tạo đồ thị
        fig, ax = plt.subplots(figsize=(10, 6))  # Khởi tạo figure và axis
        sns.countplot(data=data, x='positive_words_count', hue='positive_words_count', legend=False, palette='tab10', ax=ax)  # type: ignore # Tạo biểu đồ
        # Thêm nhãn trên các cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Tùy chỉnh đồ thị
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Xoay nhãn trục X
        ax.set_title('Số lượng các từ tích cực đã nhận xét')  # Thêm tiêu đề
        plt.tight_layout()  # Đảm bảo bố cục gọn gàng
        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)

        # Số lượng các từ tiêu cực
        st.write('### Số lượng các từ tiêu cực đã nhận xét')
        # Tạo đồ thị
        fig, ax = plt.subplots(figsize=(10, 6))  # Khởi tạo figure và axis
        sns.countplot(data=data, x='negative_words_count', hue='negative_words_count', legend=False, palette='tab10', ax=ax)  # type: ignore # Tạo biểu đồ
        # Thêm nhãn trên các cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Tùy chỉnh đồ thị
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Xoay nhãn trục X
        ax.set_title('Số lượng các từ tiêu cực đã nhận xét')  # Thêm tiêu đề
        plt.tight_layout()  # Đảm bảo bố cục gọn gàng
        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)

        # Thống kê số lượng bình luận theo quý
        comment_count_quarter = data.groupby('quarter').size().reset_index()
        comment_count_quarter.rename(columns={0: 'so_luong_binh_luan_quy'}, inplace=True)
        st.write('### Thống kê số lượng bình luận theo quý')
        # Tạo figure và axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Tạo biểu đồ lineplot
        sns.lineplot(
            data=comment_count_quarter, 
            x='quarter', 
            y='so_luong_binh_luan_quy', 
            marker='o', 
            label='Số lượng bình luận theo quý', 
            ax=ax
        )
        # Thêm giá trị trực tiếp lên biểu đồ
        for x, y in zip(comment_count_quarter['quarter'], comment_count_quarter['so_luong_binh_luan_quy']):
            ax.text(x, y, str(y), color='black', ha='center', va='bottom', fontsize=10)
        # Thiết lập tiêu đề và nhãn
        ax.set_title('Thống kê số lượng bình luận theo quý', fontsize=16)
        ax.set_xlabel('Quý', fontsize=12)
        ax.set_ylabel('Số lượng bình luận theo quý', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.grid(True)
        ax.legend()
        # Hiển thị biểu đồ
        st.pyplot(fig)

        # Tần suất Positive/Negative
        st.write('### Tần suất Positive/Negative trên tập dữ liệu')
        # Tạo figure và axis
        fig, ax = plt.subplots(figsize=(8, 5))
        # Tạo biểu đồ countplot
        sns.countplot(
            data=data, 
            x='sentiment', 
            palette='tab10', 
            ax=ax
        )
        # Thêm nhãn giá trị lên các cột
        for container in ax.containers: # type: ignore
            ax.bar_label(container)
        # Thiết lập tiêu đề và xoay nhãn trục X
        ax.set_title('Tần suất Positive/Negative trên tập dữ liệu', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # Tự động căn chỉnh layout
        plt.tight_layout()
        # Hiển thị biểu đồ
        st.pyplot(fig)


# Giao diện phần 'Thông tin về sản phẩm'
if info_options == 'Thông tin về sản phẩm':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset chưa được tải lên')
    else:
        st.write('## Truy suất thông tin về một sản phẩm bất kỳ')
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        
        # Lấy sản phẩm
        data_info = data[['ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'gia_ban', 'processed_noi_dung_binh_luan', 'ngay_binh_luan']]
        random_products = data_info.drop_duplicates(subset='ma_san_pham')
        st.session_state.random_products = random_products
        
        # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
        if 'selected_ma_san_pham' not in st.session_state:
            st.session_state.selected_ma_san_pham = None # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
            
        # Theo cách cho người dùng chọn sản phẩm từ dropdown
        # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_products.iterrows()]

        # Tạo một dropdown với options là các tuple này
        selected_product = st.selectbox(
            'Chọn sản phẩm',
            options=product_options,
            format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
        )
        # # Display the selected product
        # st.write('Bạn đã chọn:', selected_product)
        
        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_ma_san_pham = selected_product[1] # type: ignore

        if st.session_state.selected_ma_san_pham:
            st.write(f'ma_san_pham: {st.session_state.selected_ma_san_pham}')
            # Hiển thị thông tin sản phẩm được chọn
            selected_product = data[data['ma_san_pham'] == st.session_state.selected_ma_san_pham].sort_values(by='ngay_binh_luan', ascending=False)

            if not selected_product.empty:
                st.write('-'*3)
                st.write(f'#### {selected_product["ten_san_pham"].values[0]}')
                col1, col2 = st.columns([2,4.5])
                with col1:
                    st.write(f'##### {selected_product["diem_trung_binh"].values[0]} :star:', '{:,.0f}'.format(selected_product["gia_ban"].values[0]),'VNĐ')
                    product_description = selected_product['mo_ta'].values[0]
                with col2:
                    # Tần suất số sao đánh giá trên sản phẩm
                    # Tạo figure và axis
                    fig, ax = plt.subplots(figsize=(6, 2))
                    # Tạo biểu đồ countplot
                    sns.countplot(
                        data=selected_product[['so_sao']].sort_values(by='so_sao', ascending=False), 
                        y='so_sao', 
                        # palette='tab10', 
                        color='palegoldenrod',
                        ax=ax,
                        width=0.5
                    )
                    # Thêm nhãn giá trị lên các cột
                    for container in ax.containers: # type: ignore
                        ax.bar_label(container)
                    # Thiết lập tiêu đề và xoay nhãn trục X
                    ax.set_title('Số sao đánh giá trên sản phẩm', fontsize=15)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                    plt.xlabel('')
                    plt.ylabel('')
                    # Tự động căn chỉnh layout
                    plt.tight_layout()
                    # Hiển thị biểu đồ
                    st.pyplot(fig)    
                # Tabs chính
                info_tabs = st.tabs(['Thông tin sản phẩm', 'Đánh giá từ khách hàng', 'Wordcloud'])
                with info_tabs[0]:
                    st.write(product_description.replace('THÔNG TIN SẢN PHẨM','').replace('Làm sao để phân biệt hàng có trộn hay không ?\nHàng trộn sẽ không thể xuất hoá đơn đỏ (VAT) 100% được do có hàng không nguồn gốc trong đó.\nTại Hasaki, 100% hàng bán ra sẽ được xuất hoá đơn đỏ cho dù khách hàng có lấy hay không. Nếu có nhu cầu lấy hoá đơn đỏ, quý khách vui lòng lấy trước 22h cùng ngày. Vì sau 22h, hệ thống Hasaki sẽ tự động xuất hết hoá đơn cho những hàng hoá mà khách hàng không đăng kí lấy hoá đơn.\nDo xuất được hoá đơn đỏ 100% nên đảm bảo 100% hàng tại Hasaki là hàng chính hãng có nguồn gốc rõ ràng.',''))
                with info_tabs[1]:
                    # for i in range(len(selected_product['noi_dung_binh_luan'])):
                    #     st.write(f'{selected_product["ngay_binh_luan"].dt.strftime("%d-%m-%Y").values[i]}, {selected_product["ho_ten"].values[i]}, {selected_product["so_sao"].values[i]*":star:"}')
                    #     st.write(f'{selected_product["noi_dung_binh_luan"].values[i]}')
                    #     st.write('-'*3)
                    # Lọc số sao đánh giá
                    # Tạo danh sách unique số sao từ dữ liệu
                    star_ratings = sorted(selected_product["so_sao"].unique())
                    # Tạo selectbox để chọn số sao
                    selected_star = st.selectbox("Chọn số sao để lọc bình luận:", options=["Tất cả"] + star_ratings)

                    # Lọc dữ liệu dựa trên số sao đã chọn
                    if selected_star != "Tất cả":
                        filtered_reviews = selected_product[selected_product["so_sao"] == selected_star]
                    else:
                        filtered_reviews = selected_product

                    # Hiển thị các bình luận đã được lọc
                    for i in range(len(filtered_reviews)):
                        st.write(f'{filtered_reviews["ngay_binh_luan"].dt.strftime("%d-%m-%Y").values[i]}, {filtered_reviews["ho_ten"].values[i]}, {filtered_reviews["so_sao"].values[i] * ":star:"}')
                        st.write(f'{filtered_reviews["noi_dung_binh_luan"].values[i]}')
                        st.write('-' * 3)
                with info_tabs[2]:
                    filtered_product = selected_product.groupby('ma_san_pham')['processed_noi_dung_binh_luan'].apply(' '.join).reset_index()
                    filtered_product.rename(columns={'processed_noi_dung_binh_luan': 'merged_comments'}, inplace=True)
                    filtered_product['positive_words'] = filtered_product['merged_comments'].apply(lambda txt: ' '.join(tpr.find_words(txt, list_of_words=positive_words_lst)[1]))
                    filtered_product['negative_words'] = filtered_product['merged_comments'].apply(lambda txt: ' '.join(tpr.find_words(txt, list_of_words=negative_words_lst)[1]))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write('##### Wordcloud tích cực')

                        # Lấy giá trị đầu tiên từ mảng và xử lý các từ đặc biệt
                        positive_bowl = filtered_product['positive_words'].to_numpy()[0]
                        positive_bowl = tpr.process_special_word(positive_bowl)

                        # Đảm bảo positive_bowl là một chuỗi hợp lệ
                        if isinstance(positive_bowl, list):  # Nếu đầu ra là danh sách, nối các từ lại thành chuỗi
                            positive_bowl = ' '.join(positive_bowl)
                        elif not isinstance(positive_bowl, str):  # Nếu không phải chuỗi, chuyển đổi về chuỗi
                            positive_bowl = str(positive_bowl)

                        # Kiểm tra nếu không có chữ nào để tạo Wordcloud
                        if not positive_bowl.strip():  # .strip() để loại bỏ khoảng trắng
                            st.warning('Hiện tại chưa có chữ để trích xuất Wordcloud')
                        else:
                            # Tạo positive WordCloud
                            positive_wordcloud = wc(
                                width=800,
                                height=400,
                                max_words=25,
                                background_color='white',
                                colormap='viridis',
                                collocations=False
                            ).generate(positive_bowl)

                            # Hiển thị positive WordCloud trong Streamlit
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(positive_wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                    with col2:
                        st.write('##### Wordcloud tiêu cực')

                        # Lấy giá trị đầu tiên từ mảng và xử lý các từ đặc biệt
                        negative_bowl = filtered_product['negative_words'].to_numpy()[0]
                        negative_bowl = tpr.process_special_word(negative_bowl)

                        # Đảm bảo negative_bowl là một chuỗi hợp lệ
                        if isinstance(negative_bowl, list):  # Nếu đầu ra là danh sách, nối các từ lại thành chuỗi
                            negative_bowl = ' '.join(negative_bowl)
                        elif not isinstance(negative_bowl, str):  # Nếu không phải chuỗi, chuyển đổi về chuỗi
                            negative_bowl = str(negative_bowl)

                        # Kiểm tra nếu không có chữ nào để tạo Wordcloud
                        if not negative_bowl.strip():  # .strip() để loại bỏ khoảng trắng
                            st.warning('Hiện tại chưa có chữ để trích xuất Wordcloud')
                        else:
                            # Tạo negative WordCloud
                            negative_wordcloud = wc(
                                width=800,
                                height=400,
                                max_words=25,
                                background_color='white',
                                colormap='Oranges',
                                collocations=False
                            ).generate(negative_bowl)

                            # Hiển thị negative WordCloud trong Streamlit
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(negative_wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
        else:
            st.write(f'Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}')
#----------------------------------------------------------------------------------------------------#       
# Giao diện phần 'Tổng quan về dataset'
if info_options == 'Hệ thống gợi ý sản phẩm':
    st.image('img/hasaki_logo.png', use_column_width=True)
    if st.session_state['uploaded_data'] is None:
        st.warning('Dataset phục vụ hệ thống gợi ý sản phẩm chưa được tải lên')
    else:
        data = st.session_state['uploaded_data']  # Lấy dữ liệu từ session_state
        rcm_tabs = st.tabs(['Sản phẩm khuyến nghị cho khách hàng', 'Đánh giá từ khách hàng khác'])
        with rcm_tabs[0]:
            # Lấy user_Id
            data_info = data[['ma_khach_hang', 'ho_ten']]
            random_products = data_info.drop_duplicates()
            st.session_state.random_products = random_products
            
            # Kiểm tra xem 'selected_ma_khach_hang' đã có trong session_state hay chưa
            if 'selected_ma_khach_hang' not in st.session_state:
                st.session_state.selected_ma_khach_hang = None # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
                
            # Theo cách cho người dùng chọn sản phẩm từ dropdown
            # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
            product_options = [(row['ho_ten'], row['ma_khach_hang']) for index, row in st.session_state.random_products.iterrows()]

            # Tạo một dropdown với options là các tuple này
            selected_product = st.selectbox(
                'Chọn mã khách hàng',
                options=product_options,
                format_func=lambda x: x[1]  # Hiển thị userId
            )
            col1, col2 = st.columns(2)
            with col1:
                stars = st.number_input("Số sao sản phẩm không bé hơn:", value=5, min_value=1, max_value=5)
            with col2:
                top_filter = st.number_input("Số lượng sản phẩm gợi ý", min_value=1, step=1, value=5)
            # Display the selected product
            user_id = selected_product[1] # type: ignore
            name = selected_product[0] # type: ignore
            st.write(f'Tên khách hàng: {name}')
            
            # Cập nhật session_state dựa trên lựa chọn hiện tại
            st.session_state.selected_ma_khach_hang = selected_product[1] # type: ignore
            data_recommendation = data[['ma_khach_hang', 'ho_ten', 'ma_san_pham', 'ten_san_pham', 'mo_ta', 'mo_ta_special_words_remove_stopword', 'so_sao', 'diem_trung_binh', 'gia_ban', 'ngay_binh_luan']]
            # Gợi ý sản phẩm
            if 'recommendation' not in st.session_state:
                st.session_state['recommendation'] = None
            if 'history' not in st.session_state:
                st.session_state['history'] = None
            if 'user_name' not in st.session_state:
                st.session_state['user_name'] = None
            if 'user_id' not in st.session_state:
                st.session_state['user_id'] = None
            if st.button("Gợi Ý Sản Phẩm"):
                with st.spinner("Đang xử lý..."):
                    start_time = time.time()
                    recommendations = rs.surprise_recommendation(
                        input_df=data_recommendation, 
                        userId_col_name = 'ma_khach_hang',
                        productId_col_name = 'ma_san_pham',
                        product_name_col_name= 'ten_san_pham',
                        rating_col_name = 'so_sao',
                        model_algorithm = surprise_model,
                        userId = user_id,
                        rate_threshold = stars,
                        top_recommend = int(top_filter),
                        user_history=True)
                    process_time = time.time() - start_time
                st.success(f"Gợi ý hoàn tất trong {process_time:.2f}s!")
                st.write('-'*3)
                st.write(f'\nLịch sử mua hàng của {name}({user_id}).')
                st.dataframe(recommendations[1], use_container_width=True)
                st.write(f'\nSản phẩm khuyến cáo cho khách là:')
                st.dataframe(recommendations[0], use_container_width=True)
                st.session_state['recommendation'] = recommendations[0] # Lưu sản phẩm khuyến cáo
                st.session_state['hisory'] = recommendations[1] # Lưu lịch sử mua hàng
                st.session_state['user_name'] = name # Lưu sản phẩm khuyến cáo
                st.session_state['user_id'] = user_id # Lưu lịch sử mua hàng
            elif st.session_state['recommendation'] is not None:
                history = st.session_state['hisory']
                user_recommend = st.session_state['recommendation']
                last_userId = st.session_state['user_id']
                last_username = st.session_state['user_name']
                st.write('-'*3)
                st.write(f'\nLịch sử mua hàng của {last_username}({last_userId}).')
                st.dataframe(history, use_container_width=True)
                st.write(f'\nSản phẩm khuyến cáo cho khách là:')
                st.dataframe(user_recommend, use_container_width=True)
            with rcm_tabs[1]:
                last_userId = st.session_state['user_id']
                last_username = st.session_state['user_name']
                user_recommend = st.session_state['recommendation']
                st.write(f'Hệ thống gợi ý các sản phẩm này cho {last_username}({last_userId}).')
                # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
                if user_recommend is not None:
                    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in user_recommend.iterrows()]
                    # Tạo một dropdown với options là các tuple này
                    selected_product = st.selectbox(
                        'Chọn xem thông tin sản phẩm',
                        options=product_options,
                        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
                    )

                # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
                if 'selected_ma_san_pham' not in st.session_state:
                    st.session_state.selected_ma_san_pham = None # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
                
                # Cập nhật session_state dựa trên lựa chọn hiện tại
                st.session_state.selected_ma_san_pham = selected_product[1] # type: ignore

                if st.session_state.selected_ma_san_pham and user_recommend is not None:
                    st.write(f'ma_san_pham: {st.session_state.selected_ma_san_pham}')
                    # Hiển thị thông tin sản phẩm được chọn
                    selected_product = data[data['ma_san_pham'] == st.session_state.selected_ma_san_pham].sort_values(by='ngay_binh_luan', ascending=False)

                    if not selected_product.empty:
                        st.write('-'*3)
                        st.write(f'#### {selected_product["ten_san_pham"].values[0]}')
                        col1, col2 = st.columns([2,4.5])
                        with col1:
                            st.write(f'##### {selected_product["diem_trung_binh"].values[0]} :star:', '{:,.0f}'.format(selected_product["gia_ban"].values[0]),'VNĐ')
                            product_description = selected_product['mo_ta'].values[0]
                        with col2:
                            # Tần suất số sao đánh giá trên sản phẩm
                            # Tạo figure và axis
                            fig, ax = plt.subplots(figsize=(6, 2))
                            # Tạo biểu đồ countplot
                            sns.countplot(
                                data=selected_product[['so_sao']].sort_values(by='so_sao', ascending=False), 
                                y='so_sao', 
                                # palette='tab10', 
                                color='palegoldenrod',
                                ax=ax,
                                width=0.5
                            )
                            # Thêm nhãn giá trị lên các cột
                            for container in ax.containers: # type: ignore
                                ax.bar_label(container)
                            # Thiết lập tiêu đề và xoay nhãn trục X
                            ax.set_title('Số sao đánh giá trên sản phẩm', fontsize=15)
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                            plt.xlabel('')
                            plt.ylabel('')
                            # Tự động căn chỉnh layout
                            plt.tight_layout()
                            # Hiển thị biểu đồ
                            st.pyplot(fig)
                        product_description = selected_product['mo_ta'].values[0]
                        # truncated_description = ' '.join(product_description.split()[:100])
                        # Tabs chính
                        info_tabs = st.tabs(['Thông tin sản phẩm', 'Đánh giá từ khách hàng', 'Wordcloud'])
                        with info_tabs[0]:
                            st.write(product_description.replace('THÔNG TIN SẢN PHẨM','').replace('Làm sao để phân biệt hàng có trộn hay không ?\nHàng trộn sẽ không thể xuất hoá đơn đỏ (VAT) 100% được do có hàng không nguồn gốc trong đó.\nTại Hasaki, 100% hàng bán ra sẽ được xuất hoá đơn đỏ cho dù khách hàng có lấy hay không. Nếu có nhu cầu lấy hoá đơn đỏ, quý khách vui lòng lấy trước 22h cùng ngày. Vì sau 22h, hệ thống Hasaki sẽ tự động xuất hết hoá đơn cho những hàng hoá mà khách hàng không đăng kí lấy hoá đơn.\nDo xuất được hoá đơn đỏ 100% nên đảm bảo 100% hàng tại Hasaki là hàng chính hãng có nguồn gốc rõ ràng.',''))
                        with info_tabs[1]:
                            # for i in range(len(selected_product['noi_dung_binh_luan'])):
                            #     st.write(f'{selected_product["ngay_binh_luan"].dt.strftime("%d-%m-%Y").values[i]}, {selected_product["ho_ten"].values[i]}, {selected_product["so_sao"].values[i]*":star:"}')
                            #     st.write(f'{selected_product["noi_dung_binh_luan"].values[i]}')
                            #     st.write('-'*3)
                            # Lọc số sao đánh giá
                            # Tạo danh sách unique số sao từ dữ liệu
                            star_ratings = sorted(selected_product["so_sao"].unique())
                            # Tạo selectbox để chọn số sao
                            selected_star = st.selectbox("Chọn số sao để lọc bình luận:", options=["Tất cả"] + star_ratings)

                            # Lọc dữ liệu dựa trên số sao đã chọn
                            if selected_star != "Tất cả":
                                filtered_reviews = selected_product[selected_product["so_sao"] == selected_star]
                            else:
                                filtered_reviews = selected_product

                            # Hiển thị các bình luận đã được lọc
                            for i in range(len(filtered_reviews)):
                                st.write(f'{filtered_reviews["ngay_binh_luan"].dt.strftime("%d-%m-%Y").values[i]}, {filtered_reviews["ho_ten"].values[i]}, {filtered_reviews["so_sao"].values[i] * ":star:"}')
                                st.write(f'{filtered_reviews["noi_dung_binh_luan"].values[i]}')
                                st.write('-' * 3)
                        with info_tabs[2]:
                            filtered_product = selected_product.groupby('ma_san_pham')['processed_noi_dung_binh_luan'].apply(' '.join).reset_index()
                            filtered_product.rename(columns={'processed_noi_dung_binh_luan': 'merged_comments'}, inplace=True)
                            filtered_product['positive_words'] = filtered_product['merged_comments'].apply(lambda txt: ' '.join(tpr.find_words(txt, list_of_words=positive_words_lst)[1]))
                            filtered_product['negative_words'] = filtered_product['merged_comments'].apply(lambda txt: ' '.join(tpr.find_words(txt, list_of_words=negative_words_lst)[1]))
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write('##### Wordcloud tích cực')

                                # Lấy giá trị đầu tiên từ mảng và xử lý các từ đặc biệt
                                positive_bowl = filtered_product['positive_words'].to_numpy()[0]
                                positive_bowl = tpr.process_special_word(positive_bowl)

                                # Đảm bảo positive_bowl là một chuỗi hợp lệ
                                if isinstance(positive_bowl, list):  # Nếu đầu ra là danh sách, nối các từ lại thành chuỗi
                                    positive_bowl = ' '.join(positive_bowl)
                                elif not isinstance(positive_bowl, str):  # Nếu không phải chuỗi, chuyển đổi về chuỗi
                                    positive_bowl = str(positive_bowl)

                                # Kiểm tra nếu không có chữ nào để tạo Wordcloud
                                if not positive_bowl.strip():  # .strip() để loại bỏ khoảng trắng
                                    st.warning('Hiện tại chưa có chữ để trích xuất Wordcloud')
                                else:
                                    # Tạo positive WordCloud
                                    positive_wordcloud = wc(
                                        width=800,
                                        height=400,
                                        max_words=25,
                                        background_color='white',
                                        colormap='viridis',
                                        collocations=False
                                    ).generate(positive_bowl)

                                    # Hiển thị positive WordCloud trong Streamlit
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.imshow(positive_wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                            with col2:
                                st.write('##### Wordcloud tiêu cực')

                                # Lấy giá trị đầu tiên từ mảng và xử lý các từ đặc biệt
                                negative_bowl = filtered_product['negative_words'].to_numpy()[0]
                                negative_bowl = tpr.process_special_word(negative_bowl)

                                # Đảm bảo negative_bowl là một chuỗi hợp lệ
                                if isinstance(negative_bowl, list):  # Nếu đầu ra là danh sách, nối các từ lại thành chuỗi
                                    negative_bowl = ' '.join(negative_bowl)
                                elif not isinstance(negative_bowl, str):  # Nếu không phải chuỗi, chuyển đổi về chuỗi
                                    negative_bowl = str(negative_bowl)

                                # Kiểm tra nếu không có chữ nào để tạo Wordcloud
                                if not negative_bowl.strip():  # .strip() để loại bỏ khoảng trắng
                                    st.warning('Hiện tại chưa có chữ để trích xuất Wordcloud')
                                else:
                                    # Tạo negative WordCloud
                                    negative_wordcloud = wc(
                                        width=800,
                                        height=400,
                                        max_words=25,
                                        background_color='white',
                                        colormap='Oranges',
                                        collocations=False
                                    ).generate(negative_bowl)

                                    # Hiển thị negative WordCloud trong Streamlit
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.imshow(negative_wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                else:
                    st.write(f'Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}')
        
