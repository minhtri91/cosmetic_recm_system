import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from underthesea import word_tokenize, pos_tag, sent_tokenize
import pyvi
from utils import TextProcessing as tpr
import regex
import string
import os
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

import surprise
from surprise.model_selection import cross_validate
from surprise.model_selection.validation import cross_validate
from surprise import Reader, Dataset, SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, CoClustering, BaselineOnly

from utils import TextProcessing as tpr
from utils import evaluation

def compare_surprise_models(input_df, userId_col_name, productId_col_name, product_name_col_name, rating_col_name):
    """
    So sánh các mô hình của Surprise dựa trên RMSE, MAE và thời gian cross-validate.

    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame chứa dữ liệu đầu vào.
    userId_col_name : str
        Tên cột chứa mã người dùng.
    productId_col_name : str
        Tên cột chứa mã sản phẩm.
    product_name_col_name : str
        Tên cột chứa tên sản phẩm.
    rating_col_name : str
        Tên cột chứa đánh giá của người dùng.

    Returns:
    --------
    pd.DataFrame
        DataFrame chứa kết quả so sánh các mô hình.

    Ví dụ:
    --------    
    # Giả sử bạn có dữ liệu đầu vào
    result_df = compare_surprise_models(
        input_df=df, 
        userId_col_name='ma_khach_hang', 
        productId_col_name='ma_san_pham', 
        product_name_col_name='ten_san_pham', 
        rating_col_name='so_sao'
)
    """
    # Chuẩn bị dữ liệu
    df = input_df.copy()
    df.rename(columns={userId_col_name: 'userId', 
                       productId_col_name: 'productId', 
                       product_name_col_name: 'product_name',
                       rating_col_name: 'rating'}, inplace=True)
    df = df[['userId', 'productId', 'product_name', 'rating']].reset_index(drop=True)
    df['userId'] = df.userId.astype('int')
    df['productId'] = df.productId.astype('int')
    df['rating'] = df.rating.astype('float')

    # Danh sách thuật toán
    algorithms = {
        "SVD": SVD(),
        "SVDpp": SVDpp(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "KNNBaseline": KNNBaseline(),
        "KNNWithMeans": KNNWithMeans(),
        "KNNWithZScore": KNNWithZScore(),
        "CoClustering": CoClustering(),
        "BaselineOnly": BaselineOnly(),
    }

    # Chuẩn bị dữ liệu cho Surprise
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)

    # So sánh các mô hình
    results = []
    for name, algorithm in algorithms.items():
        start_time = time.time()
        scores = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        elapsed_time = time.time() - start_time
        results.append({
            "Model": name,
            "RMSE": scores['test_rmse'].mean(),
            "MAE": scores['test_mae'].mean(),
            "Cross-Validation Time (s)": elapsed_time
        })

    # Trả về DataFrame kết quả
    return pd.DataFrame(results).sort_values(by=['RMSE', 'Cross-Validation Time (s)'], ascending=True).reset_index(drop=True)

def surprise_model_builder(input_df=pd.DataFrame(), 
                           userId_col_name: str = '', 
                           productId_col_name: str = '', 
                           product_name_col_name: str = '', 
                           rating_col_name: str = '', 
                           model_algorithm: str = ''):
    # Sao chép dữ liệu
    start_process = time.time()
    df = input_df.copy()
    
    # Đổi tên các cột
    df.rename(columns={userId_col_name: 'userId', 
                       productId_col_name: 'productId', 
                       product_name_col_name: 'product_name',
                       rating_col_name: 'rating', }, inplace=True)
    
    # Lựa chọn các cột để đưa vào surprise xử lý
    df = df[['userId', 'productId', 'product_name', 'rating']]

    # Chuyển đổi loại dữ liệu
    df['userId'] = df.userId.astype('int')
    df['productId'] = df.productId.astype('int')
    df['rating'] = df.rating.astype('float')

    # Từ điển ánh xạ thuật toán
    algorithms = {
        "SVD": SVD(),
        "SVDpp": SVDpp(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "KNNBaseline": KNNBaseline(),
        "KNNWithMeans": KNNWithMeans(),
        "KNNWithZScore": KNNWithZScore(),
        "CoClustering": CoClustering(),
        "BaselineOnly": BaselineOnly(),
    }

    # Kiểm tra model_algorithm
    if model_algorithm not in algorithms:
        raise ValueError(f"Thuật toán '{model_algorithm}' không được hỗ trợ. Hãy chọn một trong các thuật toán sau: {list(algorithms.keys())}")

    # Chọn thuật toán
    algorithm_ = algorithms[model_algorithm]

    # Kiểm tra và đánh giá thuật toán với cross-validation
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)
    results = cross_validate(algorithm_, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

    # Kiểm tra độ phù hợp của thuật toán
    rmse_mean = results['test_rmse'].mean()
    if rmse_mean <= 1.0:
        pass
    else:
        raise ValueError(f"\nThuật toán '{model_algorithm}' không phù hợp vì RMSE trung bình = {rmse_mean:.2f} > 1. Hãy chọn thuật toán khác.\n")

    # Huấn luyện mô hình trên toàn bộ dữ liệu
    trainset = data.build_full_trainset()
    algorithm_.fit(trainset)
    evaluation.Save_Object(algorithm_,f'{model_algorithm}_algo.pkl')
    print(f'\nToàn bộ funtion xử lý trong {(time.time() - start_process):.2f}s')
    print(f'\nTrung bình RMSE của thuật toán {model_algorithm} = {rmse_mean:.2f}\n')
    return algorithm_

def surprise_recommendation(input_df=pd.DataFrame(), userId_col_name: str = '', productId_col_name: str = '', product_name_col_name: str = '', 
                 rating_col_name: str = '', model_algorithm=None, userId: int = 0, 
                 rate_threshold: float = 4, top_recommend: int = 5, user_history:bool = False):
    """
    Mô tả:
    ------
    Hàm recommend sản phẩm cho người dùng dựa trên Surprise Library.

    Params:
        - input_df (DataFrame): Dữ liệu đầu vào chứa thông tin userId, productId và rating.
        - userId_col_name (str): Tên cột chứa mã người dùng ở dataframe đầu vào.
        - productId_col_name (str): Tên cột chứa mã sản phẩm ở dataframe đầu vào
        - rating_col_name (str): Tên cột chứa điểm đánh giá ở dataframe đầu vào
        - model_algorithm (str): Tên của biến model được train từ function 'surprise_model_builder'
        - userId (int): ID của người dùng muốn recommend.
        - rate_threshold (float): Ngưỡng điểm đánh giá để lọc kết quả (default = 4).
        - top_recommend (int): Số lượng sản phẩm recommend tối đa(default = 5).
        - user_history (bool): Lịch sử mua hàng của khách hàng userId này.

    Returns:
        - Bảng kết quả gợi ý sản phẩm theo userId.
    """
    # Sao chép dữ liệu
    start_process = time.time()
    df = input_df.copy()
    
    # Đổi tên các cột
    df.rename(columns={userId_col_name: 'userId', 
                       productId_col_name: 'productId', 
                       product_name_col_name: 'product_name',
                       rating_col_name: 'rating', }, inplace=True)
    
    # Lựa chọn các cột để đưa vào surprise xử lý
    df = df[['userId', 'productId', 'product_name', 'rating', 'ngay_binh_luan']]

    # Chuyển đổi loại dữ liệu
    df['userId'] = df.userId.astype('int')
    df['productId'] = df.productId.astype('int')
    df['rating'] = df.rating.astype('float')

    # Chọn thuật toán
    algorithm_ = model_algorithm

    # Dự đoán điểm đánh giá cho sản phẩm
    df_score = df.copy()
    df_score['EstimateScore'] = df_score['productId'].apply(lambda x: algorithm_.predict(userId, x).est) # type: ignore
    df_score.drop_duplicates(subset=['productId'], inplace=True)
    df_score.sort_values(by=['EstimateScore'], ascending=False, inplace=True)

    # Lịch sử mua hàng của userId
    if user_history == True:
        history = df[(df['userId'] == userId) & (df['rating'] >= rate_threshold)].sort_values(by='ngay_binh_luan', ascending=False).reset_index(drop=True)
        history = history.reset_index(drop=True).head(top_recommend)
        history.rename(columns={'userId':userId_col_name, 
                                'productId':productId_col_name, 
                                'product_name':product_name_col_name,
                                'rating': rating_col_name}, inplace=True)
        history = history[[productId_col_name, product_name_col_name, rating_col_name]]
    else:
        history=''
    
    # Lọc kết quả theo ngưỡng đánh giá
    recommend = df_score[df_score['EstimateScore'] >= rate_threshold].head(top_recommend).reset_index(drop=True)
    recommend.rename(columns={'userId':userId_col_name, 
                            'productId':productId_col_name, 
                            'product_name':product_name_col_name,
                            'rating': rating_col_name}, inplace=True)
    print(f'\nToàn bộ funtion xử lý trong {(time.time() - start_process):.2f}s')
    return recommend[[productId_col_name, product_name_col_name, rating_col_name]], history

def hybrid_recommendation(
    user_id,
    search_kw,
    input_df,
    gensim_tfidf,
    gensim_index,
    gensim_dictionary,
    surprise_algorithm,
    top_n=5,
    stars_threshold=4,
):
    """
    Hybrid Recommendation dựa trên Gensim và Surprise.

    Parameters:
    -----------
    user_id : int
        ID của người dùng cần gợi ý sản phẩm.
    search_kw : str or int
        Từ khóa tìm kiếm hoặc mã sản phẩm.
    input_df : pd.DataFrame
        DataFrame chứa thông tin sản phẩm.
    gensim_tfidf : gensim.models.TfidfModel
        Mô hình TF-IDF đã huấn luyện.
    gensim_index : gensim.similarities.SparseMatrixSimilarity
        Ma trận tương tự từ Gensim.
    gensim_dictionary : gensim.corpora.Dictionary
        Từ điển của Gensim.
    surprise_algorithm : object
        Mô hình Surprise đã huấn luyện.
    top_n : int, optional
        Số sản phẩm gợi ý (default=5).
    stars_threshold : float, optional
        Ngưỡng đánh giá sản phẩm (default=4).

    Returns:
    --------
    pd.DataFrame
        DataFrame chứa thông tin sản phẩm gợi ý.
    """
    start_process = time.time()
    # Xử lý đầu vào search_kw
    if isinstance(search_kw, str) and search_kw.isdigit():
        search_kw = int(search_kw)

    # **1. Gensim Recommendation**
    if isinstance(search_kw, int):  # Tìm theo mã sản phẩm
        product_description = input_df.loc[
            input_df["ma_san_pham"] == search_kw, "mo_ta_special_words_remove_stopword"
        ].values
        if len(product_description) == 0:
            raise ValueError("Mã sản phẩm không tồn tại trong dữ liệu.")
        query = product_description[0].split()
    else:  # Tìm theo từ khóa
        query = word_tokenize(search_kw)

    # Tính toán độ tương đồng sản phẩm từ Gensim
    query_bow = gensim_dictionary.doc2bow(query)
    query_tfidf = gensim_tfidf[query_bow]
    gensim_sims = gensim_index[query_tfidf]

    # Lấy top sản phẩm từ Gensim
    gensim_results = sorted(enumerate(gensim_sims), key=lambda x: x[1], reverse=True)
    gensim_recommendations = [
        {
            "ma_san_pham": input_df.iloc[sim[0]]["ma_san_pham"],
            "ten_san_pham": input_df.iloc[sim[0]]["ten_san_pham"],
            "mo_ta": input_df.iloc[sim[0]]["mo_ta"],
            "so_sao": input_df.iloc[sim[0]]["so_sao"],
            "gia_ban": input_df.iloc[sim[0]]["gia_ban"],
            "similarity_score": sim[1],
        }
        for sim in gensim_results
        if input_df.iloc[sim[0]]["so_sao"] >= stars_threshold
    ]

    # Chuyển thành DataFrame
    gensim_df = pd.DataFrame(gensim_recommendations)

    # **2. Surprise Recommendation**
    # Dự đoán điểm đánh giá từ Surprise cho các sản phẩm từ Gensim
    gensim_df["EstimateScore"] = gensim_df["ma_san_pham"].apply(
        lambda x: surprise_algorithm.predict(user_id, x).est
    )

    # **3. Kết hợp kết quả**
    # Kết hợp similarity_score từ Gensim và EstimateScore từ Surprise
    gensim_df["final_score"] = 1/(1/gensim_df["similarity_score"]+ 1/gensim_df["EstimateScore"])

    # Sắp xếp theo final_score và trả về top N sản phẩm ['so_sao', 'final_score']
    recommendations = gensim_df.sort_values(by=['similarity_score','final_score'], ascending=False).drop_duplicates(subset='ma_san_pham').reset_index(drop=True).head(top_n)
    print(f'\nToàn bộ funtion `hybrid_recommendation` xử lý trong {(time.time() - start_process):.2f}s')
    print(f'\nCác sản phẩm recommend cho userId = {user_id}.')
    print(f'Với keyword tìm kiếm là "{search_kw}".')
    return recommendations
