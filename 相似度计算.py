from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_text_similarity(text1, text2, model_name='shibing624/text2vec-base-chinese'):
    """
    计算两段文本的相似度
    
    参数:
        text1 (str): 第一段文本
        text2 (str): 第二段文本
        model_name (str): 使用的模型名称，默认使用中文模型
    
    返回:
        float: 相似度分数 (0-1之间，越接近1表示越相似)
    """
    # 加载模型
    model = SentenceTransformer(model_name)
    
    # 生成文本嵌入向量
    embeddings = model.encode([text1, text2])
    
    # 计算余弦相似度
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return similarity