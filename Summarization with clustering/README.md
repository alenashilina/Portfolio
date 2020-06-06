## Main idea
The main idea of this little project is performing Korean text summarization using K-means clustering and it was inspired by the paper *[“Leveraging BERT for Extractive Text Summarization on Lectures”]( https://arxiv.org/abs/1906.04165)*. After getting embeddings of sentences (using BERT) I cluster them (assuming that sentences within one cluster are similar ones) and those sentences which are closest to clusters’ centers must be the ones that will make the summary. 

## Data
The dataset used for summarization: *[Naver 50 Scientific news articles]( https://github.com/theeluwin/sci-news-sum-kr-50)*. It contains 50 news articles in Korean with corresponding summaries.

## The basic process
After importing data, I get BERT sentence embeddings using files *getting_bert_embeddings.py*, *modeling.py* and *tokenization.py* which were taken from *[Google’s BERT github repository]( https://github.com/google-research/bert)*. The model used for embeddings was Google’s multilingual BERT model fine-tuned by me with 173k news articles for 463k steps (not included into the repository). Then obtained embeddings are sent to get the summary using K-means clustering algorithm. The optimal number of clusters is found using *[Calinski and Harabasz score]( https://www.tandfonline.com/doi/abs/10.1080/03610927408827101)* (also known as variance ratio criterion). It is based on the idea that between clusters sum of squares must be maximized, while within cluster sum of squares must be minimized. The higher the score – the more optimal the number of clusters is.
