
# Preprocess
1. Image features are extracted by 'Bottom-up attention', please refer the link-'https://github.com/airsplay/py-bottom-up-attention' to install the environment.
2. After install the environment, you can extract the image feature by the script in dir 'tools'.
# Training
Follow the training script in 'example' to train the model. 


# Evalution
* The retrieval results for each query should be written into a ".txt" file named "retrieval_id_list.txt"，and then can be evaluated by the script "evaluate_suit.py".
* Each line in the text file contains the query item id followed by a ranked retrieval item id list (up to 100 item ids) separated by ",".
* Examples in the text file: "{query_id}, {retrieval id_0}, {retrieval id_1}, {retrieval id_2},...,{retrieval id_99}"





