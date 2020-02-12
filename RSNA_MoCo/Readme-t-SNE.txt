Steps to plot t-SNE of MoCo_super and MoCo
1、Run generate_embeddings.py. You need to change the "model_path" with your MoCo model path or MoCo_super path. And change the "features_path" and "labels_path" to store the output features and labels. 
2、Run rewrite_labels.py. Set the "features_path" and "labels_path" as same as in step1. And set the "new_labels_path" to store the generated new labels.
3、Run t-SNE_Plotting.py. Set the features_path" as same as in step1 and set the "new_labels_path"  as same as step2. The "low" and "high" are to reduce the number of samples to show, you can also change them with numbers <300.
Done.
