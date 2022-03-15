NLP project for cse517

Example script for training:
python train.py --data_path ./data --data_name f30k_precomp --vocab_path ./vocab --logger_name ./fair_sample_flickr/ --model_name ./fair_sample_flickr/ --bi_gru --cross_attn=t2i --learning_rate=0.0002 --num_epochs=15

Example script for testing:
python train.py --data_path ./data --data_name f30k_precomp --vocab_path ./vocab --logger_name ./fair_sample_log2/ --model_name ./fair_sample_log2/ --bi_gru --cross_attn=t2i --resume flickr_model/model_best_nofair.pth.tar --test_data --neutralize
