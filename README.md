# exBERT
The details of the model is in [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.129/).


## Pre-train an exBERT model (only the extension part)

In command line:

    python Pretraining.py -e 1 
      -b 256 
      -sp path_to_storage
      -dv 0 1 2 3 -lr 1e-04 
      -str exBERT    
      -config path_to_config_file_of_the_OFF_THE_SHELF_MODEL ./config_and_vocab/exBERT/bert_config_ex_s3.json  
      -vocab ./config_and_vocab/exBERT/exBERT_vocab.txt 
      -pm_p path_to_state_dict_of_the_OFF_THE_SHELF_MODEL
      -dp path_to_your_training_data
      -ls 128 
      -p 1
You can replace the `path_to_config_file_of_the_OFF_THE_SHELF_MODEL` and `path_to_state_dict_of_the_OFF_THE_SHELF_MODEL` to any weel pre-trained model in BERT archietecture.
`./config_and_vocab/exBERT/bert_config_ex_s3.json` defines the size of extension module.

## Pre-train an exBERT model (whole model)

    python Pretraining.py -e 1 
      -b 256 
      -sp path_to_storage
      -dv 0 1 2 3 -lr 1e-04 
      -str exBERT    
      -config path_to_config_file_of_the_OFF_THE_SHELF_MODEL ./config_and_vocab/exBERT/bert_config_ex_s3.json  
      -vocab ./config_and_vocab/exBERT/exBERT_vocab.txt 
      -pm_p path_to_state_dict_of_the_OFF_THE_SHELF_MODEL
      -dp path_to_your_training_data
      -ls 128 
      -p 1
      -t_ex_only ""

`-t_ex_only ""` enable training the whole model

## Pre-train an exBERT with no extension of vocab

    python Pretraining.py -e 1 
      -b 256 
      -sp path_to_storage
      -dv 0 1 2 3 -lr 1e-04 
      -str exBERT    
      -config path_to_config_file_of_the_OFF_THE_SHELF_MODEL config_and_vocab/exBERT_no_ex_vocab/bert_config_ex_s3.json
      -vocab path_to_vocab_file_of_the_OFF_THE_SHELF_MODEL
      -pm_p path_to_state_dict_of_the_OFF_THE_SHELF_MODEL
      -dp path_to_your_training_data
      -ls 128 
      -p 1
      -t_ex_only ""


## Data preparation
Input data for pre-training script should be a .pkl file which contains two a list with two elements, e.g. \[list1, list2\].
list1 and list2 should contains the sentences like `[CLS] sentence A [SEP] sentence B [SEP]`. The only differnece between list1 and list2 is the relationship between `sentence A` and `sentence B` is IsNext or NotNext.  Please check `example_data.pkl`

We also provide a simple script to generate the data from raw text file. 
`python data_preprocess.py -voc path_to_vocab_file -ls 128 -dp path_to_txt_file -n_c 5 -rd 1 -sp ./your_data.pkl`
replace `128` to the max length limit you want
try `python data_preprocess.py -voc ./exBERT_vocab.txt -ls 128 -dp ./example_raw_text.txt -n_c 5 -rd 1 -sp ./example_data.pkl`

Or you can do your own data preparation and organize the data with the format metioned above.
