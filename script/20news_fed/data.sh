#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

sh data/manual_process/download/20news.sh

${py} -m data.manual_process.load.20news

${py} -m data.manual_process.partition.niid_quantity \
--client_number 5  \
--data_file data/store/20news/20news_data.h5  \
--partition_file data/store/20news/20news_partition.h5 \
--task_type text_classification \
--kmeans_num 0 \
--beta 5