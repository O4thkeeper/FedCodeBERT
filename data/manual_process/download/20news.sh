cd data/store || exit
mkdir 20news
cd 20news || exit
wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -zxvf 20news-bydate.tar.gz
rm 20news-bydate.tar.gz