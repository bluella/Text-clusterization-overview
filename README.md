# Text clusterization overview

This is more of a cheat sheet, than a serious project with high goals.
Data is Reddit posts for one year with #wot hashtag [posts_Reddit_wot_en.csv](https://github.com/bluella/Text-clusterization-overview/blob/master/datasets/posts_Reddit_wot_en.csv).
Text vectorization was done with four main methods:
[BoW](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html),
[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
[PV-DM](https://cs.stanford.edu/~quocle/paragraph_vector.pdf),
[PV-DBOW](https://cs.stanford.edu/~quocle/paragraph_vector.pdf),

Clusterization method is always [K-means++](https://en.wikipedia.org/wiki/K-means%2B%2B), just because i believe
modification of it makes little impact compared to change of vectorization technique.
Visualization is performed via:
[MDS](https://en.wikipedia.org/wiki/Multidimensional_scaling),
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## Install

```bash
git clone git@github.com:bluella/Text-clusterization-overview.git
cd Text-clusterization-overview
virtualenv -p /usr/bin/python3.7 tco_env
source ./tco_env/bin/activate
pip install -r requirements.txt
```

You are good to go!

## Results

TF-IDF has shown best results among other vectorization methods.
BoW is a bit less accurate.
PV-DM and PV-DBOW deliveres really weird results. Pephaps because of small dataset
size, which is not appropriate to proper model learning.
PCA visualization seems to comply more with real outcome than MDS.

## Futher development

- [Proper clusterization evaluation](https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment)

- Use pretrained model for PV-DM with help of [fasttext](https://github.com/facebookresearch/fastText) or else

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/bluella/Text-clusterization-overview/blob/master/LICENSE.md) file for details

## Acknowledgments

Heavy loads of code were taken from the following resources:

- [Great word2vec tutorial](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial)

- [BoW clusterization](http://brandonrose.org/clustering)

- [Doc2vec clusterization](https://medium.com/@ermolushka/text-clusterization-using-python-and-doc2vec-8c499668fa61)

- [Preprocessing](https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908)
