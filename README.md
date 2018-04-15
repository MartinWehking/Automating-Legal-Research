# [Automating-Legal-Research](https://github.com/MartinWehking/Automating-Legal-Research)

With the "Automating legal research" project you can download and build datasets from the translations of German legal texts found at [gesetze-im-internet.de](gesetze-im-internet.de). The datasets can be used to create TF-IDF and Doc2Vec vector spaces and perform clustering and classification on it. Additionally you can use the vector spaces to search for documents similar to a given search term.

This project is the result of our work for the Text Mining Project at the University of Passau in the Wintersemester 17/18.

## Usage

The tool requires you to have at least Python 3.6 installed.
For information on the libraries we used in this project see [libraries.txt](libraries.txt).

The tool can be started with ```python tool.py <options>```.

To find out about the available options simply start it with ```python tool.py -h```

## Examples

Creating a dataset using the default config with all legal texts from [gesetze-im-internet.de](gesetze-im-internet.de). Hint: you can remove unneeded URLs/legal texts from the config in order to reduce the time needed to fetch the data and reduce the computation time of later steps.
```
python tool.py --config=config.json --out=dataset.csv
```
Dataset is now saved as dataset.csv.

Search dataset for "travel expenses" using TF-IDF for the vector space:
```
python tool.py --dataset=dataset.csv --tfidf --search="travel expenses"
```

Generate vector space using TF-IDF, cluster using kmeans and visualize results:
```
python tool.py --dataset=dataset.csv --tfidf --cluster --algorithm=kmeans --visualize
```

Use Doc2Vec for vector space instead (worse results):
```
python tool.py --dataset=dataset.csv --doc2vec --cluster --algorithm=kmeans --visualize
```

Use HAC for clustering instead (worse results):
```
python tool.py --dataset=dataset.csv --tfidf --cluster --algorithm=HAC --visualize
```

Use TF-IDF for the vector space and classify results:
```
python tool.py --dataset=dataset.csv --tfidf --classify
```

Use Doc2Vec for vector space instead (worse results):
```
python tool.py --dataset=dataset.csv --doc2vec --classify
```
