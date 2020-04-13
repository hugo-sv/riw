# RI-W

This project implements a basic Information Retrieval (IR) system using the
boolean model.

## Requirements

1. Make sure that you are running `Python 3`. This project has only been tested using `Python 3.7`.
2. Install dependencies by running: `pip3 install -r requirements.txt`
3. Make sure that you have the necessary `nltk` data by running: `python3 -m nltk.downloader stopwords punkt wordnet`

## Usage

### 1. Getting the dataset

This project uses a dataset from Stanford University's [CS 276
course](http://web.stanford.edu/class/cs276/). To download and extract the
dataset, run:

```sh
cd Data
wget http://web.stanford.edu/class/cs276/pa/pa1-data.zip
tar -xvf pa1-data.zip
```

### 2. Indexing the dataset

Run:

```sh
python3 Build.py
```

This will load the dataset and create an inverted index of its contents. This
process can take a bit of time (~1 min 40 sec on our setup) and use a decent
amount of RAM (~2GB during our tests).

It will output an `inverted_index` file of about 80 MB, as well as a
`Filenames.json` file containing a mapping from document IDs to filenames.

### 3. Running the default queries

Run:

```sh
python3 Query.py
```

This will run each query in `Queries/dev_queries` and score its output against
the reference output (located in `Queries/dev_output`).

## Performance considerations: loading data and indexing

Our initial version of `Build.py` was pretty naive, and therefore too slow to
load and index the complete collection in a reasonable amount of time. In five
minutes, we could only load and index about one-tenth of the collection, which
was not satisfying.

We thus set out to improve its performance and started profiling its execution.

### Improvements implemented

We implemented a number of optimizations:

* **reducing memory I/O:** initially, we loaded all the files, then removed the
  stop words of each file, then lemmatized each word. We refactored this
  process, leveraging functional programming concepts (see the [`map_many`
  function](https://github.com/hugo-sv/riw/blob/b22301b45145f2ef23191d65042560f2de266a39/Build.py#L22-L27))
  to only do one memory write and one memory read per token (word).
* **miscellaneous Pythonic optimizations:** in various occurences, we achieved
  encouraging results by using somewhat faster Python directives (e.g. `key in
  dict` instead of `key in dict.keys()`) and data structures (e.g. plain `dict`
  instead of `OrderedDict`)
* **caching lemmatization results:** we obtained a very significant speed-up by
  [using memoization on the lemmatizing
  function](https://github.com/hugo-sv/riw/blob/b22301b45145f2ef23191d65042560f2de266a39/Build.py#L78)
  as the documents in the collection largely contain similar terms. The actual
  speed-up depends on the number of documents indexed -- we measured x4 for
  one-twentieth of the collection, and the improvement is undoubtedly many, many
  times higher for the whole collection. (We did not calculate the exact overall
  improvement on the whole collection, as the lemmatization without cache was
  taking an unreasonable amount of time to complete.)

To illustrate how similar the terms in the different documents are, we present
below the cache hit ratios obtained when indexing the entire collection:

|                LRU cache size                 | Cache hit ratio |
| :-------------------------------------------: | :-------------: |
| None / unbounded (actual size: 354 242 items) |      98.1%      |
|                 65 536 items                  |      97.1%      |
|                 32 768 items                  |      95.4%      |
|                 16 384 items                  |      92.4%      |
|                  8 192 items                  |      87.3%      |

### Possible future improvements

We used Python's `cProfile` module to measure the execution time of the
different parts of the function. Our observations are as follow.

![cProfile output viewed with snakeviz](https://user-images.githubusercontent.com/8351433/79143264-3766db00-7dbd-11ea-87c8-71937f861eab.png)

First, we spend (as expected) a significant portion of the execution time
waiting on file IOs. Since we are reading a large number of small files, we are
likely not limited by disk bandwidth but by disk latency. We expect that
**making concurrent disk IO requests would largely reduce overall IO time**.

Second, we start by reading all files (`loadData`, ~ 60 sec), only to process
their content in a second phase (`build_inverted_index`, ~ 40 sec). We should
**start processing file contents while waiting on disk IO** to speed up the
overall script execution time.

Finally, **memory usage could be optimized as well**, for example by inserting terms
directly in the inverted index, without saving them in an intermediary data
structure like
[`corpus`](https://github.com/hugo-sv/riw/blob/b22301b45145f2ef23191d65042560f2de266a39/Build.py#L42).
