#!/usr/bin/env python
# coding: utf-8

# **Installation and Data Preprocessing**

# #### Run below two cells for Google Colab

# In[ ]:


# !apt-get install openjdk-8-jdk-headless -qq > /dev/null
# !wget -q https://www-us.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
# !tar xf spark-2.4.5-bin-hadoop2.7.tgz
# !pip install -q findspark


# In[ ]:


# import os
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-2.4.5-bin-hadoop2.7"


# In[ ]:


import findspark
# findspark.init() # If running on google colab
findspark.init('/home/cse587/spark-2.4.0-bin-hadoop2.7') # If running on Spark VM
from pyspark.sql import *
spark = SparkSession.builder.appName('Movie Genre Prediction').master('local[*]').config("spark.executor.memory", "16g").config("spark.driver.memory", "16g").getOrCreate()


# In[4]:


import pandas as pd
# import matplotlib.pyplot as plt


# In[ ]:


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
sqlCtx = SQLContext(spark)
train_data = sqlCtx.createDataFrame(train_dataset)
test_data = sqlCtx.createDataFrame(test_dataset)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


train_data.printSchema()


# In[ ]:


test_data.printSchema()


# In[ ]:


train_df = train_data.dropna()
test_df = test_data.dropna()


# In[ ]:


from pyspark.sql.functions import col, lower, regexp_replace, split, size, when, array_contains, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, IDF, StringIndexer, CountVectorizer, Word2Vec, HashingTF
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import *


# In[ ]:


mapping_data = spark.read.csv('mapping.csv', inferSchema=True, header=True)


# In[ ]:


train_df = train_df.withColumn("genre", regexp_replace("genre", "[\\[\\]']", "")).withColumn("genre", split(col("genre"), "\\s*,\\s*").cast("array<string>"))


# In[ ]:


train_df.printSchema()


# In[ ]:


genreData = mapping_data.collect()


# In[ ]:


genreMovieCount = []
genres = []
for genre in genreData:
    print(genre["_c0"], genre["0"])
    train_df = train_df.withColumn(str(genre["_c0"]), when(array_contains(col("genre"), genre["0"]), "1").otherwise("0"))
    genres.append(genre["0"]) 
    genreMovieCount.append(int(train_df.groupby(str(genre["_c0"])).count().collect()[1]["count"]))


# In[ ]:


train_df.show(5)


# In[ ]:


fig = plt.figure(figsize=(12,9), dpi= 80)
ax = plt.axes()

ax.bar(genres, genreMovieCount )
plt.xticks(rotation='vertical')

plt.title('Movie Count based on Genres', fontsize=20)
plt.xlabel('Movie Count', fontsize=16)
plt.ylabel('Genres', fontsize=16)


# In[ ]:


def clean_text(c):
  c = lower(c)
  c = regexp_replace(c, "^rt ", "")
  c = regexp_replace(c, "(https?\://)\S+", "")
  c = regexp_replace(c, "[^a-z\\s]", "")
  return c


# In[ ]:


train_df = train_df.withColumn("plot", clean_text(col("plot")).alias("plot"))
test_df = test_df.withColumn("plot", clean_text(col("plot")).alias("plot"))


# **Part 1 - HashingTF**

# In[ ]:


htf_models = []
htf_predictions = []
for genre in genreData:
    print(genre["_c0"], genre["0"])

    genre_label = StringIndexer(inputCol=str(genre["_c0"]), outputCol="label"+str(genre["_c0"]))
    tokenizer = Tokenizer(inputCol="plot", outputCol="tokenized_plot")
    # stopword_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_plot").setStopWords(stop_words)
    stopword_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_plot")
    hashTf = HashingTF(inputCol=stopword_remover.getOutputCol(), outputCol="features")
    lr = LogisticRegression(featuresCol=hashTf.getOutputCol(), labelCol="label"+str(genre["_c0"]), family="binomial", regParam=0.01, maxIter=200)
    pipeline = Pipeline(stages=[genre_label, tokenizer, stopword_remover, hashTf, lr])    

    model = pipeline.fit(train_df)
    prediction = model.transform(test_df)
    htf_models.append(model)
    htf_predictions.append(prediction)

    model.save("Part1Model_"+str(genre["_c0"]))


# #### Use below cell to load the saved model for hashingTf

# In[ ]:


htf_models = []
htf_predictions = []
for genre in genreData:
    print(genre["_c0"], genre["0"])
    model = PipelineModel.load("hashingtf/hashingModel_"+str(genre["_c0"]))
    prediction = model.transform(test_df)
    htf_models.append(model)
    htf_predictions.append(prediction)


# In[ ]:


genreMovieCount = []
for genre in range(len(genreData)):
    print(genre)
    try:
        groupby = htf_predictions[genre].groupby("prediction").count()
        groupby.show()
        genreMovieCount.append(int(groupby.collect()[1]["count"]))
    except:
        genreMovieCount.append(0)


# In[ ]:


fig = plt.figure(figsize=(12,9), dpi= 80)
ax = plt.axes()

ax.bar(genres, genreMovieCount )
plt.xticks(rotation='vertical')

plt.title('Movie Count based on Genres (HashingTF)', fontsize=20)
plt.xlabel('Movie Count', fontsize=16)
plt.ylabel('Genres', fontsize=16)


# In[ ]:


result_htf = test_df.select("movie_id", "movie_name")
for genre in range(len(genreData)):
    result_htf = result_htf.join(htf_predictions[genre].select("movie_id", "prediction").withColumn("prediction", col("prediction").cast(IntegerType())).withColumnRenamed("prediction", str(genre)), on=["movie_id"])


# In[ ]:


result_htf = result_htf.withColumn("predictions", concat_ws(" ", col("0"), col("1"), col("2"), col("3"), col("4"), col("5"), col("6"), col("7"), col("8"), col("9"), col("10"), col("11"), col("12"), col("13"), col("14"), col("15"), col("16"), col("17"), col("18"), col("19")))


# In[ ]:


result_htf.select("movie_id","predictions").repartition(1).write.csv("/content/drive/My Drive/ColabData/Part1_Result.csv", header = 'true')


# In[ ]:


result_htf.show()


# **Part 2 -TFIDF**

# In[ ]:


tfidf_models = []
tfidf_predictions = []
for genre in genreData:
    print(genre["_c0"], genre["0"])

    genre_label = StringIndexer(inputCol=str(genre["_c0"]), outputCol="label"+str(genre["_c0"]))
    tokenizer = Tokenizer(inputCol="plot", outputCol="tokenized_plot")
    stopword_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_plot")
    count_vectorizer = CountVectorizer(inputCol=stopword_remover.getOutputCol(), outputCol="plot_vector_count", minDF=5)
    idf = IDF(inputCol=count_vectorizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(featuresCol=idf.getOutputCol(), labelCol="label"+str(genre["_c0"]), maxIter=500)
    pipeline = Pipeline(stages=[genre_label, tokenizer, stopword_remover, count_vectorizer, idf, lr])

    model = pipeline.fit(train_df)
    prediction = model.transform(test_df)
    tfidf_models.append(model)
    tfidf_predictions.append(prediction)

    model.save("Part2Model_"+str(genre["_c0"]))


# #### Run below cell to load tf-idf model

# In[ ]:


tfidf_models = []
tfidf_predictions = []
for genre in genreData:
    print(genre["_c0"], genre["0"])
    model = PipelineModel.load("tf-idf/part2_"+str(genre["_c0"]))
    prediction = model.transform(test_df)
    tfidf_models.append(model)
    tfidf_predictions.append(prediction)


# In[ ]:


genreMovieCount = []
for genre in range(len(genreData)):
    print(genre)
    try:
        groupby = tfidf_predictions[genre].groupby("prediction").count()
        groupby.show()
        genreMovieCount.append(int(groupby.collect()[1]["count"]))
    except:
        genreMovieCount.append(0)


# In[ ]:


fig = plt.figure(figsize=(12,9), dpi= 80)
ax = plt.axes()

ax.bar(genres, genreMovieCount )
plt.xticks(rotation='vertical')

plt.title('Movie Count based on Genres (TF-IDF)', fontsize=20)
plt.xlabel('Movie Count', fontsize=16)
plt.ylabel('Genres', fontsize=16)


# In[ ]:


result_tfidf = test_df.select("movie_id", "movie_name")
for genre in range(len(genreData)):
    result_tfidf = result_tfidf.join(tfidf_predictions[genre].select("movie_id", "prediction").withColumn("prediction", col("prediction").cast(IntegerType())).withColumnRenamed("prediction", str(genre)), on=["movie_id"])


# In[ ]:


result_tfidf = result_tfidf.withColumn("predictions", concat_ws(" ", col("0"), col("1"), col("2"), col("3"), col("4"), col("5"), col("6"), col("7"), col("8"), col("9"), col("10"), col("11"), col("12"), col("13"), col("14"), col("15"), col("16"), col("17"), col("18"), col("19")))


# In[ ]:


result_tfidf.select("movie_id","predictions").repartition(1).write.csv("/content/drive/My Drive/ColabData/Part2_Result.csv", header = 'true')


# In[ ]:


result_tfidf.show()


# **Part 3 - Word2Vec**

# In[ ]:


tokenizer = Tokenizer(inputCol="plot", outputCol="tokenized_plot")
word2Vec = Word2Vec(inputCol=tokenizer.getOutputCol(), outputCol="features")
pipeline = Pipeline(stages=[tokenizer, word2Vec])


# In[ ]:


word2vec_model = pipeline.fit(train_df)


# In[ ]:


train = word2vec_model.transform(train_df)


# In[ ]:


word2vec_model2 = pipeline.fit(test_df)


# In[ ]:


test = word2vec_model2.transform(test_df)


# In[ ]:


word2vec_models = []
word2vec_predictions = []

for genre in genreData:
    print(genre["_c0"], genre["0"])

    genre_label = StringIndexer(inputCol=str(genre["_c0"]), outputCol="label"+str(genre["_c0"]))
    lr = LogisticRegression(featuresCol="features", labelCol="label"+str(genre["_c0"]), maxIter=800)
    pipeline = Pipeline(stages=[genre_label, lr])    

    model = pipeline.fit(train)
    prediction = model.transform(test)
    word2vec_models.append(model)
    word2vec_predictions.append(prediction)

    model.save("Part3Model_"+str(genre["_c0"]))


# #### Run below cell to load word2vec modal

# In[ ]:


word2vec_models = []
word2vec_predictions = []
for genre in genreData:
    print(genre["_c0"], genre["0"])
    model = PipelineModel.load("word2vec/word2vecModel_"+str(genre["_c0"]))
    prediction = model.transform(test)
    word2vec_models.append(model)
    word2vec_predictions.append(prediction)


# In[ ]:


genreMovieCount = []
for genre in range(len(genreData)):
    print(genre)
    try:
        groupby = word2vec_predictions[genre].groupby("prediction").count()
        groupby.show()
        genreMovieCount.append(int(groupby.collect()[1]["count"]))
    except:
        genreMovieCount.append(0)


# In[ ]:


fig = plt.figure(figsize=(12,9), dpi= 80)
ax = plt.axes()

ax.bar(genres, genreMovieCount )
plt.xticks(rotation='vertical')

plt.title('Movie Count based on Genres (Word2Vec)', fontsize=20)
plt.xlabel('Movie Count', fontsize=16)
plt.ylabel('Genres', fontsize=16)


# In[ ]:


result_word2vec = test_df.select("movie_id", "movie_name")
for genre in range(len(genreData)):
    result_word2vec = result_word2vec.join(word2vec_predictions[genre].select("movie_id", "prediction").withColumn("prediction", col("prediction").cast(IntegerType())).withColumnRenamed("prediction", str(genre)), on=["movie_id"])


# In[ ]:


result_word2vec = result_word2vec.withColumn("predictions", concat_ws(" ", col("0"), col("1"), col("2"), col("3"), col("4"), col("5"), col("6"), col("7"), col("8"), col("9"), col("10"), col("11"), col("12"), col("13"), col("14"), col("15"), col("16"), col("17"), col("18"), col("19")))


# In[ ]:


result_word2vec.select("movie_id","predictions").repartition(1).write.csv("/content/drive/My Drive/ColabData/Part3_Result.csv", header = 'true')


# In[ ]:


result_word2vec.show()

