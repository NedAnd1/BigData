import sys
from os import path
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

# initialize the Spark environment
context= SparkContext();
context.setLogLevel("ERROR");

# initialize the Spark SQL session
#  # spark= SparkSession(context)
spark= SparkSession.builder									\
                   .master("local")							\
                   .appName("KMeans")						\
                   .config( conf= context.getConf() )		\
                   .getOrCreate();

# the default value of `k`
k= 2;

# point to the input file, with a default file path if no input arg is provided
dataFilePath= path.expanduser("~/data/kmeans_input.txt");
if len(sys.argv) > 1 :
	dataFilePath= sys.argv[1] ;
	if  len(sys.argv) > 2  and  sys.argv[2].isdigit()  :
		k= int( sys.argv[2] ) ;  # allow the user to provide a different k-value


# Read the text data into Spark as a data frapne
inputData= spark.read.format("libsvm").load(dataFilePath);

# Creates a machine learning instance of the k-means algorithm
kmeans= KMeans( k= k, seed= 1 )  # setting the seed to a constant value ensures deterministic results

# Creates a model of the data fitted by the k-means algorithm
model= kmeans.fit(inputData);

# Retrieve the mean values i.e. the averages of the clusters
clusterAverages= model.clusterCenters();


# Predict the cluster for each data point
clusterData= model.transform(inputData);

# Show expected cluster values vs. predicted cluster values
clusterData.select("label", "prediction")									\
           .withColumn("prediction", clusterData.prediction + 1 )			\
           .groupBy("label", "prediction").count().orderBy("label")			\
           .withColumnRenamed("label", "Expected Cluster")					\
           .withColumnRenamed("prediction", "Predicted Cluster")			\
           .withColumnRenamed("count", "Count")								\
           .show();


# Output the cluster centers
i= 1;
for point in clusterAverages :
	print( "Cluster", i, "Center:", "(" + ", ".join(map(str, point)) + ")" )
	i+= 1;
