import sys
from os import path
from array import array
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row

# initialize the Spark environment
context= SparkContext();
context.setLogLevel("ERROR");

# initialize the Spark SQL session
#  # spark= SparkSession(context)
spark= SparkSession.builder									\
                   .master("local")							\
                   .appName("TweetCount")					\
                   .config( conf= context.getConf() )		\
                   .getOrCreate();


# intelligently figure out the file paths to the tweets and the geographic map based on user input
tweetsFilePath= path.expanduser("~/data/tweets.json");
geoMapFilePath= path.expanduser("~/data/cityStateMap.json");
if len(sys.argv) > 1 :
	if len(sys.argv) > 2 :

		if  "tweet" in sys.argv[2].lower()  or  "citystate" in sys.argv[1].lower()  :
			tweetsFilePath= sys.argv[2]
			geoMapFilePath= sys.argv[1];
		else :
			tweetsFilePath= sys.argv[1]
			geoMapFilePath= sys.argv[2];
	
	elif  "citystate" in sys.argv[1].lower()  :
		geoMapFilePath= sys.argv[1];
	else :
		tweetsFilePath= sys.argv[1];


# Read the Tweets into Spark as a data frame
if tweetsFilePath.lower().endswith(".json") :
	tweets= spark.read.json(tweetsFilePath);
else :
	tweets= context.textFile(tweetsFilePath)											\
                   .map( lambda line : line.split() )									\
                   .map( lambda tokens : Row( user= tokens[0], geo= tokens[1] ) )		\
                   .toDF();

# Read the Geographic Map into Spark as a data frame
if geoMapFilePath.lower().endswith(".json") :
	geoMap= spark.read.json(geoMapFilePath);
else :
	geoMap= context.textFile(geoMapFilePath)											\
                   .map( lambda line : line.split() )									\
                   .map( lambda tokens : Row( city= tokens[0], state= tokens[1] ) )		\
                   .toDF();

# Count the number of tweets by city 
tweetsByCity= tweets.groupBy("geo").count();

# Map the cities to states and for each state Add up their tweet counts
tweetsByState= tweetsByCity.join(geoMap, tweetsByCity.geo == geoMap.city )				\
                           .select(geoMap.state, tweetsByCity["count"])					\
                           .groupBy("state")											\
                           .sum("count")												\
                           .withColumnRenamed("sum(count)", "count");

print("\nTweets By State");
tweetsByState.show();

print("\nTweets By City");
tweetsByCity.show();