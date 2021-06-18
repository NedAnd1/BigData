import sys
from array import array
from pyspark import SparkContext, SparkConf

context= SparkContext();
context.setLogLevel("ERROR");

# Split each line into whitespace-separated tokens...
  # Each line becomes a map entry w/ its 1st token as the key and other tokens as values
adjacencyList= context.textFile(sys.argv[1])																\
                      .map( lambda line : line.split() )													\
                      .map( lambda tokens :  ( int(tokens[0]), array('l', map(int,tokens[1:])) )  )  		\
                      .persist();                              # each key han an array of 32-bit ints

print("\nAdjacency List");
for  ( key, values ) in adjacencyList.collect()  :
	print(  key,  ':',  ", ".join( map(str, values) )  );
# the keys represent the source pages of each adjancency
    # while the values are outlinks (representing their target pages)

nodeCount= adjacencyList.count();
rankValuesById= adjacencyList.mapValues( lambda outlinks : 1.0 / nodeCount );
                # each rank value starts out as an equal fraction of 1.0

outputPath= "";
if len(sys.argv) > 2 :
	outputPath= sys.argv[2]; 

iterationCount= 30;
if len(sys.argv) > 3 :
	iterationCount= int( sys.argv[3] );

dampingFactor= 0.85;
if len(sys.argv) > 4 :
	dampingFactor= float( sys.argv[4] );

# function to map the rank and outlinks of each page to a set of their targets' rank factors
def pageRankMapper( __sourceId__rankOutlinksTuple__ ) :  # python 3 no longer supports tuple parameters
	(  sourceId,  ( sourceRank, outlinks )  ) =  __sourceId__rankOutlinksTuple__;
	rankFactor= sourceRank / len(outlinks);  # each outlink gets an equal proportion of the source page's rank
	return [ ( outlink, rankFactor ) for outlink in outlinks ];


for iterationIndex in range(0, iterationCount) :
	print( "\nIntermediate PageRanks for Iteration %d" % ( iterationIndex + 1 ) );
	for  ( id, rankValue ) in rankValuesById.collect()  :
		print( id, ':', rankValue );

	# after joining the page ranks and adjacencies
	    # maps the ranks and targets of each source to a set of rank factors for each outlink
	rankFactorsById= rankValuesById.join(adjacencyList)													\
	                               .flatMap(pageRankMapper);

	# sums and dampens the rank factors for each page into rank values
	rankValuesById= rankFactorsById.reduceByKey( lambda a, b : a + b )																			\
	                               .mapValues( lambda rankFactor :  ( 1 - dampingFactor ) / nodeCount + dampingFactor * rankFactor  );

print("\n=== Final PageRanks ===");
for  ( id, rankValue ) in rankValuesById.collect()  :
	print( id, ':', rankValue );

if outputPath :  # Saves the final PageRanks to a text file in the given output directory
	rankValuesById.coalesce(1).saveAsTextFile(outputPath);