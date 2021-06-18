package edu.gsu;

import java.io.*;
import java.util.*;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PageRank
{
	public static final double dampingFactor= 0.85;

	public static class Mapper 
		extends org.apache.hadoop.mapreduce.Mapper<LongWritable, Text, IntWritable, DoubleWritable>
	{
		private Map<Integer, Double> rankValuesById;
		private final IntWritable outputKey= new IntWritable();
		private final DoubleWritable rankFactor= new DoubleWritable();

		@Override
    	protected void setup(Context context)
			throws IOException, InterruptedException
		{
			super.setup(context);
			rankValuesById= getPreviousValues(context);
			for ( var rankEntry : rankValuesById.entrySet() )
				System.out.println( rankEntry.getKey() + " " + rankEntry.getValue() );
		}

		public void map(LongWritable lineNumber, Text lineObj, Context context)
			throws IOException, InterruptedException
		{
			var adjacencyLine= new StringTokenizer( lineObj.toString() );
			if ( adjacencyLine.hasMoreTokens() )
			{
				int adjacencySource= Integer.parseInt( adjacencyLine.nextToken() ), // the following adjacencies point from this source
				    targetCount= adjacencyLine.countTokens(); // the number of adjacencies from the source to its targets
				double sourceRank= rankValuesById.get(adjacencySource); // the source's previous rank (from a cached map)
	
				while ( adjacencyLine.hasMoreTokens() )
				{
					int adjacencyTarget= Integer.parseInt( adjacencyLine.nextToken() ); // the adjacency points from the source to this target
					outputKey.set( adjacencyTarget ); // the output needs to be grouped by target
					rankFactor.set( sourceRank / targetCount ); // a factor of the power iteration method
					context.write(outputKey, rankFactor); // outputs to combiner etc...	
				}

			}
		}

		@Override
    	protected void cleanup(Context context)
			throws IOException, InterruptedException
		{
			rankValuesById= null;
			super.cleanup(context);
		}
	}

	public static class Reducer
		extends org.apache.hadoop.mapreduce.Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable>
	{
		private int rankValuesCount;
		private final DoubleWritable outputValue= new DoubleWritable();

		@Override
    	protected void setup(Context context)
			throws IOException, InterruptedException
		{
			super.setup(context);
			rankValuesCount= getPreviousValues(context).size(); // this could be further optimized, since we no longer need previous ranks here
		}

		public void reduce(IntWritable outputKey, Iterable<DoubleWritable> rankFactors, Context context)
			throws IOException, InterruptedException
		{
			double rank= ( 1 - dampingFactor ) / rankValuesCount;
			for ( DoubleWritable rankFactor : rankFactors )
				rank+= dampingFactor * rankFactor.get(); // dampened sum of all power iteration factors for the current target
			outputValue.set(rank); // output page rank of current iteration
			context.write(outputKey, outputValue);
		}
		
		@Override
    	protected void cleanup(Context context)
			throws IOException, InterruptedException
		{
			rankValuesCount= 0;
			super.cleanup(context);
		}
	}

	private static Map<Integer, Double> getPreviousValues(JobContext context)
		throws IOException
	{
		var map= new HashMap<Integer, Double>();
		Reader reader= null;
		try {
			URI[] cacheFiles= context.getCacheFiles();
			if ( cacheFiles != null && cacheFiles.length > 0 )
			{
				var fileSys= FileSystem.get(context.getConfiguration());
				reader= new BufferedReader(new InputStreamReader(fileSys.open( new Path( cacheFiles[0] ) )));
				
				// StreamTokenizer parses numbers by default
				var tokenizer= new StreamTokenizer(reader);
				while ( tokenizer.nextToken() == StreamTokenizer.TT_NUMBER )
				{
					int rankId= (int)tokenizer.nval;
					if ( tokenizer.nextToken() == StreamTokenizer.TT_NUMBER )
						map.put(rankId, tokenizer.nval);
					else break;
				}

				// handle invalid files
				if ( tokenizer.ttype != StreamTokenizer.TT_EOF )
					throw new IOException( cacheFiles[0].toString() );

				if ( tokenizer.lineno() != map.size()+1 ) // file was close to the expected format
					System.out.println("Warning: \"" + cacheFiles[0] + "\" doesn't have one rank entry per line. " + tokenizer.lineno() );
			}
		}
		catch ( Exception e ) {
			map.clear();
			e.printStackTrace();
		}
		finally {
			if ( reader != null )
				try {
					reader.close();
				} catch ( IOException e ) {}
		}
		return map;
	}

	/* // more optimized counter
	private static int countPreviousValues(JobContext context)
	{
		int count= 0;
		Reader reader= null;
		try {
			URI[] cacheFiles= context.getCacheFiles();
			if ( cacheFiles != null && cacheFiles.length > 0 )
			{
				var fileSystem= FileSystem.get(context.getConfiguration());
				reader= new BufferedReader(new InputStreamReader(FileSystem.open( new Path( cacheFiles[0] ) )));
				var tokenizer= new StreamTokenizer(reader);
				while ( tokenizer.nextToken() != StreamTokenizer.TT_EOF )
				{
					tokenizer.nextToken();
					++count;
				}
			}
		}
		catch ( Exception ) {	
		}
		finally {
			if ( reader != null )
				reader.close();
		}
		return count;
	}
	*/

	public static void main(String[] args) throws Exception
	{
		Path inputFilePath= new Path(args[0]), // previous rank values
		     adjacencyFilePath= new Path(args[1]),
			 outputDirectory= new Path(args[2]);

		int iterationCount= Integer.parseInt(args[3]),
			failureCount= 0;

		for ( int iterationIndex= 0; iterationIndex < iterationCount; ++iterationIndex )
		{
			System.out.println( "Iteration: " + iterationIndex );

			Configuration config= new Configuration();
			Job job= Job.getInstance(config, "Power Iteration Method");
			Path iterationOutputPath= new Path( outputDirectory, "iteration" + iterationIndex );

			job.setJarByClass(PageRank.class);
			job.setMapperClass(Mapper.class);
			job.setReducerClass(Reducer.class);
			job.setOutputKeyClass(IntWritable.class); // each hadoop output value has an integer index
			job.setOutputValueClass(DoubleWritable.class); // each hadoop output value is a double-precision floating point
			job.addCacheFile(inputFilePath.toUri());
			FileInputFormat.addInputPath(job, adjacencyFilePath);			
			FileOutputFormat.setOutputPath(job, iterationOutputPath );

			if ( ! job.waitForCompletion(true) )
				++failureCount;

			// set the next iteration's input file
			inputFilePath= new Path(iterationOutputPath, "part-r-00000");
		}

		System.exit(failureCount); // report the number of failed iterations
	}

}