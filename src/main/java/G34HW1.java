import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class G34HW1{

    public static final String inputFile = "input_20K.csv";
    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: num_partitions, <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);
        int T = Integer.parseInt(args[1]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(inputFile).repartition(K).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


        JavaPairRDD<String, Float> normalizedRating;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // STANDARD WORD COUNT with reduceByKey
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        normalizedRating = rawData
                .mapToPair((review) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = review.split(",");
                    return new Tuple2<>(tokens[1],tokens[0]+","+tokens[2]);
                })
                .groupByKey() // <-- REDUCE PHASE (R1)
                .flatMapToPair( (element) -> {
                    ArrayList<Tuple2<String, Float>> products = new ArrayList<>();
                    float mean = 0;
                    int counter=0;
                    for (String s : element._2()){
                        String[] list= s.split(",");
                        Float rating =Float.parseFloat(list[1]);
                        products.add(new Tuple2<>( list[0],rating ));
                        mean+= rating;
                        counter++;
                    }
                    mean/=counter;
                    ArrayList<Tuple2<String, Float>> normProducts = new ArrayList<>();
                    for (Tuple2<String, Float> t : products ){
                        normProducts.add(new Tuple2<>( t._1 , t._2-mean));
                    }
                    return  normProducts.iterator();
                });
        JavaPairRDD<String, Float> maxNormRatings = normalizedRating.reduceByKey( (x, y) -> Math.max(x, y) );

                //.mapToPair( (element) ->  new Tuple2<>(element._2,element._1))
        List<Tuple2<Float, String>> topList= maxNormRatings.mapToPair(x -> x.swap())
                .sortByKey(false)
                .take(T);
        File f = new File(args[2]);
        boolean fExists =f.createNewFile();

        FileWriter fw = new FileWriter( args[2] );

        fw.write("INPUT PARAMETERS: K="+K+" T="+T+" file="+inputFile+"\n\nOUTPUT:\n" );
        for(Tuple2<Float, String> t : topList ){
            fw.write("Product "+t._2+" maxNormRating "+t._1+"\n");
        }
        fw.close();


    }

}