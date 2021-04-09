import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G34HW1{

    public static void main(String[] args) throws IOException {

        //Checks the input parameters number
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: num_partitions num_top_results input_file_path");
        }

        //Sets Spark configuration
        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //Reads number of partitions and number of desired top results
        int K = Integer.parseInt(args[0]);
        int T = Integer.parseInt(args[1]);
        String inputFile = args[2];

        //Reads input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(inputFile).repartition(K).cache();

        /* NORMALIZED RATING RDD */
        JavaPairRDD<String, Float> normalizedRating;
        normalizedRating = rawData
                .mapToPair((review) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = review.split(",");
                    return new Tuple2<>(tokens[1], tokens[0] + "," + tokens[2]);
                })
                .groupByKey() // <-- REDUCE PHASE (R1)
                .flatMapToPair((element) -> {
                    ArrayList<Tuple2<String, Float>> products = new ArrayList<>();
                    float mean = 0;
                    int counter = 0;
                    for (String s : element._2()){
                        String[] list = s.split(",");
                        Float rating = Float.parseFloat(list[1]);
                        products.add(new Tuple2<>(list[0], rating));
                        mean += rating;
                        counter++;
                    }
                    mean /= counter;
                    ArrayList<Tuple2<String, Float>> normProducts = new ArrayList<>();
                    for (Tuple2<String, Float> t : products){
                        normProducts.add(new Tuple2<>(t._1, t._2 - mean));
                    }
                    return  normProducts.iterator();
                });

        /* MAX NORMALIZED RATING RDD */
        JavaPairRDD<String, Float> maxNormRatings = normalizedRating.reduceByKey((x, y) -> Math.max(x, y));

        /* MAX NORMALIZED RATING RDD (REVERSED) */
        List<Tuple2<Float, String>> topList = maxNormRatings
                .mapToPair(x -> x.swap())
                .sortByKey(false)
                .take(T);

        System.out.println("INPUT PARAMETERS: K=" + K + " T=" + T + " file=" + inputFile + "\n\nOUTPUT:");
        for (Tuple2<Float, String> t : topList) {
            System.out.println("Product " + t._2 + " maxNormRating " + t._1);
        }
    }
}