import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class G34HW2 {

    /*
    To read the input text file (e.g., inputPath) containing a clustering
    into the RDD fullClustering do:

    JavaPairRDD<Vector,Integer> fullClustering = sc.textFile(inputPath)
               .mapToPair(x -> strToTuple(x));
    */
    public static Tuple2<Vector, Integer> strToTuple (String str){
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        Vector point = Vectors.dense(data);
        Integer cluster = Integer.valueOf(tokens[tokens.length-1]);
        Tuple2<Vector, Integer> pair = new Tuple2<>(point, cluster);
        return pair;
    }

    public static void main(String[] args) throws IOException {

        //Checks the input parameters number
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_name number_of_clusters_k sample_size_per_cluster_t");
        }

        //Sets Spark configuration
        SparkConf conf = new SparkConf(true).setAppName("Homework2");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //Reads number of partitions and number of desired top results
        String inputFile = args[0];
        int k = Integer.parseInt(args[1]);
        int t = Integer.parseInt(args[2]);

        JavaRDD<String> docs = sc.textFile(inputFile).cache();
        long dataSize = docs.count();

        JavaPairRDD<Vector,Integer> fullClustering = docs.mapToPair(x -> strToTuple(x)).repartition(12).cache();

        Map<Integer, Long> clusterSizes = fullClustering
                .map(x -> x._2)
                .countByValue();

        Broadcast<Map<Integer, Long>> sharedClusterSizes = sc.broadcast(clusterSizes);

        Random rand = new Random();
        JavaPairRDD clusteringSampleRDD = fullClustering.filter((x) -> {
                double p = Math.min(t / sharedClusterSizes.getValue().get(x._2).doubleValue(), 1);
                return rand.nextDouble() <= p;
            }
        );

        Broadcast<List<Tuple2<Vector, Integer>>> clusteringSample = sc.broadcast(
                clusteringSampleRDD.collect());

        //******** Point 4 ********
        long start1 = System.currentTimeMillis();
        JavaRDD<Double> scoreRDD = fullClustering.map((x) -> {
            Vector point1 = x._1;
            int clusterIndex1 = x._2;

            double sumSqDistances = 0;
            double[] partialSqDistances = new double[k];
            for (Tuple2<Vector, Integer> tuple2 : clusteringSample.getValue()) {
                Vector point2 = tuple2._1;
                int clusterIndex2 = tuple2._2;

                if (clusterIndex1 == clusterIndex2) {
                    sumSqDistances += Vectors.sqdist(point1, point2);
                }

                if (clusterIndex1 != clusterIndex2) {
                    partialSqDistances[clusterIndex2] += Vectors.sqdist(point1, point2);
                }
            }

            double aApprox = sumSqDistances / Math.min(t, sharedClusterSizes.getValue().get(clusterIndex1).doubleValue());

            //Computes the avg distance and the min
            double bApprox = Integer.MAX_VALUE;
            for (int i = 0; i < k; i++) {
                if (i != clusterIndex1) {
                    bApprox = Math.min(partialSqDistances[i] / Math.min(t, sharedClusterSizes.getValue().get(i).doubleValue()), bApprox);
                }
            }

            return (bApprox - aApprox) / Math.max(aApprox, bApprox) / dataSize;
        });

        double approxSilhFull = scoreRDD.reduce((x, y) -> x + y);

        long end1 = System.currentTimeMillis();

        //******** Point 5 ********
        long start2 = System.currentTimeMillis();
        long sampleSize = clusteringSample.getValue().size();

        //Computes the size of each cluster in clusteringSample
        int[] clustersSize = new int[k];
        for (Tuple2<Vector, Integer> tuple : clusteringSample.getValue()) {
            int clusterIndex = tuple._2;
            clustersSize[clusterIndex] = clustersSize[clusterIndex] + 1;
        }


        double exactSilhSample = 0;
        for (Tuple2<Vector, Integer> tuple1 : clusteringSample.getValue()) {
            Vector point1 = tuple1._1;
            int clusterIndex1 = tuple1._2;

            double sum = 0;
            double[] clustersDistance = new double[k];
            for (Tuple2<Vector, Integer> tuple2 : clusteringSample.getValue()) {
                Vector point2 = tuple2._1;
                int clusterIndex2 = tuple2._2;

                if (!point1.equals(point2) && clusterIndex1 == clusterIndex2) {
                    sum += Vectors.sqdist(point1, point2);
                }

                if (clusterIndex1 != clusterIndex2) {
                    clustersDistance[clusterIndex2] += Vectors.sqdist(point1, point2);
                }
            }
            double a = sum / clustersSize[clusterIndex1];

            //Computes the avg distance and the min
            double b = Integer.MAX_VALUE;
            for (int i = 0; i < k; i++) {
                if (i != clusterIndex1) {
                    b = Math.min(clustersDistance[i] / clustersSize[i], b);
                }
            }

            double silhouettePoint1 = (b - a) / Math.max(a, b) / sampleSize;
            exactSilhSample += silhouettePoint1;
        }

        long end2 = System.currentTimeMillis();

        //******** Point 6 ********
        System.out.println("Value of approxSilhFull = " + approxSilhFull);
        System.out.println("Time to compute approxSilhFull = " + (end1 - start1) + " ms");
        System.out.println("Value of exactSilhSample = " + exactSilhSample);
        System.out.println("Time to compute exactSilhSample = " + (end2 - start2) + " ms");
    }
}