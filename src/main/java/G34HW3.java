import javafx.util.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.KMeans;
import scala.Tuple2;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class G34HW3 {

    public static Vector strToTuple (String str){
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static Pair<Double, Long> approxSilhouette(JavaSparkContext sc, JavaPairRDD<Vector, Integer> rdd, double t, int k){

        long start = System.currentTimeMillis();

        Map<Integer, Long> clusterSizes = rdd
                .map(x -> x._2)
                .countByValue();

        long dataSize = rdd.count();

        Broadcast<Map<Integer, Long>> sharedClusterSizes = sc.broadcast(clusterSizes);

        Random rand = new Random();
        JavaPairRDD<Vector, Integer> clusteringSampleRDD = rdd.filter((x) -> {
                    double p = Math.min(t / sharedClusterSizes.getValue().get(x._2).doubleValue(), 1);
                    return rand.nextDouble() <= p;
                }
        );

        Broadcast<List<Tuple2<Vector, Integer>>> clusteringSample = sc.broadcast(
                clusteringSampleRDD.collect());


        JavaRDD<Double> scoreRDD = rdd.map((x) -> {
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

            return (bApprox - aApprox) / Math.max(aApprox, bApprox);
        });

        //approx sil score
        double approxSil = scoreRDD.reduce(Double::sum) / dataSize;

        long end = System.currentTimeMillis();

        return new Pair<>(approxSil, end - start);
    }


    public static void main(String[] args) throws IOException {

        //Check the number of input parameters
        if (args.length != 6) {
            throw new IllegalArgumentException("USAGE: filename kstart h iter M L");
        }

        String filename = args[0];

        //initial number of clusters
        int kstart = Integer.parseInt(args[1]);

        //number of values of k that the program will test
        int h = Integer.parseInt(args[2]);

        //number of iterations of Lloyd's algorithm
        int iter = Integer.parseInt(args[3]);

        //expected size of the sample used to approximate the silhouette coefficient
        int M = Integer.parseInt(args[4]);

        //number of partitions of the RDDs containing the input points and their clustering
        int L = Integer.parseInt(args[5]);


        //Set up Spark configuration
        SparkConf conf = new SparkConf(true).setAppName("Homework3")
                .set("spark.locality.wait", "0s");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");



        JavaRDD<Vector> inputPoints = sc.textFile(filename).map(G34HW3::strToTuple).
                repartition(L).cache();

        long start = System.currentTimeMillis();
        long dataSize = inputPoints.count();
        long readingTime = System.currentTimeMillis() - start;

        System.out.println("Time for input reading = " + readingTime + "\n");

        for(int k = kstart; k <= kstart + h - 1; k++){

            long startClustering = System.currentTimeMillis();
            KMeansModel model = KMeans.train(inputPoints.rdd(), k, iter);
            Vector[] centers = model.clusterCenters();

            JavaPairRDD<Vector, Integer> currentClustering = inputPoints.mapToPair((x) -> {

                double minLen = Double.MAX_VALUE;
                int clusterIndex = 0;

                for(int i = 0; i < centers.length; i++){
                    double len = Vectors.sqdist(x, centers[i]);
                    if(len < minLen){
                        clusterIndex = i;
                        minLen = len;
                    }
                }
                return new Tuple2<>(x, clusterIndex);
            });
            currentClustering.cache();
            long endClustering = System.currentTimeMillis();
            Pair<Double, Long> pair = approxSilhouette(sc, currentClustering, M/ (double) k, k);

            System.out.println("Number of clusters k = " + k);
            System.out.println("Silhouette coefficient = " + pair.getKey());
            System.out.println("Time for clustering = " + (endClustering - startClustering));
            System.out.println("Time for silhouette computation = " + pair.getValue() + "\n");
        }


    }


}
