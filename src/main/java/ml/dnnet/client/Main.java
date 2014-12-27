package ml.dnnet.client;

import ml.dnnet.commons.data.LabelledData;
import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.data.UnlabelledData;
import ml.dnnet.commons.io.IO;
import ml.dnnet.commons.util.Log;
import ml.dnnet.core.NeuralNetwork;
import ml.dnnet.core.Topography;
import ml.dnnet.core.impl.BackpropagationNeuralNetwork;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;

import java.util.List;

public class Main
{
    static Logger log = Log.getLogger(Main.class);

    public static void main(String args[]) throws Exception
    {
        SparkConf sparkConf = new SparkConf().setAppName("NeuralNetwork");
        SparkContext sc = new SparkContext(sparkConf);
        log.info("SparkContext initialized\n");

        JavaRDD<LabelledDataPoint> labelledData = IO.fetchLabelledData(sc);
        LabelledData trainSet = new LabelledData(labelledData);
        log.info("Input Loaded\n");

        List<Integer> topography = Topography.getTopography(trainSet);
        log.info("Topography Initialized\n");

        BackpropagationNeuralNetwork nnet = new BackpropagationNeuralNetwork(topography, NeuralNetwork.Mode.REGRESSION);
        log.info("Neural Network created\n");


        nnet.train(trainSet);
/*
        JavaRDD<DoubleMatrix> unlabelledData = (IO.fetchUnlabelledData(sc));
        UnlabelledData data = new UnlabelledData(unlabelledData);
        log.info("Unlabelled Data Loaded\n");

        LabelledData predicted = nnet.predict(data);
        log.info("Prediction Complete\n");

        List<LabelledDataPoint> result = predicted.getData().collect();
        for(LabelledDataPoint line:result)
        {
            System.out.println("Input : " + line.getX());
            System.out.println("Output : " + line.getY());
            System.out.println();
        }
*/
    }
}
