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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

public class Main
{
    static Logger log = Log.getLogger(Main.class);

    public static void main(String args[]) throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

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

        nnet.gradientCheck(trainSet);
        log.info("Gradient Checking Complete");

        System.out.print("Start training (if Gradient checking was correct) ? [Y] ");
        String choice = br.readLine();
        if (!choice.isEmpty() && choice.toLowerCase().charAt(0) == 'n')
        {
            System.exit(0);
        }

        log.info("Training Started...\n");
        int iterations = nnet.train(trainSet);
        log.info("Training complete\n");

        if (iterations == -1)
        {
            log.info("Did not converge\n");
        }
        else
        {
            log.info("Converged in " + iterations + " iterations\n");
        }

        JavaRDD<DoubleMatrix> unlabelledData = (IO.fetchUnlabelledData(sc));
        UnlabelledData data = new UnlabelledData(unlabelledData);
        log.info("Unlabelled Data Loaded\n");

        LabelledData predicted = nnet.predict(data);
        log.info("Prediction Complete\n");

        List<LabelledDataPoint> result = predicted.getData().collect();
        for (LabelledDataPoint line : result)
        {
            System.out.println("Input : " + line.getX());
            System.out.println("Output : " + line.getY());
            System.out.println();
        }
    }
}
