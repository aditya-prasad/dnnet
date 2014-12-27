package ml.dnnet.core.impl;

import ml.dnnet.commons.data.LabelledData;
import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.data.UnlabelledData;
import ml.dnnet.commons.transferfunction.TransferFunction;
import ml.dnnet.commons.transferfunction.TransferFunctionFactory;
import ml.dnnet.core.Constants;
import ml.dnnet.core.NeuralNetwork;
import ml.dnnet.core.NeuronLayer;
import ml.dnnet.core.process.BackPropagate;
import ml.dnnet.core.process.FeedForward;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class BackpropagationNeuralNetwork implements NeuralNetwork
{
    private Mode mode;
    private List<Integer> topography;
    public List<NeuronLayer> layers;

    public BackpropagationNeuralNetwork(List<Integer> topography, Mode mode)
    {
        this.mode = mode;
        this.topography = topography;

        initLayers();
    }

    @Override
    public void train(LabelledData labelledData) throws Exception
    {
        System.out.println("\nTraining...\n");

        JavaRDD<LabelledDataPoint> data = labelledData.getData();
        JavaRDD<List<DoubleMatrix>> deltaWeightsRDD = data.map(new BackPropagate(layers));
        System.out.println(deltaWeightsRDD.collect());

        /*
        for (int epoch = 1; epoch <= Constants.MAX_EPOCHS; epoch++)
        {
            JavaRDD<LabelledDataPoint> data = labelledData.getData();

            for (int batchIndex = 0; batchIndex < 10; batchIndex++)
            {
                double sampleFraction = 1.0 / (10.0 - batchIndex);
                JavaRDD<LabelledDataPoint> currentBatch = data.sample(false, sampleFraction);
                data = data.subtract(currentBatch);

                JavaRDD<List<DoubleMatrix>> deltaWeightsRDD = currentBatch.map(new BackPropagate(layers));
                JavaRDD<List<DoubleMatrix>> aggregatedDeltaWeights = deltaWeightsRDD.reduce(new BackPropagate(layers));
            }
        }*/
    }

    @Override
    public LabelledData predict(UnlabelledData unlabelledData) throws Exception
    {
        FeedForward feedForward = new FeedForward(layers);
        JavaRDD<DoubleMatrix> input = unlabelledData.getData();
        JavaRDD<LabelledDataPoint> predicted = input.map(new FeedForward(layers));
        return new LabelledData(predicted);
    }

    private void initLayers()
    {
        TransferFunction sigmoid = TransferFunctionFactory.sigmoid();

        layers = new ArrayList<>();
        int layerCount = this.topography.size();
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
        {
            NeuronLayer neuronLayer = null;

            // Activation of the final layer depends on the execution mode
            if (layerIndex == layerCount - 1)
            {
                if (this.mode == Mode.REGRESSION)
                {
                    neuronLayer = new NeuronLayer(layerIndex, TransferFunctionFactory.linear(), topography);
                }
                else if (this.mode == Mode.CLASSIFICATION)
                {
                    if (topography.get(layerCount - 1) > 1)
                    {
                        neuronLayer = new NeuronLayer(layerIndex, TransferFunctionFactory.softmax(), topography);
                    }
                    else
                    {
                        neuronLayer = new NeuronLayer(layerIndex, sigmoid, topography);
                    }
                }
            }
            else if (layerIndex == 0)
            {
                neuronLayer = new NeuronLayer(layerIndex, TransferFunctionFactory.linear(), topography);
            }
            else
            {
                neuronLayer = new NeuronLayer(layerIndex, sigmoid, topography);
            }

            layers.add(neuronLayer);
        }
    }

}
