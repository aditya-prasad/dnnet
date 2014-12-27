package ml.dnnet.core.process;

import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.transferfunction.TransferFunction;
import ml.dnnet.commons.util.ListUtil;
import ml.dnnet.commons.util.VectorUtil;
import ml.dnnet.core.NeuronLayer;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

import java.util.List;

public class BackPropagate implements Function<LabelledDataPoint, List<DoubleMatrix>>
{
    private List<NeuronLayer> layers;
    private FeedForward feedForward;

    public BackPropagate(List<NeuronLayer> layers)
    {
        this.layers = layers;
        this.feedForward = new FeedForward(layers);
    }

    public List<DoubleMatrix> run(DoubleMatrix input, DoubleMatrix targetOutput) throws Exception
    {
        List<DoubleMatrix> layerInputs = this.feedForward.getAllInputs(input);

        DoubleMatrix calculatedOutput = layers.get(this.layers.size() - 1).getActivations(layerInputs.get(layerInputs.size() - 1));
        calculatedOutput = VectorUtil.removeIntercept(calculatedOutput);

        List<DoubleMatrix> deltas = ListUtil.initialize(this.layers.size());
        deltas.set(0, new DoubleMatrix());

        List<DoubleMatrix> deltaWeights = ListUtil.initialize(this.layers.size());
        deltaWeights.set(0, new DoubleMatrix());

        int layerCount = this.layers.size();
        for (int layerIndex = layerCount - 1; layerIndex >= 0; --layerIndex)
        {
            if (layerIndex == layerCount - 1)
            {
                DoubleMatrix delta = calculatedOutput.sub(targetOutput);
                deltas.set(layerIndex, delta);
            }
            else
            {
                NeuronLayer currentLayer = layers.get(layerIndex);
                NeuronLayer nextLayer = layers.get(layerIndex + 1);
                TransferFunction transferFunction = currentLayer.getTransferFunction();

                DoubleMatrix layerInput = layerInputs.get(layerIndex);
                DoubleMatrix nextLayerDelta = deltas.get(layerIndex + 1);

                if (layerIndex != 0)
                {
                    DoubleMatrix inputDerivative = transferFunction.calcDerivative(layerInput);

                    DoubleMatrix weightMatrix = nextLayer.getWeights();
                    DoubleMatrix weightMatrixTranspose = weightMatrix.transpose();
                    DoubleMatrix deltaPartOne = weightMatrixTranspose.mmul(nextLayerDelta);
                    deltaPartOne = VectorUtil.removeIntercept(deltaPartOne);

                    DoubleMatrix delta = deltaPartOne.mul(inputDerivative);
                    deltas.set(layerIndex, delta);
                }

                DoubleMatrix activation = currentLayer.getActivations(layerInput);
                DoubleMatrix activationTranspose = activation.transpose();
                DoubleMatrix deltaWeight = nextLayerDelta.mmul(activationTranspose);
                deltaWeights.set(layerIndex + 1, deltaWeight);
            }
        }

        return deltaWeights;
    }

    @Override
    public List<DoubleMatrix> call(LabelledDataPoint labelledDataPoint) throws Exception
    {
        DoubleMatrix input = labelledDataPoint.getX();
        DoubleMatrix output = labelledDataPoint.getY();
        return run(input, output);
    }
}