package ml.dnnet.core.process;

import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.util.VectorUtil;
import ml.dnnet.core.NeuronLayer;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class FeedForward implements Function<DoubleMatrix, LabelledDataPoint>
{
    private List<NeuronLayer> layers;

    public FeedForward(List<NeuronLayer> layers)
    {
        this.layers = layers;
    }

    public DoubleMatrix predict(DoubleMatrix input) throws Exception
    {
        int layerCount = layers.size();
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
        {
            NeuronLayer currentLayer = layers.get(layerIndex);
            input = currentLayer.getActivations(currentLayer.getInputs(input));
        }

        input = VectorUtil.removeIntercept(input);

        return input;
    }

    public List<DoubleMatrix> getAllInputs(DoubleMatrix input)throws Exception
    {
        List<DoubleMatrix> inputs = new ArrayList<>();

        int layerCount = layers.size();
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
        {
            NeuronLayer currentLayer = layers.get(layerIndex);
            DoubleMatrix currentLayerInput = currentLayer.getInputs(input);
            inputs.add(currentLayerInput);
            input = currentLayer.getActivations(currentLayerInput);
        }

        return inputs;
    }

    @Override
    public LabelledDataPoint call(DoubleMatrix x) throws Exception
    {
        DoubleMatrix y = predict(x);
        return new LabelledDataPoint(x, y);
    }
}
