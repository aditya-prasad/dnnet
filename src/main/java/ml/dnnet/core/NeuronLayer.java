package ml.dnnet.core;

import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.transferfunction.TransferFunction;
import ml.dnnet.commons.util.VectorUtil;
import org.jblas.DoubleMatrix;

import javax.xml.transform.Source;
import java.io.Serializable;
import java.util.List;

public class NeuronLayer implements Serializable
{
    private int layerId;
    private int layerCount;
    private TransferFunction transferFunction;
    private DoubleMatrix weights;

    public NeuronLayer(int layerId, TransferFunction transferFunction, List<Integer> topography)
    {
        this.layerId = layerId;
        this.layerCount = topography.size();
        this.transferFunction = transferFunction;

        if (!isInput())
        {
            int prevLayerNeuronCount = topography.get(layerId - 1);
            int neuronCount = topography.get(layerId);

            weights = DoubleMatrix.rand(neuronCount, (prevLayerNeuronCount + 1));
            weights.subi(0.5);
            weights.muli(2.0);
        }
        else
        {
            weights = new DoubleMatrix();
        }
    }

    public int getLayerId()
    {
        return layerId;
    }

    public boolean isInput()
    {
        return (layerId == 0);
    }

    public boolean isOutput()
    {
        return (layerId == layerCount - 1);
    }

    public DoubleMatrix getActivations(DoubleMatrix input)
    {
        if (!isInput())
        {
            DoubleMatrix activation = transferFunction.calcValue(input);
            activation = VectorUtil.appendIntercept(activation);
            return activation;
        }
        else
        {
            input = VectorUtil.appendIntercept(input);
            return input;
        }
    }

    public DoubleMatrix getInputs(DoubleMatrix prevLayerActivations)
    {
        if (!isInput())
        {
            //System.out.println(weights + " X " + prevLayerActivations);
            DoubleMatrix inputs = weights.mmul(prevLayerActivations);
            return inputs;
        }
        else
        {
            return prevLayerActivations;
        }
    }

    public DoubleMatrix getWeights()
    {
        return weights;
    }

    public TransferFunction getTransferFunction()
    {
        return transferFunction;
    }

    public void updateWeights(DoubleMatrix deltaWeights)
    {
        if (!isInput())
        {
            weights.addi(deltaWeights);
        }
    }

    public void print()
    {
        System.out.print("Layer: " + layerId + " (" + transferFunction.toString() + ")");
        if (isInput())
        {
            System.out.println(" [Input]");
        }
        else if (isOutput())
        {
            System.out.println(" [Output]");
        }
        else
        {
            System.out.println();
        }
        int rowCount = weights.rows;
        if(rowCount == 0)
        {
            System.out.println("[No Weights]");
        }
        for(int i=0; i<rowCount; i++)
        {
            System.out.println(weights.getRow(i));
        }
        System.out.println();
    }
}
