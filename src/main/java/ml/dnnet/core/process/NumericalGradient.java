package ml.dnnet.core.process;

import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.util.ListUtil;
import ml.dnnet.core.NeuronLayer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class NumericalGradient implements Function<LabelledDataPoint, List<DoubleMatrix>>
{
    private static final double EPSILON = 0.00000001;

    private List<NeuronLayer> layers;

    public NumericalGradient(List<NeuronLayer> layers)
    {
        this.layers = layers;
    }

    @Override
    public List<DoubleMatrix> call(LabelledDataPoint labelledDataPoint) throws Exception
    {
        List<DoubleMatrix> weightDerivatives = ListUtil.initialize(layers.size(), new DoubleMatrix());

        DoubleMatrix input = labelledDataPoint.getX();
        DoubleMatrix targetOutput = labelledDataPoint.getY();

        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++)
        {
            NeuronLayer layer = layers.get(layerIndex);

            DoubleMatrix weights = layer.getWeights();
            DoubleMatrix weightDerivative = DoubleMatrix.zeros(weights.rows, weights.columns);

            for (int row = 0; row < weights.rows; row++)
            {
                for (int column = 0; column < weights.columns; column++)
                {
                    DoubleMatrix deltaWeights = DoubleMatrix.zeros(weights.rows, weights.columns);

                    deltaWeights.put(row, column, EPSILON);
                    layer.updateWeights(deltaWeights);

                    DoubleMatrix calculatedOutput1 = (new FeedForward(layers)).predict(input);

                    deltaWeights.put(row, column, -2.0 * EPSILON);
                    layer.updateWeights(deltaWeights);

                    DoubleMatrix calculatedOutput2 = (new FeedForward(layers)).predict(input);

                    double derivative = (calcCost(targetOutput, calculatedOutput1) - calcCost(targetOutput, calculatedOutput2)) / (2.0 * EPSILON);
                    weightDerivative.put(row, column, derivative);

                    deltaWeights.put(row, column, EPSILON);
                    layer.updateWeights(deltaWeights);
                }
            }

            weightDerivatives.set(layerIndex, weightDerivative);
        }

        return weightDerivatives;
    }

    public static double calcCost(DoubleMatrix target, DoubleMatrix calculated)
    {
        DoubleMatrix diff = target.sub(calculated);
        DoubleMatrix diffSq = diff.mul(diff);
        double cost = diffSq.sum();
        return 0.5 * cost;
    }
}
