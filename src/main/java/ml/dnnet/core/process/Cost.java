package ml.dnnet.core.process;

import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.core.NeuralNetwork;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

public class Cost implements Function<LabelledDataPoint, Double>
{
    private FeedForward feedForward;
    private NeuralNetwork.Mode mode;

    public Cost(NeuralNetwork.Mode mode, FeedForward feedForward)
    {
        this.mode = mode;
        this.feedForward = feedForward;
    }

    @Override
    public Double call(LabelledDataPoint labelledDataPoint) throws Exception
    {
        DoubleMatrix input = labelledDataPoint.getX();
        DoubleMatrix output = feedForward.predict(input);
        DoubleMatrix target = labelledDataPoint.getY();

        if (mode == NeuralNetwork.Mode.REGRESSION)
        {
            DoubleMatrix diff = target.sub(output);
            diff.muli(diff);
            double cost = diff.sum();
            cost /= 2.0;
            return cost;
        }
        else
        {
            double cost = 0.0;
            for (int i = 0; i < output.length; i++)
            {
                cost += (-1.0 * (target.get(i) * Math.log(output.get(i))));
            }
            return cost;
        }
    }
}
