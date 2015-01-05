package ml.dnnet.core.process;

import ml.dnnet.commons.data.LabelledDataPoint;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

public class Test implements Function<LabelledDataPoint, DoubleMatrix>
{
    private FeedForward feedForward;

    public Test(FeedForward feedForward)
    {
        this.feedForward = feedForward;
    }

    @Override
    public DoubleMatrix call(LabelledDataPoint labelledDataPoint) throws Exception
    {
        DoubleMatrix input = labelledDataPoint.getX();
        DoubleMatrix output = feedForward.predict(input);
        DoubleMatrix target = labelledDataPoint.getY();
        DoubleMatrix diff = target.sub(output);
        diff.muli(diff);
        return diff;
    }
}
