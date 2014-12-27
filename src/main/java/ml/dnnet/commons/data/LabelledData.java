package ml.dnnet.commons.data;

import org.apache.spark.api.java.JavaRDD;

public class LabelledData
{
    private JavaRDD<LabelledDataPoint> data;
    private int inputDemension;
    private int outputDimension;

    public LabelledData(JavaRDD<LabelledDataPoint> data)
    {
        this.data = data;
        init();
    }

    private void init()
    {
        LabelledDataPoint dummy = data.first();
        inputDemension = dummy.getX().rows;
        outputDimension = dummy.getY().rows;
    }

    public JavaRDD<LabelledDataPoint> getData()
    {
        return data;
    }

    public int getInputDemension()
    {
        return inputDemension;
    }

    public int getOutputDimension()
    {
        return outputDimension;
    }
}
