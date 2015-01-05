package ml.dnnet.commons.data;

import org.apache.spark.api.java.JavaRDD;
import org.jblas.DoubleMatrix;

public class UnlabelledData
{
    private JavaRDD<DoubleMatrix> data;
    private int inputDemension;

    public UnlabelledData(JavaRDD<DoubleMatrix> data)
    {
        this.data = data;
        init();
    }

    private void init()
    {
        DoubleMatrix dummy = data.first();
        inputDemension = dummy.length;
    }

    public JavaRDD<DoubleMatrix> getData()
    {
        return data;
    }

    public int getInputDemension()
    {
        return inputDemension;
    }
}
