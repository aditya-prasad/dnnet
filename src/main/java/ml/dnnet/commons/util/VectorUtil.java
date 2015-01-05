package ml.dnnet.commons.util;

import org.jblas.DoubleMatrix;

public class VectorUtil
{
    public static DoubleMatrix appendIntercept(DoubleMatrix d)
    {
        DoubleMatrix result = DoubleMatrix.ones(d.length + 1);

        for (int i = 0; i < d.length; i++)
        {
            result.put(i, d.get(i));
        }

        return result;
    }

    public static DoubleMatrix removeIntercept(DoubleMatrix d)
    {
        DoubleMatrix result = DoubleMatrix.ones(d.length - 1);

        for (int i = 0; i < result.length; i++)
        {
            result.put(i, d.get(i));
        }

        return result;
    }
}
