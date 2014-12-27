package ml.dnnet.commons.data;

import org.jblas.DoubleMatrix;

import java.io.Serializable;

public class LabelledDataPoint implements Serializable
{
    DoubleMatrix x;
    DoubleMatrix y;

    public LabelledDataPoint(DoubleMatrix x, DoubleMatrix y)
    {
        this.x = x;
        this.y = y;
    }

    public DoubleMatrix getX()
    {
        return x;
    }

    public DoubleMatrix getY()
    {
        return y;
    }

    public String toString()
    {
        return "Input : " + x + "; Target Output : " + y;
    }
}
