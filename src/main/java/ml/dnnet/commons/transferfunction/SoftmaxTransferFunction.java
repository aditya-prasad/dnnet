package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

public class SoftmaxTransferFunction implements TransferFunction
{
    @Override
    public double calcValue(double x)
    {
        return 1.0;
    }

    @Override
    public double calcDerivative(double x)
    {
        return 0.0;
    }

    @Override
    public DoubleMatrix calcValue(DoubleMatrix vector)
    {
        double sum = 0.0;
        int length = vector.length;
        for (int i = 0; i < length; i++)
        {
            double x = Math.exp(-1.0 * vector.get(i));
            sum += x;
            vector.put(i, x);
        }
        vector.muli(1.0 / sum);
        return vector;
    }

    @Override
    public DoubleMatrix calcDerivative(DoubleMatrix vector)
    {
        DoubleMatrix y = calcValue(vector);
        vector = y.rsub(1.0);
        vector = y.mul(vector);
        return vector;
    }
}
