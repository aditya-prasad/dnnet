package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

public class SoftmaxTransferFunction implements TransferFunction
{
    @Override
    public double calcValue(final double x)
    {
        return 1.0;
    }

    @Override
    public double calcDerivative(final double x)
    {
        return 0.0;
    }

    @Override
    public DoubleMatrix calcValue(final DoubleMatrix vector)
    {
        double sum = 0.0;
        int length = vector.length;
        DoubleMatrix result = new DoubleMatrix(length);
        for (int i = 0; i < length; i++)
        {
            double x = Math.exp(-1.0 * vector.get(i));
            sum += x;
            result.put(i, x);
        }
        result.muli(1.0 / sum);
        return result;
    }

    @Override
    public DoubleMatrix calcDerivative(final DoubleMatrix vector)
    {
        DoubleMatrix y = calcValue(vector);
        DoubleMatrix result = y.rsub(1.0);
        result = y.mul(result);
        return result;
    }

    @Override
    public String toString()
    {
        return "SoftmaxTransferFunction";
    }
}
