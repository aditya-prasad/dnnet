package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

public class SigmoidTransferFunction implements TransferFunction
{
    @Override
    public double calcValue(final double x)
    {
        return 1.0 / (1.0 + Math.exp(-1.0 * x));
    }

    @Override
    public double calcDerivative(final double x)
    {
        return calcValue(x) * (1.0 - calcValue(x));
    }

    @Override
    public DoubleMatrix calcValue(final DoubleMatrix vector)
    {
        int length = vector.length;
        DoubleMatrix result = new DoubleMatrix(length);
        for (int i = 0; i < length; i++)
        {
            double x = vector.get(i);
            result.put(i, calcValue(x));
        }
        return result;
    }

    @Override
    public DoubleMatrix calcDerivative(DoubleMatrix vector)
    {
        int length = vector.length;
        DoubleMatrix result = new DoubleMatrix(length);
        for (int i = 0; i < length; i++)
        {
            double x = vector.get(i);
            result.put(i, calcDerivative(x));
        }
        return result;
    }

    @Override
    public String toString()
    {
        return "SigmoidTransferFunction";
    }
}
