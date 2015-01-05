package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

public class LinearTransferFunction implements TransferFunction
{
    @Override
    public double calcValue(double x)
    {
        return x;
    }

    @Override
    public double calcDerivative(double x)
    {
        return 1.0;
    }

    @Override
    public DoubleMatrix calcValue(DoubleMatrix vector)
    {
        return vector;
    }

    @Override
    public DoubleMatrix calcDerivative(DoubleMatrix vector)
    {
        return DoubleMatrix.ones(vector.length);
    }

    @Override
    public String toString()
    {
        return "LinearTransferFunction";
    }
}
