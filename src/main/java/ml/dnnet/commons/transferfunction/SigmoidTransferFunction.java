package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

public class SigmoidTransferFunction implements TransferFunction
{
    @Override
    public double calcValue(double x)
    {
        return 1.0 / (1.0 + Math.exp(-1.0 * x));
    }

    @Override
    public double calcDerivative(double x)
    {
        return calcValue(x)*(1.0 - calcValue(x));
    }

    @Override
    public DoubleMatrix calcValue(DoubleMatrix vector)
    {
        int length = vector.length;
        for(int i=0; i<length; i++)
        {
            double x = vector.get(i);
            vector.put(i, calcValue(x));
        }
        return vector;
    }

    @Override
    public DoubleMatrix calcDerivative(DoubleMatrix vector)
    {
        int length = vector.length;
        for(int i=0; i<length; i++)
        {
            double x = vector.get(i);
            vector.put(i, calcDerivative(x));
        }
        return vector;
    }
}
