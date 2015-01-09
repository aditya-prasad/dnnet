package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

import java.io.Serializable;

public interface TransferFunction extends Serializable
{
    public double calcValue(final double x);
    public double calcDerivative(final double x);

    public DoubleMatrix calcValue(final DoubleMatrix vector);
    public DoubleMatrix calcDerivative(final DoubleMatrix vector);
}
