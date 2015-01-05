package ml.dnnet.commons.transferfunction;

import org.jblas.DoubleMatrix;

import java.io.Serializable;

public interface TransferFunction extends Serializable
{
    public double calcValue(double x);
    public double calcDerivative(double x);

    public DoubleMatrix calcValue(DoubleMatrix vector);
    public DoubleMatrix calcDerivative(DoubleMatrix vector);
}
