package ml.dnnet.commons.transferfunction;

public class TransferFunctionFactory
{
    public static TransferFunction sigmoid()
    {
        return new SigmoidTransferFunction();
    }

    public static TransferFunction softmax()
    {
        return new SoftmaxTransferFunction();
    }

    public static TransferFunction linear()
    {
        return new LinearTransferFunction();
    }
}
