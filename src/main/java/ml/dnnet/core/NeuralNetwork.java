package ml.dnnet.core;

import ml.dnnet.commons.data.LabelledData;
import ml.dnnet.commons.data.UnlabelledData;

public interface NeuralNetwork
{
    public int train(LabelledData labelledData) throws Exception;

    public LabelledData predict(UnlabelledData unlabelledData) throws Exception;

    public void gradientCheck(LabelledData labelledData) throws Exception;

    static enum Mode
    {
        REGRESSION,
        CLASSIFICATION;
    }

}
