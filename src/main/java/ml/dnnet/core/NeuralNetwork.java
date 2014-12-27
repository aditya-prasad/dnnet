package ml.dnnet.core;

import ml.dnnet.commons.data.LabelledData;
import ml.dnnet.commons.data.UnlabelledData;

public interface NeuralNetwork
{
    public void train(LabelledData labelledData) throws Exception;

    public LabelledData predict(UnlabelledData unlabelledData) throws Exception;

    static enum Mode
    {
        REGRESSION,
        CLASSIFICATION;
    }

}
