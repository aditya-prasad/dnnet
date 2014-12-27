package ml.dnnet.core;

import ml.dnnet.commons.data.LabelledData;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Topography
{
    public static List<Integer> getTopography(LabelledData labelledData) throws IOException
    {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

        System.out.print("Enter Topography for the hidden layers (comma separated) : ");
        List<Integer> topography = new ArrayList<>();
        String topoString = in.readLine();
        if (topoString != null && !topoString.isEmpty())
        {
            List<String> topoBits = Arrays.asList(topoString.split(","));
            topoBits.forEach(topoBit -> topography.add(Integer.parseInt(topoBit)));
        }
        topography.add(0, labelledData.getInputDemension());
        topography.add(labelledData.getOutputDimension());
        return topography;
    }
}
