package ml.dnnet.commons.io;

import ml.dnnet.commons.data.LabelledDataPoint;
import ml.dnnet.commons.util.Log;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IO
{
    static Logger log = Log.getLogger(IO.class);

    public static JavaRDD<LabelledDataPoint> fetchLabelledData(SparkContext sc) throws IOException
    {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

        System.out.print("Enter Input file: ");
        String xSource = in.readLine();

        System.out.print("Enter Target Output file: ");
        String ySource = in.readLine();

        JavaRDD<String> xRawRDD = sc.textFile(xSource, sc.textFile$default$2()).toJavaRDD();
        JavaRDD<String> yRawRDD = sc.textFile(ySource, sc.textFile$default$2()).toJavaRDD();

        JavaRDD<List<String>> xRDD = xRawRDD.map(line -> Arrays.asList(line.split(",")));
        JavaPairRDD<Long, List<Double>> x = xRDD.mapToPair(new RawInputParser());

        JavaRDD<List<String>> yRDD = yRawRDD.map(line -> Arrays.asList(line.split(",")));
        JavaPairRDD<Long, List<Double>> y = yRDD.mapToPair(new RawInputParser());

        JavaPairRDD<Long, Tuple2<List<Double>, List<Double>>> input = x.join(y);

        JavaRDD<LabelledDataPoint> processedInput = input.map(new InputProcessor());
        return processedInput;
    }

    public static JavaRDD<DoubleMatrix> fetchUnlabelledData(SparkContext sc) throws IOException
    {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

        System.out.print("Enter Input file: ");
        String xSource = in.readLine();

        JavaRDD<String> xRawRDD = sc.textFile(xSource, sc.textFile$default$2()).toJavaRDD();

        JavaRDD<List<String>> xRDD = xRawRDD.map(line -> Arrays.asList(line.split(",")));

        JavaPairRDD<Long, List<Double>> x = xRDD.mapToPair(new RawInputParser());

        JavaRDD<DoubleMatrix> input = x.map(new Function<Tuple2<Long, List<Double>>, DoubleMatrix>()
        {
            @Override
            public DoubleMatrix call(Tuple2<Long, List<Double>> inputRow) throws Exception
            {
                return new DoubleMatrix(inputRow._2);
            }
        });

        return input;
    }

    private static class RawInputParser implements PairFunction<List<String>, Long, List<Double>>
    {
        @Override
        public Tuple2<Long, List<Double>> call(List<String> strings) throws Exception
        {
            List<Double> value = new ArrayList<Double>();
            long key = -1;
            boolean flag = true;

            for (String s : strings)
            {
                if (flag)
                {
                    key = Long.parseLong(s);
                    flag = false;
                }
                else
                {
                    value.add(Double.parseDouble(s));
                }

            }

            return new Tuple2<Long, List<Double>>(key, value);
        }
    }

    private static class InputProcessor implements Function<Tuple2<Long, Tuple2<List<Double>, List<Double>>>, LabelledDataPoint>
    {
        @Override
        public LabelledDataPoint call(Tuple2<Long, Tuple2<List<Double>, List<Double>>> input) throws Exception
        {
            Tuple2<List<Double>, List<Double>> values = input._2();
            List<Double> inVector = values._1();
            List<Double> outVector = values._2();

            DoubleMatrix in = new DoubleMatrix(inVector);
            DoubleMatrix out = new DoubleMatrix(outVector);

            return new LabelledDataPoint(in, out);
        }
    }
}
