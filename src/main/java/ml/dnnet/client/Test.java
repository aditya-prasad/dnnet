package ml.dnnet.client;

import java.io.BufferedReader;
import java.io.FileReader;

public class Test
{
    public static void main(String[] args)throws Exception
    {
        BufferedReader br = new BufferedReader(new FileReader("D:/codex/xor_in.csv"));
        String line;
        while((line = br.readLine()) != null)
        {
            String[] tokens = line.split(",");
            int index = Integer.parseInt(tokens[0]);
            double x1 = Double.parseDouble(tokens[1]);
            double x2 = Double.parseDouble(tokens[2]);
            if((x1 + x2) == 1.0)
            {
                System.out.println(index + "," + 1.0);
            }
            else
            {
                System.out.println(index + "," + 0.0);
            }
        }
    }
}
