package ml.dnnet.commons.util;

import java.util.ArrayList;
import java.util.List;

public class ListUtil
{
    public static <T> List<T> initialize(int capacity)
    {
        List<T> list = new ArrayList<>();
        for(int i=0; i<capacity; i++)
        {
            list.add(null);
        }
        return list;
    }

    public static <T> List<T> initialize(int capacity, T defaultValue)
    {
        List<T> list = new ArrayList<>();
        for(int i=0; i<capacity; i++)
        {
            list.add(defaultValue);
        }
        return list;
    }
}
