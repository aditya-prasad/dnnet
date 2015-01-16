package ml.dnnet.commons.util;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.PatternLayout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.OutputStreamWriter;

public class Log
{
    public static Logger getLogger(Class<?> clazz)
    {
        LogManager.getLogger(clazz).setLevel(Level.ALL);
        LogManager.getLogger(clazz).setAdditivity(false);

        ConsoleAppender appender = new ConsoleAppender();
        appender.setLayout(new PatternLayout("[%5p] %d{dd-MM-yyyy HH:mm:ss(SSS)} : %m%n"));
        appender.setWriter(new OutputStreamWriter(System.out));
        LogManager.getLogger(clazz).addAppender(appender);

        return LoggerFactory.getLogger(clazz);
    }
}
