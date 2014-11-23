package org.wso2.siddhi.performance;

import java.util.concurrent.Executors;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.stream.StreamJunction;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.stream.output.StreamCallback;

public class ComplexFilterSingleQueryPerformance
{
	private static int count = 0;
    private static int eventCount = 0;
    private static int prevEventCount = 0;
    private static volatile long start = System.currentTimeMillis();
    
    private static Options cliOptions;
    
	private static void Help() 
	{
		// This prints out some help
		HelpFormatter formater = new HelpFormatter();

		formater.printHelp("ComplexFilterSingleQueryPerformance", cliOptions);
		System.exit(0);
	}

    public static void main(String[] args) throws InterruptedException {
    	
    	cliOptions = new Options();
    	cliOptions.addOption("g", "enable-gpu", true, "Enable GPU processing");
    	cliOptions.addOption("r", "ringbuffer-size", true, "Disruptor RingBuffer size - in power of two");
    	cliOptions.addOption("t", "threadpool-size", true, "Executor service pool size");
    	cliOptions.addOption("b", "events-per-tblock", true, "Number of Events per thread block in GPU");

    	CommandLineParser cliParser = new BasicParser();
    	CommandLine cmd = null;
    	
    	boolean gpuEnabled = false;
    	int defaultBufferSize = 1024;
    	int threadPoolSize = 4;
    	int eventBlockSize = 256;
    	
		try 
		{
			cmd = cliParser.parse(cliOptions, args);

			if (cmd.hasOption("g"))
			{
				gpuEnabled = Boolean.parseBoolean(cmd.getOptionValue("g"));
			}

			if (cmd.hasOption("r")) 
			{
				defaultBufferSize = Integer.parseInt(cmd.getOptionValue("r"));
			} 
			
			if (cmd.hasOption("t")) 
			{
				threadPoolSize = Integer.parseInt(cmd.getOptionValue("t"));
			} 

			if (cmd.hasOption("b")) 
			{
				eventBlockSize = Integer.parseInt(cmd.getOptionValue("b"));
			} 
			
		} 
		catch (ParseException e) 
		{
			e.printStackTrace();
			Help();
		}
    	    	   	
    	System.out.println("Siddhi.Config [GPUEnabled=" + gpuEnabled + "|RingBufferSize=" + defaultBufferSize + 
    			"|ThreadPoolSize=" + threadPoolSize + "|EventBlockSize=" + eventBlockSize + "]");
    	
        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setDefaultEventBufferSize(defaultBufferSize);
        siddhiManager.getSiddhiContext().setExecutorService(Executors.newFixedThreadPool(threadPoolSize));

        String cseEventStream = "@config(async = 'true') define stream cseEventStream (symbol string, price float, volume int, change float, pctchange float);";
        StringBuilder sb = new StringBuilder();
        sb.append("@info(name = 'query1') ");
        if(gpuEnabled)
        {
        	sb.append("@gpu(filter='true', blocksize='").append(eventBlockSize).append("') ");
        }
        sb.append("from cseEventStream[pctchange > 0.1 and change < 2.5 and volume > 100 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;");
        
        String query1 = sb.toString();
        
        System.out.println("Stream def   = [ " + cseEventStream + " ]");
        System.out.println("Filter query = [ " + query1 + " ]");

        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(cseEventStream + query1);

        executionPlanRuntime.addCallback("outputStream", new StreamCallback() {
            @Override
            public void receive(Event[] inEvents) {
            	eventCount += inEvents.length;
                count++;
                if (count % 10000000 == 0) {
                    long end = System.currentTimeMillis();
                    //double tp = (10000000 * 1000.0 / (end - start));
                    double tp = ((eventCount - prevEventCount) * 1000.0) / (end - start);
                    System.out.println("Throughput = " + tp + " Event/sec " + (eventCount - prevEventCount));
                    //System.out.println("," + tp);
                    start = end;
                    prevEventCount = eventCount;
                }
            }

        });

        for (StreamJunction streamJunction : executionPlanRuntime.getStreamJunctions().values()) {
            streamJunction.startProcessing();
        }

        InputHandler inputHandler = executionPlanRuntime.getInputHandler("cseEventStream");
        while (true) {
            inputHandler.send(new Object[]{"WSO2", 55.6f, 1000, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100, 2.4f, 0.05f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 500, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 50, 2.5f, 0.2f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 200, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 55.6f, 1000, 2.4f, 0.05f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 300, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100, 2.4f, 0.05f });
        }
    }
}
