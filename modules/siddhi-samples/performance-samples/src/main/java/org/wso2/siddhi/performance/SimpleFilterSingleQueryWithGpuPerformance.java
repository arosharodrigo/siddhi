package org.wso2.siddhi.performance;

import java.util.concurrent.Executors;

import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.stream.StreamJunction;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.stream.output.StreamCallback;

public class SimpleFilterSingleQueryWithGpuPerformance
{
	private static int count = 0;
    private static int eventCount = 0;
    private static int prevEventCount = 0;
    private static volatile long start = System.currentTimeMillis();

    public static void main(String[] args) throws InterruptedException {
    	
    	int defaultBufferSize = 1204;
    	int threadPoolSize = 4;
    	int eventBlockSize = 256;
    	
    	if(args.length == 1)
    	{
    		defaultBufferSize = Integer.parseInt(args[0]);
    	}
    	else if(args.length == 2)
    	{
    		defaultBufferSize = Integer.parseInt(args[0]);
    		threadPoolSize = Integer.parseInt(args[1]);
    	}
    	else if(args.length == 3)
    	{
    		defaultBufferSize = Integer.parseInt(args[0]);
    		threadPoolSize = Integer.parseInt(args[1]);
    		eventBlockSize = Integer.parseInt(args[2]);
    	}
    	
    	System.out.println("Siddhi.Config = DefaultEventBufferSize: " + defaultBufferSize + 
    			" ThreadPoolSize: " + threadPoolSize + " EventBlockSize: " + eventBlockSize);
    	
        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setDefaultEventBufferSize(defaultBufferSize);
        siddhiManager.getSiddhiContext().setExecutorService(Executors.newFixedThreadPool(threadPoolSize));

        String cseEventStream = "@config(async = 'true') define stream cseEventStream (symbol string, price float, volume int);";
        String query1 = "@info(name = 'query1') @gpu(filter='true', block.size='" + eventBlockSize + "', string.sizes='8') "
        		+ " from cseEventStream[70 > price] select symbol,price,volume insert into outputStream ;";

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
            inputHandler.send(new Object[]{"WSO2", 55.6f, 100});
            inputHandler.send(new Object[]{"IBM", 75.6f, 100});
            inputHandler.send(new Object[]{"WSO2", 55.6f, 100});
            inputHandler.send(new Object[]{"IBM", 75.6f, 100});
            inputHandler.send(new Object[]{"WSO2", 55.6f, 100});
            inputHandler.send(new Object[]{"IBM", 75.6f, 100});
            inputHandler.send(new Object[]{"WSO2", 55.6f, 100});
            inputHandler.send(new Object[]{"IBM", 75.6f, 100});
        }
    }
}
