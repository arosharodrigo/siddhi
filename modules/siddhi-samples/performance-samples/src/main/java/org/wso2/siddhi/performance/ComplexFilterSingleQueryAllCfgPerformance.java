package org.wso2.siddhi.performance;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import javax.swing.text.StyledEditorKit.BoldAction;

import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.stream.StreamJunction;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.stream.output.StreamCallback;

public class ComplexFilterSingleQueryAllCfgPerformance {
    private static int count = 0;
    private static int eventCount = 0;
    private static int prevEventCount = 0;
    private static volatile long start = System.currentTimeMillis();

    private static void Execution(long eventGenCount, boolean gpuEnabled, int defaultBufferSize, int threadPoolSize, int blockSize) 
	    throws InterruptedException {
    	
    	count = 0;
    	eventCount = 0;
    	prevEventCount = 0;
    	start = System.currentTimeMillis();
    	
    	final List<Double> throughputList = new ArrayList<Double>();
    	
	System.out.println("ComplexFilterSingleQueryAllCfgPerformance [GPUEnabled=" + gpuEnabled
			+ "|RingBufferSize=" + defaultBufferSize
			+ "|ThreadPoolSize=" + threadPoolSize
			+ "|EventBlockSize=" + blockSize + "]");
    	
        final SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setDefaultEventBufferSize(defaultBufferSize);
        siddhiManager.getSiddhiContext().setExecutorService(Executors.newFixedThreadPool(threadPoolSize));

        String cseEventStream = "@config(async = 'true') define stream cseEventStream (symbol string, price float, volume int, change float, pctchange float);";
        StringBuilder sb = new StringBuilder();
        sb.append("@info(name = 'query1') ");
        if(gpuEnabled)
        {
            sb.append("@gpu(filter='true', block.size='").append(blockSize).append("', string.sizes='8')");
        }
        sb.append("from cseEventStream[pctchange > 0.1 and change < 2.5 and volume > 100 and price < 70] select symbol,price,volume,change insert into outputStream ;");
        
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
                    throughputList.add(tp);
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
        long currentGenEventCount = 0;
        while (currentGenEventCount < eventGenCount) {
            inputHandler.send(new Object[]{"WSO2", 55.6f, 1000, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100, 2.4f, 0.05f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 500, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 50, 2.5f, 0.2f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 200, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 55.6f, 1000, 2.4f, 0.05f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 300, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100, 2.4f, 0.05f });
            
            currentGenEventCount += 8;
        }
        
        double totalThroughput = 0;
	for (Double tp : throughputList) {
	    totalThroughput += tp;
	}
        
        double avgThroughput = totalThroughput / throughputList.size();
        System.out.println("ComplexFilterSingleQueryAllCfgPerformance [GPUEnabled=" + gpuEnabled + 
		" TotalEventCount=" + eventGenCount +
        	" DefaultEventBufferSize=" + defaultBufferSize +
        	" ThreadPoolSize=" + threadPoolSize + 
        	" EventBlockSize=" + blockSize + 
        	"] AvgThroughput = " + avgThroughput + " Event/sec");
    }
    
    public static void main(String[] args) throws InterruptedException {
    	
	final int[] defaultBufferSizes = { 1024, 2048, 4096, 8192, 16384 };
	final int[] threadPoolSizes = { 2, 4, 8, 16 };
	final int[] blockSizes = { 64, 128, 256, 512, 1024 };

	for (int b : defaultBufferSizes) {
	    for (int t : threadPoolSizes) {
		for (int l : blockSizes) {
		    if(b > l*2 && b / l <= 128) // limit GPU thread blocks
		    {
			long eventCount = 5000000l * b; 
			Execution(eventCount, false, b, t, l);
			Execution(eventCount, true, b, t, l);
		    }
		}
	    }
	}
    }
}
