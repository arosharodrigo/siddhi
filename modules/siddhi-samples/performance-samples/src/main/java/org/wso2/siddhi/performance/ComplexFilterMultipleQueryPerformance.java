package org.wso2.siddhi.performance;

import java.util.ArrayList;
import java.util.List;
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

public class ComplexFilterMultipleQueryPerformance {
    private static int count = 0;
    private static int eventCount = 0;
    private static int prevEventCount = 0;
    private static volatile long start = System.currentTimeMillis();

    private static Options cliOptions;

    private static class TestQuery {
        public String query;
        public int cudaDeviceId;
        
        public TestQuery(String query, int cudaDeviceId) {
            this.query = query;
            this.cudaDeviceId = cudaDeviceId;
        }
        
    }
    
    private static TestQuery [] queries = {
        new TestQuery("from cseEventStream[pctchange > 0.1 and change < 2.5 and volume > 100 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;", 0),
        new TestQuery("from cseEventStream[pctchange > 0.1 and change > 2.5 and volume > 100 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;", 1),
        new TestQuery("from cseEventStream[pctchange > 0.1 and change > 2.5 and volume > 300 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;", 0),
        new TestQuery("from cseEventStream[pctchange > 0.1 and change > 2.5 and volume > 300 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;", 1),
        new TestQuery("from cseEventStream[pctchange > 0.1 and change < 2.5 and volume > 100 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;", 0),
        new TestQuery("from cseEventStream[pctchange > 0.1 and change > 2.5 and volume > 100 and price < 70] select symbol,price,volume,change,pctchange insert into outputStream ;", 1)
    };
    
    private static void Help() {
        // This prints out some help
        HelpFormatter formater = new HelpFormatter();

        formater.printHelp("ComplexFilterMultipleQueryPerformance", cliOptions);
        System.exit(0);
    }

    public static void main(String[] args) throws InterruptedException {

        cliOptions = new Options();
        cliOptions.addOption("a", "enable-async", true, "Enable Async processing");
        cliOptions.addOption("g", "enable-gpu", true, "Enable GPU processing");
        cliOptions.addOption("e", "event-count", true, "Total number of events to be generated");
        cliOptions.addOption("q", "query-count", true, "Number of Siddhi Queries to be generated");
        cliOptions.addOption("r", "ringbuffer-size", true, "Disruptor RingBuffer size - in power of two");
        cliOptions.addOption("t", "threadpool-size", true, "Executor service pool size");
        cliOptions.addOption("b", "events-per-tblock", true, "Number of Events per thread block in GPU");

        CommandLineParser cliParser = new BasicParser();
        CommandLine cmd = null;

        boolean asyncEnabled = true;
        boolean gpuEnabled = false;
        long totalEventCount = 50000000l;
        int queryCount = 1;
        int defaultBufferSize = 1024;
        int threadPoolSize = 4;
        int eventBlockSize = 256;

        final List<Double> throughputList = new ArrayList<Double>();

        try {
            cmd = cliParser.parse(cliOptions, args);

            if (cmd.hasOption("a")) {
                asyncEnabled = Boolean.parseBoolean(cmd.getOptionValue("a"));
            }

            if (cmd.hasOption("g")) {
                gpuEnabled = Boolean.parseBoolean(cmd.getOptionValue("g"));
            }

            if (cmd.hasOption("e")) {
                totalEventCount = Long.parseLong(cmd.getOptionValue("e"));
            }
            
            if (cmd.hasOption("q")) {
                queryCount = Integer.parseInt(cmd.getOptionValue("q"));
            }

            if (cmd.hasOption("r")) {
                defaultBufferSize = Integer.parseInt(cmd.getOptionValue("r"));
            }

            if (cmd.hasOption("t")) {
                threadPoolSize = Integer.parseInt(cmd.getOptionValue("t"));
            }

            if (cmd.hasOption("b")) {
                eventBlockSize = Integer.parseInt(cmd.getOptionValue("b"));
            }

        } catch (ParseException e) {
            e.printStackTrace();
            Help();
        }

        System.out.println("Siddhi.Config [EnableAsync=" + asyncEnabled +
                "|GPUEnabled=" + gpuEnabled + 
                "|EventCount=" + totalEventCount +
                "|QueryCount=" + queryCount +
                "|RingBufferSize=" + defaultBufferSize + 
                "|ThreadPoolSize=" + threadPoolSize + 
                "|EventBlockSize=" + eventBlockSize + "]");

        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setDefaultEventBufferSize(defaultBufferSize);
        siddhiManager.getSiddhiContext().setExecutorService(Executors.newFixedThreadPool(threadPoolSize));

        String cseEventStream = "@config(async = '" + (asyncEnabled ? "true" : "false" ) + "') define stream cseEventStream (symbol string, price float, volume int, change float, pctchange float);";

        System.out.println("Stream def    = [ " + cseEventStream + " ]");

        StringBuffer execString = new StringBuffer();
        execString.append(cseEventStream);
        
        for(int i=0; i<queryCount; ++i) {
            StringBuilder sb = new StringBuilder();
            
            sb.append("@info(name = 'query" + (i + 1) + "') ");
            if(gpuEnabled)
            {
                sb.append("@gpu(filter='true', cuda.device=").append(queries[i].cudaDeviceId)
                    .append("block.size='").append(eventBlockSize).append("', string.sizes='8')");
            }
            sb.append(queries[i].query);

            String query = sb.toString();
            System.out.println("Filter query" + (i+1) + " = [ " + query + " ]");
            
            execString.append(query);
        }
        
        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(execString.toString());

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
        while (currentGenEventCount < totalEventCount) {
            inputHandler.send(new Object[]{"WSO2", 55.6f, 1000, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100, 2.5f, 0.05f });
            inputHandler.send(new Object[]{"GOOG", 55.6f, 500, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"WSO2", 75.6f, 50, 2.5f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 55.6f, 200, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"GOOG", 55.6f, 1000, 2.5f, 0.05f });
            inputHandler.send(new Object[]{"WSO2", 55.6f, 300, 2.4f, 0.2f });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100, 2.5f, 0.05f });

            currentGenEventCount += 8;
        }

        double totalThroughput = 0;
        for (Double tp : throughputList) {
            totalThroughput += tp;
        }

        double avgThroughput = totalThroughput / throughputList.size();
        System.out.println("ComplexFilterMultipleQueryPerformance [EnableAsync=" + asyncEnabled + 
                " GPUEnabled=" + gpuEnabled + 
                " TotalEventCount=" + totalEventCount +
                " QueryCount=" + queryCount +
                " DefaultEventBufferSize=" + defaultBufferSize +
                " ThreadPoolSize=" + threadPoolSize + 
                " EventBlockSize=" + eventBlockSize + 
                "] AvgThroughput = " + avgThroughput + " Event/sec");


        System.exit(0);
    }
}
