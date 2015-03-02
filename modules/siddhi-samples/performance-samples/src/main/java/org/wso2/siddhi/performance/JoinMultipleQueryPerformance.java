package org.wso2.siddhi.performance;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.stream.output.StreamCallback;
import org.wso2.siddhi.core.util.EventPrinter;

public class JoinMultipleQueryPerformance {

    private static Options cliOptions;
    private static OutputPerfromanceCalculator twitterPerformanceCalculator = null;
    private static OutputPerfromanceCalculator tradePerformanceCalculator = null;
    
    private static class OutputPerfromanceCalculator {
        String name;
        int count = 0;
        int eventCount = 0;
        int prevEventCount = 0;
        volatile long start = System.currentTimeMillis();
        final List<Double> throughputList = new ArrayList<Double>();
        final DecimalFormat decimalFormat = new DecimalFormat("###.##");
        
        public OutputPerfromanceCalculator(String name) {
            this.name = "<" + name + ">";
        }
        
        public void calculate(int currentEventCount) {
            eventCount += currentEventCount;
            count++;
            if (count % 1000000 == 0) {
                long end = System.currentTimeMillis();
                double tp = ((eventCount - prevEventCount) * 1000.0) / (end - start);
                throughputList.add(tp);
                System.out.println(name + " Throughput = " + decimalFormat.format(tp) + " Event/sec " + (eventCount - prevEventCount));
                start = end;
                prevEventCount = eventCount;
            }
        }
        
        public double getAverageThroughput() {
            double totalThroughput = 0;
            
            for (Double tp : throughputList) {
                totalThroughput += tp;
            }
            
            double avgThroughput = totalThroughput / throughputList.size();
            
            return avgThroughput;
        }
        
        public void printAverageThroughput() {
            double totalThroughput = 0;
            
            for (Double tp : throughputList) {
                totalThroughput += tp;
            }
            
            double avgThroughput = totalThroughput / throughputList.size();
            
            System.out.println(name + " AvgThroughput = " + avgThroughput + " Event/sec");
        }
    }
    
    public interface EventSender 
    {
        public int sendEvents(long iteration) throws InterruptedException;
    } 
    
    private static class EventSenderRunner implements Runnable {

        private EventSender eventSender;
        private long numberOfEvents;

        public EventSenderRunner(EventSender eventSender, long numberOfEvents) {
            super();
            this.eventSender = eventSender;
            this.numberOfEvents = numberOfEvents;
        }
        
        @Override
        public void run() {
            try {
                
                long currentGenEventCount = 0;
                while (currentGenEventCount < numberOfEvents) {
                    currentGenEventCount += eventSender.sendEvents(currentGenEventCount);
                }
                
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        
    }
    
    private static class StockEventSender implements EventSender {

        private InputHandler inputHandler;
        
        public StockEventSender(InputHandler inputHandler) {
            super();
            this.inputHandler = inputHandler;
        }

        @Override
        public int sendEvents(long iteration) throws InterruptedException {
            inputHandler.send(new Object[]{"WSO2", 55.6f, 1000 });
            inputHandler.send(new Object[]{"IBM", 75.6f, 100});
            inputHandler.send(new Object[]{"GOOG", 55.6f, 500});
            inputHandler.send(new Object[]{"AAPL", 75.6f, 50});
            inputHandler.send(new Object[]{"INTC", 75.6f, 100});
            return 5;
        }
        
    }
    
    private static class TwitterEventSender implements EventSender {

        private InputHandler inputHandler;
        
        public TwitterEventSender(InputHandler inputHandler) {
            super();
            this.inputHandler = inputHandler;
        }

        @Override
        public int sendEvents(long iteration) throws InterruptedException {
            inputHandler.send(new Object[]{"WSO2", iteration});
            inputHandler.send(new Object[]{"IBM", iteration});
            inputHandler.send(new Object[]{"VOD", iteration});
            inputHandler.send(new Object[]{"AAPL", iteration});
            inputHandler.send(new Object[]{"GOOG", iteration});
            return 5;
        }
    }
    
    private static class TradeEventSender implements EventSender {

        private InputHandler inputHandler;
        
        public TradeEventSender(InputHandler inputHandler) {
            super();
            this.inputHandler = inputHandler;
        }

        @Override
        public int sendEvents(long iteration) throws InterruptedException {
            inputHandler.send(new Object[]{"QQQQ", 20.5f, 100});
            inputHandler.send(new Object[]{"BARC", 200.0f, 1000});
            inputHandler.send(new Object[]{"GOOG", 200.5f, 1000});
            inputHandler.send(new Object[]{"WSO2", 100.5f, 100});
            inputHandler.send(new Object[]{"AAPL", 150.5f, 100});
            return 5;
        }
    }
    
    private static class TestQuery {
        public String query;
        public int cudaDeviceId;
        public TestQuery(String query, int cudaDeviceId) {
            this.query = query;
            this.cudaDeviceId = cudaDeviceId;
        }
    }
    
    private static TestQuery [] queries = {
        new TestQuery("from cseStockStream#window.length(1000) as a join twitterStream#window.length(200) as b " 
                + " on a.symbol==b.company " 
                + " select a.symbol as symbol, b.numoccur, a.bidPrice, a.qty " 
                + " insert into twitterStockStream ; ", 1),
                
        new TestQuery("from cseStockStream#window.length(100) as a join cseTradeStream#window.length(1000) as b " 
                + " on a.bidPrice <= b.tradePrice and a.qty <= b.volume " 
                + " select a.symbol as symbol, a.bidPrice, b.tradePrice " 
                + " insert into stockTradeStream ; ", 1),
                
        new TestQuery("from cseStockStream#window.length(1000) as a join twitterStream#window.length(200) as b " 
                + " on a.symbol==b.company " 
                + " select a.symbol as symbol, b.numoccur, a.bidPrice, a.qty " 
                + " insert into twitterStockStream ; ", 1),
                
        new TestQuery("from cseStockStream#window.length(100) as a join cseTradeStream#window.length(1000) as b " 
                + " on a.bidPrice <= b.tradePrice and a.qty <= b.volume " 
                + " select a.symbol as symbol, a.bidPrice, b.tradePrice " 
                + " insert into stockTradeStream ; ", 1)
    };
    
    private static void Help() {
        // This prints out some help
        HelpFormatter formater = new HelpFormatter();
        formater.printHelp("JoinMultipleQueryPerformance", cliOptions);
        System.exit(0);
    }
    
    public static void main(String[] args) throws InterruptedException {
        
        cliOptions = new Options();
        cliOptions.addOption("a", "enable-async", true, "Enable Async processing");
        cliOptions.addOption("g", "enable-gpu", true, "Enable GPU processing");
        cliOptions.addOption("e", "event-count", true, "Total number of events to be generated");
        cliOptions.addOption("q", "query-count", true, "Number of Siddhi Queries to be generated");
        cliOptions.addOption("r", "ringbuffer-size", true, "Disruptor RingBuffer size - in power of two");
        cliOptions.addOption("Z", "batch-max-size", true, "GPU Event batch max size");
        cliOptions.addOption("z", "batch-min-size", true, "GPU Event batch min size");
        cliOptions.addOption("t", "threadpool-size", true, "Executor service pool size");
        cliOptions.addOption("b", "events-per-tblock", true, "Number of Events per thread block in GPU");
        cliOptions.addOption("s", "strict-batch-scheduling", true, "Strict batch size policy");
        
        CommandLineParser cliParser = new BasicParser();
        CommandLine cmd = null;
        final DecimalFormat decimalFormat = new DecimalFormat("###.##");
        
        twitterPerformanceCalculator = new OutputPerfromanceCalculator("twitterStockStream");
        tradePerformanceCalculator = new OutputPerfromanceCalculator("stockTradeStream");        
        
        boolean asyncEnabled = true;
        boolean gpuEnabled = false;
        long totalEventCount = 50000000l;
        int queryCount = 1;
        int defaultBufferSize = 1024;
        int threadPoolSize = 4;
        int eventBlockSize = 256;
        boolean softBatchScheduling = true;
        int maxEventBatchSize = 1024;
        int minEventBatchSize = 32;
        
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
            if (cmd.hasOption("Z")) {
                maxEventBatchSize = Integer.parseInt(cmd.getOptionValue("Z"));
            }
            if (cmd.hasOption("z")) {
                minEventBatchSize = Integer.parseInt(cmd.getOptionValue("z"));
            }
            if (cmd.hasOption("s")) {
                softBatchScheduling = !Boolean.parseBoolean(cmd.getOptionValue("s"));
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
                "|EventBlockSize=" + eventBlockSize + 
                "|EventBatchMaxSize=" + maxEventBatchSize +
                "|EventBatchMinSize=" + minEventBatchSize +
                "|SoftBatchScheduling=" + softBatchScheduling + 
                "]");
        
        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setEventBufferSize(defaultBufferSize); //.setDefaultEventBufferSize(defaultBufferSize);
        siddhiManager.getSiddhiContext().setExecutorService(new ThreadPoolExecutor(threadPoolSize, Integer.MAX_VALUE,
                60L, TimeUnit.SECONDS,
                new LinkedBlockingDeque<Runnable>()));
//                Executors.new newFixedThreadPool(threadPoolSize);
        siddhiManager.getSiddhiContext().setScheduledExecutorService(Executors.newScheduledThreadPool(threadPoolSize));
        
        String cseEventStream = "@plan:name('JoinMultipleQuery') " + (asyncEnabled ? "@plan:parallel" : "" ) + " "
                + "define stream cseStockStream (symbol string, bidPrice float, qty int); "
                + "define stream twitterStream (company string, numoccur long); "
                + "define stream cseTradeStream (symbol string, tradePrice float, volume int); ";
        
        System.out.println("Stream def = [ " + cseEventStream + " ]");
        StringBuffer execString = new StringBuffer();
        execString.append(cseEventStream);
        
        for(int i=0; i<queryCount; ++i) {
            StringBuilder sb = new StringBuilder();
            sb.append("@info(name = 'query" + (i + 1) + "') ");
            if(gpuEnabled)
            {
                sb.append("@gpu(")
                .append("cuda.device='").append(queries[i].cudaDeviceId).append("', ")
                .append("batch.max.size='").append(maxEventBatchSize).append("', ")
                .append("batch.min.size='").append(minEventBatchSize).append("', ")
                .append("block.size='").append(eventBlockSize).append("', ")
                .append("batch.schedule='").append(softBatchScheduling ? "soft" : "hard").append("', ")
                .append("string.sizes='symbol=8', ")
                .append("work.size='100' ")
                .append(") ")
                .append("@performance(batch.count='1000') ");
            }
            sb.append(queries[i].query);
            String query = sb.toString();
            System.out.println("Filter query" + (i+1) + " = [ " + query + " ]");
            execString.append(query);
        }
        
        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(execString.toString());
        
        executionPlanRuntime.addCallback("twitterStockStream", new StreamCallback() {
            @Override
            public void receive(Event[] inEvents) {
                twitterPerformanceCalculator.calculate(inEvents.length);
//                EventPrinter.print(inEvents);
            }
        });
        
        if(queryCount > 1)
        {
            executionPlanRuntime.addCallback("stockTradeStream", new StreamCallback() {
                @Override
                public void receive(Event[] inEvents) {
                    tradePerformanceCalculator.calculate(inEvents.length);
//                    EventPrinter.print(inEvents);
                }
            });
        }
                
        InputHandler inputHandlerStock = executionPlanRuntime.getInputHandler("cseStockStream");
        InputHandler inputHandlerTwitter = executionPlanRuntime.getInputHandler("twitterStream");
        InputHandler inputHandlerTrade = executionPlanRuntime.getInputHandler("cseTradeStream");
        executionPlanRuntime.start();
        
        Thread stockThread = new Thread(new EventSenderRunner(new StockEventSender(inputHandlerStock), totalEventCount/3));
        Thread twitterThread = new Thread(new EventSenderRunner(new TwitterEventSender(inputHandlerTwitter), totalEventCount/3));
        Thread tradeThread = new Thread(new EventSenderRunner(new TradeEventSender(inputHandlerTrade), totalEventCount/3));

        stockThread.start();
        twitterThread.start();
        if(queryCount > 1) {
            tradeThread.start();
        }

        stockThread.join();
        twitterThread.join();
        if(queryCount > 1) {
            tradeThread.join();
        }
              
        System.out.println("JoinMultipleQueryPerformance [EnableAsync=" + asyncEnabled +
                " GPUEnabled=" + gpuEnabled +
                " TotalEventCount=" + totalEventCount +
                " QueryCount=" + queryCount +
                " DefaultRingBufferSize=" + defaultBufferSize +
                " ThreadPoolSize=" + threadPoolSize +
                " EventBlockSize=" + eventBlockSize +
                " EventBatchMaxSize=" + maxEventBatchSize +
                " EventBatchMinSize=" + minEventBatchSize +
                " SoftBatchScheduling=" + softBatchScheduling + 
                "]");
        
        twitterPerformanceCalculator.printAverageThroughput();
        tradePerformanceCalculator.printAverageThroughput();
        
        List<SummaryStatistics> statList  = new ArrayList<SummaryStatistics>();
        executionPlanRuntime.getStatistics(statList);
        
        executionPlanRuntime.shutdown();
        
        StatisticalSummaryValues totalStatistics = AggregateSummaryStatistics.aggregate(statList);
        
        System.out.println(new StringBuilder()
        .append("EventProcessTroughputEPS ")
        .append("DatasetCount=").append(statList.size())
        .append("|length=").append(totalStatistics.getN())
        .append("|Avg=").append(decimalFormat.format(totalStatistics.getMean()))
        .append("|Min=").append(decimalFormat.format(totalStatistics.getMin()))
        .append("|Max=").append(decimalFormat.format(totalStatistics.getMax()))
        .append("|Var=").append(decimalFormat.format(totalStatistics.getVariance()))
        .append("|StdDev=").append(decimalFormat.format(totalStatistics.getStandardDeviation())).toString());
        
        System.exit(0);
    }
}
