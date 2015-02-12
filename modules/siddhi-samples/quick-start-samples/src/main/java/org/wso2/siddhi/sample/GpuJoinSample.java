package org.wso2.siddhi.sample;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.stream.output.StreamCallback;
import org.wso2.siddhi.core.util.EventPrinter;

public class GpuJoinSample {
private static final Logger log = Logger.getLogger(GpuFilterSample.class);
    
    public static void main(String[] args) throws InterruptedException {

        // Create Siddhi Manager
        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setEventBufferSize(4096);

        String executionPlan = "@plan:name('GpuJoinSample') @plan:parallel "
                + "define stream cseEventStream (symbol string, price float, volume int); "
                + "define stream twitterStream (company string, numoccur int); "
                + "@info(name = 'query1') @gpu(block.size='128', cuda.device='1', string.sizes='symbol=8,company=8') "
                + " from cseEventStream#window.length(1000) join twitterStream#window.length(1000) " 
                + " on cseEventStream.symbol== twitterStream.company " 
                + " select cseEventStream.symbol as symbol, twitterStream.numoccur, cseEventStream.price, cseEventStream.volume " 
                + " insert into outputStream ;";

        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(executionPlan);

        executionPlanRuntime.addCallback("cseEventStream", new StreamCallback() {
            @Override
            public void receive(Event[] events) {
                EventPrinter.print(events);
                
            }

        });
        
        executionPlanRuntime.addCallback("twitterStream", new StreamCallback() {
            @Override
            public void receive(Event[] events) {
                EventPrinter.print(events);
                
            }

        });
        
        executionPlanRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long timeStamp, Event[] inEvents, Event[] removeEvents) {
                System.out.print("OutEvents : ");
                EventPrinter.print(timeStamp, inEvents, removeEvents);
            }

        });
        
        InputHandler cseEventStreamHandler = executionPlanRuntime.getInputHandler("cseEventStream");
        InputHandler twitterStreamHandler = executionPlanRuntime.getInputHandler("twitterStream");
        executionPlanRuntime.start();
        
        System.out.println("JoinSample");
        for(int i=0;i<100; ++i) {
            cseEventStreamHandler.send(new Object[]{"WSO2", 55.6f, i});
            twitterStreamHandler.send(new Object[]{"WSO2", i});
            cseEventStreamHandler.send(new Object[]{"IBM", 75.6f, i});
            cseEventStreamHandler.send(new Object[]{"CSCO", 100.2f, i});
            twitterStreamHandler.send(new Object[]{"LLOY", i});
            twitterStreamHandler.send(new Object[]{"CL", i});
            cseEventStreamHandler.send(new Object[]{"GOOG", 55.0f, i});
            cseEventStreamHandler.send(new Object[]{"BARC", 25.1f, i});
            twitterStreamHandler.send(new Object[]{"RB", i});
            twitterStreamHandler.send(new Object[]{"NG", i});
            Thread.sleep(500);
        }

        executionPlanRuntime.shutdown();
    }
}
