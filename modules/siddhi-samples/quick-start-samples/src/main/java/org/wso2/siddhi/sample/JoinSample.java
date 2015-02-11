package org.wso2.siddhi.sample;

import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.stream.output.StreamCallback;
import org.wso2.siddhi.core.util.EventPrinter;

public class JoinSample {
    public static void main(String[] args) throws InterruptedException {

        SiddhiManager siddhiManager = new SiddhiManager();

        String streams = "@plan:name('planname') " + //@plan:parallel " +
//        String streams = "" +
                "define stream cseEventStream (symbol string, price float, volume int); " +
                "define stream twitterStream (user string, tweet string, company string); ";
        String query = "" +
                "@info(name = 'query1') " +
                "from cseEventStream#window.time(1 sec) join twitterStream#window.time(1 sec) " +
                "on cseEventStream.symbol== twitterStream.company " +
                "select cseEventStream.symbol as symbol, twitterStream.tweet, cseEventStream.price, cseEventStream.volume " +
                "insert into outputStream ;";

        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(streams + query);

//        executionPlanRuntime.addCallback("cseEventStream", new StreamCallback() {
//            @Override
//            public void receive(Event[] events) {
//                EventPrinter.print(events);
//                
//            }
//
//        });
        
//        executionPlanRuntime.addCallback("twitterStream", new StreamCallback() {
//            @Override
//            public void receive(Event[] events) {
//                EventPrinter.print(events);
//                
//            }
//
//        });
        
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
        for(int i=0;i<100000; ++i) {
            cseEventStreamHandler.send(new Object[]{"WSO2", 55.6f, i});
            twitterStreamHandler.send(new Object[]{"User1", "Hello World"+i, "WSO2"});
            cseEventStreamHandler.send(new Object[]{"IBM", 75.6f, i});
            cseEventStreamHandler.send(new Object[]{"CSCO", 100.2f, i});
            twitterStreamHandler.send(new Object[]{"User2", "Dummy"+i, "LLOY"});
            twitterStreamHandler.send(new Object[]{"User3", "Dummy"+i, "CL"});
            cseEventStreamHandler.send(new Object[]{"GOOG", 55.0f, i});
            cseEventStreamHandler.send(new Object[]{"BARC", 25.1f, i});
            twitterStreamHandler.send(new Object[]{"User4", "Dummy"+i, "RB"});
            twitterStreamHandler.send(new Object[]{"User5", "Dummy"+i, "NG"});
            Thread.sleep(500);
        }
        Thread.sleep(100);
        executionPlanRuntime.shutdown();
    }
}
