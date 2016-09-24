package org.wso2.siddhi.sample.arosha;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.util.EventPrinter;
import org.wso2.siddhi.sample.SimpleFilterSample;

/**
 * Created by arosha on 7/29/16.
 */
public class MyFilterTest {

    private static final Logger log = Logger.getLogger(SimpleFilterSample.class);

    public static void main(String[] args) throws InterruptedException {
        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setEventBufferSize(4096);

        String executionPlan =
                "@plan:name('FilterSample') " +
                        "@plan:parallel " +
                        "define stream cseEventStream (symbol string, price float, volume long);" +
                        "@info(name = 'query1') from cseEventStream[volume < 150]#window.length(2) " +
                        "select symbol,price insert into outputStream ;";

        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(executionPlan);

        executionPlanRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long timeStamp, Event[] inEvents, Event[] removeEvents) {
                EventPrinter.print(timeStamp, inEvents, removeEvents);
                log.debug("timestamp=" + timeStamp +
                        " inEvents=" + (inEvents != null ? inEvents.length : 0) +
                        " removeEvents=" + (removeEvents != null ? removeEvents.length : 0));
            }
        });

        InputHandler inputHandler = executionPlanRuntime.getInputHandler("cseEventStream");
        executionPlanRuntime.start();

        long count = 0;
        while(true) {
            inputHandler.send(new Object[]{"IBM", 700f, 100l});
            inputHandler.send(new Object[]{"WSO2", 60.5f, 200l});
            inputHandler.send(new Object[]{"GOOG", 50f, 30l});
            inputHandler.send(new Object[]{"IBM", 76.6f, 400l});
            inputHandler.send(new Object[]{"WSO2", 45.6f, 50l});

            count += 5;
            if(count > 10) {
                break;
            }
        }


        executionPlanRuntime.shutdown();

    }

}
