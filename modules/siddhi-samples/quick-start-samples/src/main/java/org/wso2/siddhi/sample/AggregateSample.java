package org.wso2.siddhi.sample;

import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.util.EventPrinter;

public class AggregateSample {
    public static void main(String[] args) throws InterruptedException {

        // Create Siddhi Manager
        SiddhiManager siddhiManager = new SiddhiManager();

        String executionPlan = "@plan:name('planname') @plan:parallel define stream cseEventStream (symbol string, price1 float, price2 float, volume long , quantity int);"
                + "@info(name = 'query1') from cseEventStream[price1 > 60]#window.length(5) select symbol, price1, avg(price2) as avgPrice, quantity insert into outputStream;";

        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(executionPlan);

        executionPlanRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long timeStamp, Event[] inEvents, Event[] removeEvents) {
                EventPrinter.print(timeStamp, inEvents, removeEvents);
            }
        });

        System.out.println("AggregateSample");
        InputHandler inputHandler = executionPlanRuntime.getInputHandler("cseEventStream");
        for(int i=0;i<100; ++i) {
            inputHandler.send(new Object[]{"WSO2", 50f, 60f, 60l, i});
            inputHandler.send(new Object[]{"WSO2", 70f, i*50f, 40l, i});
            inputHandler.send(new Object[]{"WSO2", 70f, 44f, 200l, i});
        }
        Thread.sleep(100);

    }
}
