/*
 * Copyright (c) 2005 - 2014, WSO2 Inc. (http://www.wso2.org)
 * All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.wso2.siddhi.sample;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.ExecutionPlanRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.util.EventPrinter;

public class SimpleFilterSample {
    private static final Logger log = Logger.getLogger(SimpleFilterSample.class);

    public static void main(String[] args) throws InterruptedException {

        // Create Siddhi Manager
        SiddhiManager siddhiManager = new SiddhiManager();
        siddhiManager.getSiddhiContext().setEventBufferSize(4096);

        String executionPlan = "@plan:name('FilterSample') @plan:parallel define stream cseEventStream (symbol string, price float, volume long);"
                + "@info(name = 'query1') from cseEventStream[volume < 150] select symbol,price insert into outputStream ;";

        ExecutionPlanRuntime executionPlanRuntime = siddhiManager.createExecutionPlanRuntime(executionPlan);

        executionPlanRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long timeStamp, Event[] inEvents, Event[] removeEvents) {
              //EventPrinter.print(timeStamp, inEvents, removeEvents);
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
            if(count == 1000000000) {
                break;
            }
        }

        executionPlanRuntime.shutdown();

    }
}
