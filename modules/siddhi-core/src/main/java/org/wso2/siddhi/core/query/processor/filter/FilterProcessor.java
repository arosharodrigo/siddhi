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
package org.wso2.siddhi.core.query.processor.filter;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventIterator;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.query.api.definition.Attribute;


public class FilterProcessor implements Processor {

	private static final Logger log = Logger.getLogger(FilterProcessor.class);
    protected Processor next;
    private ExpressionExecutor conditionExecutor;

    public FilterProcessor(ExpressionExecutor conditionExecutor) {
        if(Attribute.Type.BOOL.equals(conditionExecutor.getReturnType())) {
            this.conditionExecutor = conditionExecutor;
        }else{
            throw new OperationNotSupportedException("Return type of "+conditionExecutor.toString()+" should be of type BOOL. " +
                    "Actual type: "+conditionExecutor.getReturnType().toString());
        }
    }

    public FilterProcessor cloneProcessor(){
        return new FilterProcessor(conditionExecutor.cloneExecutor());
    }

    @Override
    public void process(StreamEvent event) {
        StreamEventIterator iterator = event.getIterator();
        int count = 0;
        while (iterator.hasNext()){
            StreamEvent streamEvent = iterator.next();
            count++;
            if (!(Boolean) conditionExecutor.execute(streamEvent)){
                iterator.remove();
            }
        }
        //log.info("Batch count : " + count);
        if(iterator.getFirstElement() != null){
            this.next.process(iterator.getFirstElement());
        }
    }

    @Override
    public Processor getNextProcessor() {
        return next;
    }

    @Override
    public void setNextProcessor(Processor processor) {
        next = processor;
    }

    @Override
    public void setToLast(Processor processor) {
        if (next == null) {
            this.next = processor;
        } else {
            this.next.setNextProcessor(processor);
        }
    }


}
