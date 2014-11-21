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

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventIterator;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.CudaEvent;
import org.wso2.siddhi.query.api.definition.Attribute;


public class FilterProcessor implements Processor {

	private static final Logger log = Logger.getLogger(FilterProcessor.class);
    protected Processor next;
    private ExpressionExecutor conditionExecutor;
    private SiddhiGpu.GpuEventConsumer gpuEventConsumer = null;
    private int gpuProcessMinimumEventCount = 256;
    private List<SiddhiGpu.CudaEvent> cudaEventList = null;
    private StreamEvent [] inputStreamEvents = null;
    private int inputStreamEventIndex = 0;

    public FilterProcessor(ExpressionExecutor conditionExecutor) {
        if(Attribute.Type.BOOL.equals(conditionExecutor.getReturnType())) {
            this.conditionExecutor = conditionExecutor;
        }else{
            throw new OperationNotSupportedException("Return type of "+conditionExecutor.toString()+" should be of type BOOL. " +
                    "Actual type: "+conditionExecutor.getReturnType().toString());
        }
    }
    
    public FilterProcessor(ExpressionExecutor conditionExecutor, SiddhiGpu.GpuEventConsumer gpuEventConsumer, int threshold) {
    	this.gpuEventConsumer = gpuEventConsumer;
    	this.gpuProcessMinimumEventCount = threshold;
    	
    	this.cudaEventList = new ArrayList<SiddhiGpu.CudaEvent>(gpuEventConsumer.GetMaxBufferSize());
    	this.inputStreamEvents = new StreamEvent[gpuEventConsumer.GetMaxBufferSize()];
    	
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
    	
    	if(gpuEventConsumer == null)
    	{
    		StreamEventIterator iterator = event.getIterator();
    		while (iterator.hasNext()){
    			StreamEvent streamEvent = iterator.next();
    			if (!(Boolean) conditionExecutor.execute(streamEvent)){
    				iterator.remove();
    			}
    		}
    		
    		if(iterator.getFirstElement() != null){
    			this.next.process(iterator.getFirstElement());
    		}
    	}
    	else
    	{

    		// ############################################################################################################
    		//TODO: check batch size and use GPU processing if size exceed minimum threshold 
    		// number of events in batch should at least exceed block size

    		// process all events with GPU
    		// remove non matching events OR add matching events to a new StreamEvent
    		
    		int eventCount = 0;
    		StreamEventIterator iterator = event.getIterator();
    		while (iterator.hasNext()){
    			StreamEvent streamEvent = iterator.next();
    			eventCount++;
    		}

    		if(eventCount >= gpuProcessMinimumEventCount)
    		{
    			cudaEventList.clear();
    			inputStreamEventIndex = 0;

    			iterator = event.getIterator();
    			while (iterator.hasNext()){
    				StreamEvent streamEvent = iterator.next();

    				inputStreamEvents[inputStreamEventIndex++] = streamEvent;

    				CudaEvent cudaEvent = new CudaEvent(streamEvent.getTimestamp());

    				int i = 0;
    				for(Object attrib : streamEvent.getOutputData())
    				{
    					if(attrib instanceof Integer)
    					{
    						cudaEvent.AddIntAttribute(i++, ((Integer) attrib).intValue());
    					}
    					else if(attrib instanceof Long)
    					{
    						cudaEvent.AddLongAttribute(i++, ((Long) attrib).longValue());
    					}
    					else if(attrib instanceof Boolean)
    					{
    						cudaEvent.AddBoolAttribute(i++, ((Boolean) attrib).booleanValue());
    					}
    					else if(attrib instanceof Float)
    					{
    						cudaEvent.AddFloatAttribute(i++, ((Float) attrib).floatValue());
    					}
    					else if(attrib instanceof Double)
    					{
    						cudaEvent.AddDoubleAttribute(i++, ((Double) attrib).doubleValue());
    					}
    					else if(attrib instanceof String)
    					{
    						cudaEvent.AddStringAttribute(i++, attrib.toString());
    					}
    				}

    				cudaEventList.add(cudaEvent);
    			}

    			gpuEventConsumer.OnEvents(
    					new PointerPointer<SiddhiGpu.CudaEvent>(cudaEventList.toArray(new SiddhiGpu.CudaEvent[cudaEventList.size()])), 
    					cudaEventList.size());

    			IntPointer matchingEvents = gpuEventConsumer.GetMatchingEvents();

    			if(matchingEvents != null)
    			{
    				StreamEvent resultStreamEvent = inputStreamEvents[matchingEvents.get(0)];;

    				for(int i=1; i<matchingEvents.limit(); ++i) {
    					resultStreamEvent.addToLast(inputStreamEvents[matchingEvents.get(i)]); // optimize
    				}

    				this.next.process(resultStreamEvent);
    			}
    			else
    			{
    				log.debug("Result count : Empty");
    			}
    		}
    		else
    		{
    			iterator = event.getIterator();
        		while (iterator.hasNext()){
        			StreamEvent streamEvent = iterator.next();
        			if (!(Boolean) conditionExecutor.execute(streamEvent)){
        				iterator.remove();
        			}
        		}
        		
        		if(iterator.getFirstElement() != null){
        			this.next.process(iterator.getFirstElement());
        		}
    		}

    		//log.info("Batch count : " + count);

    		// #############################################################################################################
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
