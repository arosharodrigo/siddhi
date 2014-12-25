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
package org.wso2.siddhi.core.util.parser;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.config.SiddhiContext;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.query.QueryAnnotations;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.query.processor.filter.FilterProcessor;
import org.wso2.siddhi.core.query.processor.window.WindowProcessor;
import org.wso2.siddhi.core.stream.QueryStreamReceiver;
import org.wso2.siddhi.core.stream.runtime.SingleStreamRuntime;
import org.wso2.siddhi.core.util.SiddhiClassLoader;
import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.query.api.definition.StreamDefinition;
import org.wso2.siddhi.query.api.execution.query.input.handler.Filter;
import org.wso2.siddhi.query.api.execution.query.input.handler.StreamHandler;
import org.wso2.siddhi.query.api.execution.query.input.handler.Window;
import org.wso2.siddhi.query.api.execution.query.input.stream.SingleInputStream;
import org.wso2.siddhi.query.api.expression.Expression;

import java.util.List;

public class SingleInputStreamParser {

	private static final Logger log = Logger.getLogger(SingleInputStreamParser.class);
	
    /**
     * Parse single InputStream and return SingleStreamRuntime
     *
     * @param inputStream     single input stream to be parsed
     * @param context         query to be parsed
     * @param metaStreamEvent Meta event used to collect execution info of stream associated with query
     * @param executors       List to hold VariableExpressionExecutors to update after query parsing
     * @return
     */
    public static SingleStreamRuntime parseInputStream(SingleInputStream inputStream, SiddhiContext context,
                                                       MetaStreamEvent metaStreamEvent, List<VariableExpressionExecutor> executors,
                                                       QueryAnnotations queryAnnotations) {
        Processor processor = null;
        int i = 0;
        if (!inputStream.getStreamHandlers().isEmpty()) {
            for (StreamHandler handler : inputStream.getStreamHandlers()) {
                if (i == 0) {
                    processor = generateProcessor(handler, context, metaStreamEvent, executors, queryAnnotations);
                    i++;
                } else {
                    processor.setToLast(generateProcessor(handler, context, metaStreamEvent, executors, queryAnnotations));
                }
            }
        }
        metaStreamEvent.intializeAfterWindowData();
        QueryStreamReceiver queryStreamReceiver = new QueryStreamReceiver((StreamDefinition) metaStreamEvent.getInputDefinition());
        return new SingleStreamRuntime(queryStreamReceiver, processor);
    }

    private static Processor generateProcessor(StreamHandler handler, SiddhiContext context, MetaStreamEvent metaStreamEvent,
                                               List<VariableExpressionExecutor> executors,
                                               QueryAnnotations queryAnnotations) {
        if (handler instanceof Filter) {
            Expression condition = ((Filter) handler).getFilterExpression();
            
            if(queryAnnotations.GetAnnotationBooleanValue(SiddhiConstants.ANNOTATION_GPU, SiddhiConstants.ANNOTATION_ELEMENT_GPU_FILTER))
            {
            	Integer eventsPerBlock = queryAnnotations.GetAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
            			SiddhiConstants.ANNOTATION_ELEMENT_GPU_BLOCK_SIZE);
            	
            	Integer minEventCount = queryAnnotations.GetAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
            			SiddhiConstants.ANNOTATION_ELEMENT_GPU_MIN_EVENT_COUNT);
            	
            	String stringAttributeSizes = queryAnnotations.GetAnnotationStringValue(SiddhiConstants.ANNOTATION_GPU, 
            			SiddhiConstants.ANNOTATION_ELEMENT_GPU_STRING_SIZES);
            	
            	String queryName = queryAnnotations.GetAnnotationStringValue(SiddhiConstants.ANNOTATION_INFO, 
            		SiddhiConstants.ANNOTATION_ELEMENT_INFO_NAME);
            	 
            	if(eventsPerBlock == null)
            	{
            		eventsPerBlock = new Integer(256);
            	}
            	
            	if(minEventCount == null)
            	{
            		minEventCount = eventsPerBlock;
            	}
            	
            	SiddhiGpu.GpuEventConsumer gpuEventConsumer = new SiddhiGpu.GpuEventConsumer(
            			SiddhiGpu.SingleFilterKernel, queryName, context.getDefaultEventBufferSize(), eventsPerBlock);
            	
            	gpuEventConsumer.Initialize();

            	try {
            	    GpuExpressionParser gpuExpressionParser = new GpuExpressionParser();

            	    SiddhiGpu.Filter gpuFilter = gpuExpressionParser.parseExpression(condition, context, metaStreamEvent);
            	    gpuEventConsumer.AddFilter(gpuFilter);
            	    gpuEventConsumer.ConfigureFilters();

            	    FilterProcessor filterProcessor = new FilterProcessor(
            		    ExpressionParser.parseExpression(condition, context, metaStreamEvent, executors, false),
            		    gpuEventConsumer, minEventCount, stringAttributeSizes); 
            	    filterProcessor.setVariablePositionToAttributeNameMapper(gpuExpressionParser.getVariablePositionToAttributeNameMapper());

            	    return filterProcessor;

            	} catch(RuntimeException ex) {
            	    log.info("GPU Filter creation failed : " + ex.getMessage());
            	    ex.printStackTrace();
            	    return new FilterProcessor(ExpressionParser.parseExpression(condition, context, metaStreamEvent, executors, false));
            	}
            }
            else
            {
            	return new FilterProcessor(ExpressionParser.parseExpression(condition, context, metaStreamEvent, executors, false));  //metaStreamEvent has stream definition info
            }
            
        } else if (handler instanceof Window) {
            WindowProcessor windowProcessor = (WindowProcessor) SiddhiClassLoader.loadSiddhiImplementation(((Window) handler).getFunction(),
                    WindowProcessor.class);
            windowProcessor.setParameters(((Window) handler).getParameters());
            windowProcessor.init();
            return windowProcessor;

        } else {
            //TODO else if (window function etc)
            throw new OperationNotSupportedException("Only filter operation is supported at the moment");
        }
    }
}
