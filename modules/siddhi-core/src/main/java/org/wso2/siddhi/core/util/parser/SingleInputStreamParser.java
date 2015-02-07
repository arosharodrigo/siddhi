/*
 * Copyright (c) 2014, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
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
import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.MetaComplexEvent;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.exception.DefinitionNotExistException;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.gpu.config.GpuQueryContext;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.input.stream.GpuStreamRuntime;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.gpu.util.parser.GpuFilterExpressionParser;
import org.wso2.siddhi.core.query.QueryAnnotations;
import org.wso2.siddhi.core.query.input.ProcessStreamReceiver;
import org.wso2.siddhi.core.query.input.stream.single.SingleStreamRuntime;
import org.wso2.siddhi.core.query.input.stream.single.SingleThreadEntryValveProcessor;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.query.processor.SchedulingProcessor;
import org.wso2.siddhi.core.query.processor.filter.FilterProcessor;
import org.wso2.siddhi.core.query.processor.stream.StreamProcessor;
import org.wso2.siddhi.core.query.processor.stream_function.StreamFunctionProcessor;
import org.wso2.siddhi.core.query.processor.window.LengthWindowProcessor;
import org.wso2.siddhi.core.query.processor.window.WindowProcessor;
import org.wso2.siddhi.core.util.Scheduler;
import org.wso2.siddhi.core.util.SiddhiClassLoader;
import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.query.api.definition.AbstractDefinition;
import org.wso2.siddhi.query.api.execution.query.input.handler.Filter;
import org.wso2.siddhi.query.api.execution.query.input.handler.StreamFunction;
import org.wso2.siddhi.query.api.execution.query.input.handler.StreamHandler;
import org.wso2.siddhi.query.api.execution.query.input.handler.Window;
import org.wso2.siddhi.query.api.execution.query.input.stream.SingleInputStream;
import org.wso2.siddhi.query.api.expression.Expression;

import java.util.List;
import java.util.Map;

public class SingleInputStreamParser {
    
    private static final Logger log = Logger.getLogger(SingleInputStreamParser.class);
    
    public static SingleStreamRuntime parseInputStream(SingleInputStream inputStream, ExecutionPlanContext executionPlanContext,
            List<VariableExpressionExecutor> executors, Map<String, AbstractDefinition> definitionMap,
            MetaComplexEvent metaComplexEvent, ProcessStreamReceiver processStreamReceiver, GpuQueryContext gpuQueryContext) {

        Processor processor = null;
        Processor singleThreadValve = null;
        boolean first = true;

        MetaStreamEvent metaStreamEvent;
        if (metaComplexEvent instanceof MetaStateEvent) {
            metaStreamEvent = new MetaStreamEvent();
            ((MetaStateEvent) metaComplexEvent).addEvent(metaStreamEvent);
            initMetaStreamEvent(inputStream, definitionMap, metaStreamEvent);
        } else {
            metaStreamEvent = (MetaStreamEvent) metaComplexEvent;
            initMetaStreamEvent(inputStream, definitionMap, metaStreamEvent);
        }

        if (!inputStream.getStreamHandlers().isEmpty()) {
            /* create processor chain from StreamHandlers */
            for (StreamHandler handler : inputStream.getStreamHandlers()) {

                Processor currentProcessor = generateProcessor(handler, metaComplexEvent, executors, executionPlanContext,
                        ((GpuProcessStreamReceiver)processStreamReceiver), gpuQueryContext);

                if (currentProcessor instanceof SchedulingProcessor) {
                    if (singleThreadValve == null) {

                        singleThreadValve = new SingleThreadEntryValveProcessor(executionPlanContext);
                        if (first) {
                            processor = singleThreadValve;
                            first = false;
                        } else {
                            processor.setToLast(singleThreadValve);
                        }
                    }
                    Scheduler scheduler = new Scheduler(executionPlanContext.getScheduledExecutorService(), singleThreadValve);
                    ((SchedulingProcessor) currentProcessor).setScheduler(scheduler);
                }

                if (first) {
                    processor = currentProcessor;
                    first = false;
                } else {
                    processor.setToLast(currentProcessor);
                }
            }
        }

        metaStreamEvent.initializeAfterWindowData();
        return new GpuStreamRuntime(processStreamReceiver, processor, metaComplexEvent);
    }

    /**
     * Parse single InputStream and return SingleStreamRuntime
     *
     * @param inputStream           single input stream to be parsed
     * @param executionPlanContext  query to be parsed
     * @param executors             List to hold VariableExpressionExecutors to update after query parsing
     * @param definitionMap
     * @param metaComplexEvent      @return
     * @param processStreamReceiver
     */
    public static SingleStreamRuntime parseInputStream(SingleInputStream inputStream, ExecutionPlanContext executionPlanContext,
                                                       List<VariableExpressionExecutor> executors, Map<String, AbstractDefinition> definitionMap,
                                                       MetaComplexEvent metaComplexEvent, ProcessStreamReceiver processStreamReceiver) {
        Processor processor = null;
        Processor singleThreadValve = null;
        boolean first = true;
        
        MetaStreamEvent metaStreamEvent;
        if (metaComplexEvent instanceof MetaStateEvent) {
            metaStreamEvent = new MetaStreamEvent();
            ((MetaStateEvent) metaComplexEvent).addEvent(metaStreamEvent);
            initMetaStreamEvent(inputStream, definitionMap, metaStreamEvent);
        } else {
            metaStreamEvent = (MetaStreamEvent) metaComplexEvent;
            initMetaStreamEvent(inputStream, definitionMap, metaStreamEvent);
        }
        
        if (!inputStream.getStreamHandlers().isEmpty()) {
            /* create processor chain from StreamHandlers */
            for (StreamHandler handler : inputStream.getStreamHandlers()) {

                Processor currentProcessor = generateProcessor(handler, metaComplexEvent, executors, executionPlanContext);

                if (currentProcessor instanceof SchedulingProcessor) {
                    if (singleThreadValve == null) {

                        singleThreadValve = new SingleThreadEntryValveProcessor(executionPlanContext);
                        if (first) {
                            processor = singleThreadValve;
                            first = false;
                        } else {
                            processor.setToLast(singleThreadValve);
                        }
                    }
                    Scheduler scheduler = new Scheduler(executionPlanContext.getScheduledExecutorService(), singleThreadValve);
                    ((SchedulingProcessor) currentProcessor).setScheduler(scheduler);
                }

                if (first) {
                    processor = currentProcessor;
                    first = false;
                } else {
                    processor.setToLast(currentProcessor);
                }
            }
        }

        metaStreamEvent.initializeAfterWindowData();
        return new SingleStreamRuntime(processStreamReceiver, processor, metaComplexEvent);
        
    }

    
    private static Processor generateProcessor(StreamHandler handler, MetaComplexEvent metaEvent, List<VariableExpressionExecutor> executors, 
            ExecutionPlanContext context, GpuProcessStreamReceiver gpuProcessStreamReceiver, GpuQueryContext gpuQueryContext) {
        
        ExpressionExecutor[] inputExpressions = new ExpressionExecutor[handler.getParameters().length];
        Expression[] parameters = handler.getParameters();
        MetaStreamEvent metaStreamEvent;
        int stateIndex = SiddhiConstants.UNKNOWN_STATE;
        if (metaEvent instanceof MetaStateEvent) {
            stateIndex = ((MetaStateEvent) metaEvent).getStreamEventCount() - 1;
            metaStreamEvent = ((MetaStateEvent) metaEvent).getMetaStreamEvent(stateIndex);
        } else {
            metaStreamEvent = (MetaStreamEvent) metaEvent;
        }
        for (int i = 0, parametersLength = parameters.length; i < parametersLength; i++) {
            inputExpressions[i] = ExpressionParser.parseExpression(parameters[i], metaEvent, stateIndex, executors,
                    context, false, SiddhiConstants.LAST);
        }
        
        String streamId = gpuProcessStreamReceiver.getStreamId();
        GpuQueryProcessor gpuQueryProcessor = gpuProcessStreamReceiver.getGpuQueryProcessor();
        SiddhiGpu.GpuQueryRuntime gpuQueryRuntime = gpuQueryProcessor.getGpuQueryRuntime();
        
        if (handler instanceof Filter) {
            
            // add filter processor 
            GpuFilterExpressionParser gpuExpressionParser = new GpuFilterExpressionParser();
            SiddhiGpu.GpuFilterProcessor gpuFilterProcessor = gpuExpressionParser.parseExpression(parameters[0], metaEvent, stateIndex, context, gpuQueryContext);
            gpuFilterProcessor.SetThreadBlockSize(gpuQueryContext.getThreadsPerBlock());
            gpuQueryRuntime.AddProcessor(streamId, gpuFilterProcessor);
            gpuQueryProcessor.AddGpuProcessor(gpuFilterProcessor);
            
            return new FilterProcessor(inputExpressions[0]);
            
        } else if (handler instanceof Window) {
            WindowProcessor windowProcessor = (WindowProcessor) SiddhiClassLoader.loadSiddhiImplementation(((Window) handler).getFunction(),
                    WindowProcessor.class);
            windowProcessor.initProcessor(metaStreamEvent.getInputDefinition(), inputExpressions);
            
            // TODO: only support length window
            if(windowProcessor instanceof LengthWindowProcessor)
            {
                LengthWindowProcessor lengthWindowProcessor = (LengthWindowProcessor) windowProcessor;
                
                SiddhiGpu.GpuLengthSlidingWindowProcessor gpuLengthWindowProcessor = 
                        new SiddhiGpu.GpuLengthSlidingWindowProcessor(lengthWindowProcessor.getLength());
                
                gpuQueryRuntime.AddProcessor(streamId, gpuLengthWindowProcessor);
                gpuQueryProcessor.AddGpuProcessor(gpuLengthWindowProcessor);
            }
            else
            {
                throw new OperationNotSupportedException(windowProcessor.getClass().getSimpleName() + " not supported in GPU processing");
            }
            
            return windowProcessor;

        } else if (handler instanceof StreamFunction) {
            StreamProcessor streamProcessor = (StreamFunctionProcessor) SiddhiClassLoader.loadSiddhiImplementation(
                    ((StreamFunction) handler).getFunction(), StreamFunctionProcessor.class);
            metaStreamEvent.setInputDefinition(streamProcessor.initProcessor(metaStreamEvent.getInputDefinition(),
                    inputExpressions));
            return streamProcessor;

        } else {
            throw new IllegalStateException(handler.getClass().getName() + " is not supported");
        }
    }

    private static Processor generateProcessor(StreamHandler handler, MetaComplexEvent metaEvent, List<VariableExpressionExecutor> executors, 
            ExecutionPlanContext context) {
        
        ExpressionExecutor[] inputExpressions = new ExpressionExecutor[handler.getParameters().length];
        Expression[] parameters = handler.getParameters();
        MetaStreamEvent metaStreamEvent;
        int stateIndex = SiddhiConstants.UNKNOWN_STATE;
        if (metaEvent instanceof MetaStateEvent) {
            stateIndex = ((MetaStateEvent) metaEvent).getStreamEventCount() - 1;
            metaStreamEvent = ((MetaStateEvent) metaEvent).getMetaStreamEvent(stateIndex);
        } else {
            metaStreamEvent = (MetaStreamEvent) metaEvent;
        }
        for (int i = 0, parametersLength = parameters.length; i < parametersLength; i++) {
            inputExpressions[i] = ExpressionParser.parseExpression(parameters[i], metaEvent, stateIndex, executors,
                    context, false, SiddhiConstants.LAST);
        }
        
        if (handler instanceof Filter) {
            return new FilterProcessor(inputExpressions[0]);

        } else if (handler instanceof Window) {
            WindowProcessor windowProcessor = (WindowProcessor) SiddhiClassLoader.loadSiddhiImplementation(((Window) handler).getFunction(),
                    WindowProcessor.class);
            windowProcessor.initProcessor(metaStreamEvent.getInputDefinition(), inputExpressions);
            return windowProcessor;

        } else if (handler instanceof StreamFunction) {
            StreamProcessor streamProcessor = (StreamFunctionProcessor) SiddhiClassLoader.loadSiddhiImplementation(
                    ((StreamFunction) handler).getFunction(), StreamFunctionProcessor.class);
            metaStreamEvent.setInputDefinition(streamProcessor.initProcessor(metaStreamEvent.getInputDefinition(),
                    inputExpressions));
            return streamProcessor;

        } else {
            throw new IllegalStateException(handler.getClass().getName() + " is not supported");
        }
    }

    /**
     * Method to generate MetaStreamEvent reagent to the given input stream. Empty definition will be created and
     * definition and reference is will be set accordingly in this method.
     *
     * @param inputStream
     * @param definitionMap
     * @param metaStreamEvent
     * @return
     */
    private static void initMetaStreamEvent(SingleInputStream inputStream, Map<String,AbstractDefinition> definitionMap, 
            MetaStreamEvent metaStreamEvent) {
        String streamId = inputStream.getStreamId();
        if(inputStream.isInnerStream()){
            streamId = "#".concat(streamId);
        }
        if (definitionMap != null && definitionMap.containsKey(streamId)) {
            AbstractDefinition inputDefinition = definitionMap.get(streamId);
            metaStreamEvent.setInputDefinition(inputDefinition);
            metaStreamEvent.setInitialAttributeSize(inputDefinition.getAttributeList().size());
        } else {
            throw new DefinitionNotExistException("Stream definition with stream ID '" + inputStream.getStreamId() + "' has not been defined");
        }
        if ((inputStream.getStreamReferenceId() != null) &&
                !(inputStream.getStreamId()).equals(inputStream.getStreamReferenceId())) { //if ref id is provided
            metaStreamEvent.setInputReferenceId(inputStream.getStreamReferenceId());
        }
    }


}
