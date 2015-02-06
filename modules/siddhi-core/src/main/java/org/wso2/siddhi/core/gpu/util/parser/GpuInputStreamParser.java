package org.wso2.siddhi.core.gpu.util.parser;

import java.util.List;
import java.util.Map;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.exception.DefinitionNotExistException;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.gpu.config.GpuQueryContext;
import org.wso2.siddhi.core.gpu.event.state.GpuMetaStateEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.query.QueryAnnotations;
import org.wso2.siddhi.core.query.input.MultiProcessStreamReceiver;
import org.wso2.siddhi.core.query.input.ProcessStreamReceiver;
import org.wso2.siddhi.core.query.input.stream.StreamRuntime;
import org.wso2.siddhi.core.query.input.stream.join.JoinStreamRuntime;
import org.wso2.siddhi.core.query.input.stream.single.SingleStreamRuntime;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.query.processor.filter.FilterProcessor;
import org.wso2.siddhi.core.query.processor.window.LengthWindowProcessor;
import org.wso2.siddhi.core.util.parser.JoinInputStreamParser;
import org.wso2.siddhi.core.util.parser.SingleInputStreamParser;
import org.wso2.siddhi.core.util.parser.StateInputStreamParser;
import org.wso2.siddhi.query.api.definition.AbstractDefinition;
import org.wso2.siddhi.query.api.execution.query.input.stream.BasicSingleInputStream;
import org.wso2.siddhi.query.api.execution.query.input.stream.InputStream;
import org.wso2.siddhi.query.api.execution.query.input.stream.JoinInputStream;
import org.wso2.siddhi.query.api.execution.query.input.stream.SingleInputStream;
import org.wso2.siddhi.query.api.execution.query.input.stream.StateInputStream;

public class GpuInputStreamParser {

    public static StreamRuntime parse(InputStream inputStream, ExecutionPlanContext executionPlanContext,
            Map<String, AbstractDefinition> definitionMap,
            List<VariableExpressionExecutor> executors,
            GpuQueryContext gpuQueryContext) {
        
        if (inputStream instanceof BasicSingleInputStream || inputStream instanceof SingleInputStream) {

            GpuMetaStreamEvent gpuMetaEvent = new GpuMetaStreamEvent(inputStream, definitionMap, gpuQueryContext);
            GpuQueryProcessor gpuQueryProcessor = new GpuQueryProcessor(gpuMetaEvent, gpuQueryContext);
            GpuProcessStreamReceiver processStreamReceiver = new GpuProcessStreamReceiver(((SingleInputStream) inputStream).getStreamId(), 
                    gpuQueryContext.getQueryName());
            gpuQueryProcessor.addStream(((SingleInputStream) inputStream).getStreamId(), gpuMetaEvent);
            processStreamReceiver.setGpuMetaEvent(gpuMetaEvent);
            processStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor);
            
            return SingleInputStreamParser.parseInputStream((SingleInputStream) inputStream,
                    executionPlanContext, executors, definitionMap, new MetaStreamEvent(), processStreamReceiver, gpuQueryContext);

        } else if (inputStream instanceof JoinInputStream) {

            GpuMetaStateEvent gpuMetaStateEvent = new GpuMetaStateEvent(2);
            
            SingleInputStream leftInputStream = (SingleInputStream) ((JoinInputStream) inputStream).getLeftInputStream();
            SingleInputStream rightInputStream = (SingleInputStream) ((JoinInputStream) inputStream).getRightInputStream();
            
            gpuMetaStateEvent.addEvent(new GpuMetaStreamEvent(leftInputStream, definitionMap, gpuQueryContext));
            gpuMetaStateEvent.addEvent(new GpuMetaStreamEvent(rightInputStream, definitionMap, gpuQueryContext));

            GpuQueryProcessor gpuQueryProcessor = new GpuQueryProcessor(gpuMetaStateEvent, gpuQueryContext);
            
            GpuProcessStreamReceiver leftGpuProcessStreamReceiver = new GpuProcessStreamReceiver(leftInputStream.getStreamId(), gpuQueryContext.getQueryName() + "_left");
            GpuProcessStreamReceiver rightGpuProcessStreamReceiver = new GpuProcessStreamReceiver(rightInputStream.getStreamId(), gpuQueryContext.getQueryName() + "_right");
            gpuQueryProcessor.addStream(leftInputStream.getStreamId(), gpuMetaStateEvent.getMetaStreamEvent(0));
            gpuQueryProcessor.addStream(rightInputStream.getStreamId(), gpuMetaStateEvent.getMetaStreamEvent(1));
            
            leftGpuProcessStreamReceiver.setGpuMetaEvent(gpuMetaStateEvent.getMetaStreamEvent(0));
            rightGpuProcessStreamReceiver.setGpuMetaEvent(gpuMetaStateEvent.getMetaStreamEvent(1));
            
            leftGpuProcessStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor);
            rightGpuProcessStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor);
            
            MetaStateEvent metaStateEvent = new MetaStateEvent(2);
            metaStateEvent.addEvent(new MetaStreamEvent());
            metaStateEvent.addEvent(new MetaStreamEvent());
            
            SingleStreamRuntime leftStreamRuntime = SingleInputStreamParser.parseInputStream(
                    (SingleInputStream) ((JoinInputStream) inputStream).getLeftInputStream(),
                    executionPlanContext, executors, definitionMap,
                    metaStateEvent.getMetaStreamEvent(0), leftGpuProcessStreamReceiver, gpuQueryContext);
            
            SingleStreamRuntime rightStreamRuntime = SingleInputStreamParser.parseInputStream(
                    (SingleInputStream) ((JoinInputStream) inputStream).getRightInputStream(),
                    executionPlanContext, executors, definitionMap,
                    metaStateEvent.getMetaStreamEvent(1), rightGpuProcessStreamReceiver, gpuQueryContext);
            
            return JoinInputStreamParser.parseInputStream(leftStreamRuntime, rightStreamRuntime,
                    (JoinInputStream) inputStream, executionPlanContext, metaStateEvent, executors, gpuQueryContext);
            
//            ///////
//            
//            ProcessStreamReceiver leftProcessStreamReceiver;
//            ProcessStreamReceiver rightProcessStreamReceiver;
//            if (inputStream.getAllStreamIds().size() == 2) {
//                leftProcessStreamReceiver = new ProcessStreamReceiver(((SingleInputStream) ((JoinInputStream) inputStream)
//                        .getLeftInputStream()).getStreamId());
//                rightProcessStreamReceiver = new ProcessStreamReceiver(((SingleInputStream) ((JoinInputStream) inputStream)
//                        .getRightInputStream()).getStreamId());
//            } else {
//                rightProcessStreamReceiver = new MultiProcessStreamReceiver(inputStream.getAllStreamIds().get(0), 2);
//                leftProcessStreamReceiver = rightProcessStreamReceiver;
//            }
//            MetaStateEvent metaStateEvent = new MetaStateEvent(2);
//            metaStateEvent.addEvent(new MetaStreamEvent());
//            metaStateEvent.addEvent(new MetaStreamEvent());
//
//            SingleStreamRuntime leftStreamRuntime = SingleInputStreamParser.parseInputStream(
//                    (SingleInputStream) ((JoinInputStream) inputStream).getLeftInputStream(),
//                    executionPlanContext, executors, definitionMap,
//                    metaStateEvent.getMetaStreamEvent(0), leftProcessStreamReceiver);
//
//            SingleStreamRuntime rightStreamRuntime = SingleInputStreamParser.parseInputStream(
//                    (SingleInputStream) ((JoinInputStream) inputStream).getRightInputStream(),
//                    executionPlanContext, executors, definitionMap,
//                    metaStateEvent.getMetaStreamEvent(1), rightProcessStreamReceiver);
//
//            return JoinInputStreamParser.parseInputStream(leftStreamRuntime, rightStreamRuntime,
//                    (JoinInputStream) inputStream, executionPlanContext, metaStateEvent, executors);
            
        } else if (inputStream instanceof StateInputStream) {
            MetaStateEvent metaStateEvent = new MetaStateEvent(inputStream.getAllStreamIds().size());
            return StateInputStreamParser.parseInputStream(((StateInputStream) inputStream), executionPlanContext,
                    metaStateEvent, executors, definitionMap);
        } else {
            // TODO: pattern, etc
            throw new OperationNotSupportedException();
        }
    }

    /**
     * Method to generate MetaStreamEvent reagent to the given input stream.
     * Empty definition will be created and definition and reference is will be
     * set accordingly in this method.
     *
     * @param inputStream
     * @param definitionMap
     * @return
     */
    public static MetaStreamEvent generateMetaStreamEvent(SingleInputStream inputStream, Map<String,
            AbstractDefinition> definitionMap) {
        MetaStreamEvent metaStreamEvent = new MetaStreamEvent();
        String streamId = inputStream.getStreamId();
        if (inputStream.isInnerStream()) {
            streamId = "#".concat(streamId);
        }
        if (definitionMap != null && definitionMap.containsKey(streamId)) {
            AbstractDefinition inputDefinition = definitionMap.get(streamId);
            metaStreamEvent.setInputDefinition(inputDefinition);
            metaStreamEvent.setInitialAttributeSize(inputDefinition.getAttributeList().size());
        } else {
            throw new DefinitionNotExistException("Stream definition with stream ID " + inputStream.getStreamId() + " has not been defined");
        }
        if ((inputStream.getStreamReferenceId() != null) &&
                !(inputStream.getStreamId()).equals(inputStream.getStreamReferenceId())) { 
            metaStreamEvent.setInputReferenceId(inputStream.getStreamReferenceId());
        }
        return metaStreamEvent;
    }
}
