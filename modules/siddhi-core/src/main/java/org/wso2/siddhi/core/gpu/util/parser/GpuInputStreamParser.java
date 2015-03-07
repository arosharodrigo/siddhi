package org.wso2.siddhi.core.gpu.util.parser;

import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;

import javassist.CannotCompileException;
import javassist.ClassPool;
import javassist.CtClass;
import javassist.CtConstructor;
import javassist.CtMethod;
import javassist.CtNewConstructor;
import javassist.CtNewMethod;
import javassist.Modifier;
import javassist.NotFoundException;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.exception.DefinitionNotExistException;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.gpu.config.GpuQueryContext;
import org.wso2.siddhi.core.gpu.event.state.GpuMetaStateEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.query.input.stream.StreamRuntime;
import org.wso2.siddhi.core.query.input.stream.single.SingleStreamRuntime;
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

    public static GpuProcessStreamReceiver getGpuProcessStreamReceiver(GpuMetaStreamEvent gpuMetaEvent, 
            String streamId, String referenceId, String queryName) {
        
        GpuProcessStreamReceiver processStreamReceiver = null;
        try {
            ClassPool pool = ClassPool.getDefault();
            
            String className = "GpuProcessStreamReceiver" + streamId + (referenceId != null ? referenceId : "") + queryName;
            String fqdn = "org.wso2.siddhi.core.gpu.query.input.gen." + className;
            CtClass gpuProcStrmRecevrClass = pool.makeClass(fqdn);
            final CtClass superClass = pool.get( "org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver" );
            gpuProcStrmRecevrClass.setSuperclass(superClass);
            gpuProcStrmRecevrClass.setModifiers( Modifier.PUBLIC );
            
            StringBuffer constructor = new StringBuffer();
            constructor.append("public ").append(className).append("(String streamId, String referenceId, String queryName) {");
            constructor.append("   super(streamId, referenceId, queryName); ");
            constructor.append("}");
            
            CtConstructor ctConstructor = CtNewConstructor.make(constructor.toString(), gpuProcStrmRecevrClass);
            gpuProcStrmRecevrClass.addConstructor(ctConstructor);
            
            // -- public void serialize(Event event) --
            StringBuilder serializeBuffer = new StringBuilder();

            serializeBuffer.append("public void serialize(org.wso2.siddhi.core.event.Event event) { ");

            serializeBuffer.append("eventBufferWriter.writeShort((short)(!event.isExpired() ? 0 : 1)); \n");
            serializeBuffer.append("eventBufferWriter.writeLong(gpuQueryProcessor.getNextSequenceNumber()); \n");
            serializeBuffer.append("eventBufferWriter.writeLong(event.getTimestamp()); \n");
            serializeBuffer.append("Object [] data = event.getData(); \n");
            int index = 0; 
            
            for (GpuEventAttribute attrib : gpuMetaEvent.getAttributes()) {
                switch(attrib.type) {
                case BOOL:
                    serializeBuffer.append("eventBufferWriter.writeBool(((Boolean) data[").append(index++).append("]).booleanValue()); \n");
                    break;
                case INT:
                    serializeBuffer.append("eventBufferWriter.writeInt(((Integer) data[").append(index++).append("]).intValue()); \n");
                    break;
                case LONG:
                    serializeBuffer.append("eventBufferWriter.writeLong(((Long) data[").append(index++).append("]).longValue()); \n");
                    break;
                case FLOAT:
                    serializeBuffer.append("eventBufferWriter.writeFloat(((Float) data[").append(index++).append("]).floatValue()); \n");
                    break;
                case DOUBLE:
                    serializeBuffer.append("eventBufferWriter.writeDouble(((Double) data[").append(index++).append("]).doubleValue()); \n");
                    break;
                case STRING: 
                    serializeBuffer.append("eventBufferWriter.writeString((String) data[").append(index++).append("], ").append(attrib.length).append("); \n");
                    break;
                }
            }
            
            serializeBuffer.append("}");

            CtMethod serializeMethod = CtNewMethod.make(serializeBuffer.toString(), gpuProcStrmRecevrClass);
            gpuProcStrmRecevrClass.addMethod(serializeMethod);
            
//            gpuProcStrmRecevrClass.debugWriteFile("/home/prabodha/javassist");
            processStreamReceiver = (GpuProcessStreamReceiver)gpuProcStrmRecevrClass.toClass().getConstructor(String.class, String.class, String.class)
                  .newInstance(streamId, referenceId, queryName);
            
        } catch (NotFoundException e) {
            e.printStackTrace();
        } catch (CannotCompileException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (SecurityException e) {
            e.printStackTrace();
        }
        
        return processStreamReceiver;
    }
    
    public static StreamRuntime parse(InputStream inputStream, ExecutionPlanContext executionPlanContext,
            Map<String, AbstractDefinition> definitionMap,
            List<VariableExpressionExecutor> executors,
            GpuQueryContext gpuQueryContext) {
        
        if (inputStream instanceof BasicSingleInputStream || inputStream instanceof SingleInputStream) {

            GpuMetaStreamEvent gpuMetaEvent = new GpuMetaStreamEvent(inputStream, definitionMap, gpuQueryContext);
            gpuMetaEvent.setStreamIndex(0);
            GpuQueryProcessor gpuQueryProcessor = new GpuQueryProcessor(gpuQueryContext, gpuQueryContext.getQueryName());
            
            GpuProcessStreamReceiver processStreamReceiver = getGpuProcessStreamReceiver(gpuMetaEvent, 
                    ((SingleInputStream) inputStream).getStreamId(), 
                    ((SingleInputStream) inputStream).getStreamReferenceId(),
                    gpuQueryContext.getQueryName());

            if(processStreamReceiver == null) {
                processStreamReceiver = new GpuProcessStreamReceiver(((SingleInputStream) inputStream).getStreamId(), 
                        ((SingleInputStream) inputStream).getStreamReferenceId(),
                        gpuQueryContext.getQueryName());
            }
            gpuQueryProcessor.addStream(((SingleInputStream) inputStream).getStreamId(), gpuMetaEvent);
            processStreamReceiver.setGpuMetaEvent(gpuMetaEvent);
            processStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor);
            processStreamReceiver.setPerfromanceCalculateBatchCount(gpuQueryContext.getPerfromanceCalculateBatchCount());
            processStreamReceiver.setSoftBatchScheduling(gpuQueryContext.isBatchSoftScheduling());
            processStreamReceiver.setMaximumEventBatchSize(gpuQueryContext.getEventBatchMaximumSize());
            processStreamReceiver.setMinimumEventBatchSize(gpuQueryContext.getEventBatchMinimumSize());
            processStreamReceiver.setSelectorWorkerCount(gpuQueryContext.getSelectorWorkerCount());
            
            return SingleInputStreamParser.parseInputStream((SingleInputStream) inputStream,
                    executionPlanContext, executors, definitionMap, new MetaStreamEvent(), processStreamReceiver, gpuQueryContext);

        } else if (inputStream instanceof JoinInputStream) {

            GpuMetaStateEvent gpuMetaStateEvent = new GpuMetaStateEvent(2);
            
            SingleInputStream leftInputStream = (SingleInputStream) ((JoinInputStream) inputStream).getLeftInputStream();
            SingleInputStream rightInputStream = (SingleInputStream) ((JoinInputStream) inputStream).getRightInputStream();
            
            gpuMetaStateEvent.addEvent(new GpuMetaStreamEvent(leftInputStream, definitionMap, gpuQueryContext));
            gpuMetaStateEvent.addEvent(new GpuMetaStreamEvent(rightInputStream, definitionMap, gpuQueryContext));

            GpuQueryProcessor gpuQueryProcessor = new GpuQueryProcessor(gpuQueryContext, gpuQueryContext.getQueryName());
            
            String leftStreamId = leftInputStream.getStreamId() + (leftInputStream.getStreamReferenceId() != null ? "_" + leftInputStream.getStreamReferenceId() : "");
            GpuProcessStreamReceiver leftGpuProcessStreamReceiver = getGpuProcessStreamReceiver(
                    gpuMetaStateEvent.getMetaStreamEvent(0), 
                    leftInputStream.getStreamId(), 
                    leftInputStream.getStreamReferenceId(),
                    gpuQueryContext.getQueryName() + "_left");
                    
            if(leftGpuProcessStreamReceiver == null) {
                leftGpuProcessStreamReceiver = new GpuProcessStreamReceiver(leftInputStream.getStreamId(), 
                        leftInputStream.getStreamReferenceId(),
                        gpuQueryContext.getQueryName() + "_left");
            }
            
            
            String rightStreamId = rightInputStream.getStreamId() + (rightInputStream.getStreamReferenceId() != null ? "_" + rightInputStream.getStreamReferenceId() : "");
            GpuProcessStreamReceiver rightGpuProcessStreamReceiver = getGpuProcessStreamReceiver(
                    gpuMetaStateEvent.getMetaStreamEvent(1), 
                    rightInputStream.getStreamId(),
                    rightInputStream.getStreamReferenceId(),
                    gpuQueryContext.getQueryName() + "_right");

            if(rightGpuProcessStreamReceiver == null) {
                rightGpuProcessStreamReceiver = new GpuProcessStreamReceiver(rightInputStream.getStreamId(), 
                        rightInputStream.getStreamReferenceId(),
                        gpuQueryContext.getQueryName() + "_right");
            }
            
            gpuQueryProcessor.addStream(leftStreamId, gpuMetaStateEvent.getMetaStreamEvent(0));
            gpuQueryProcessor.addStream(rightStreamId, gpuMetaStateEvent.getMetaStreamEvent(1));
            
            leftGpuProcessStreamReceiver.setGpuMetaEvent(gpuMetaStateEvent.getMetaStreamEvent(0));
            rightGpuProcessStreamReceiver.setGpuMetaEvent(gpuMetaStateEvent.getMetaStreamEvent(1));
            
            leftGpuProcessStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor);
            rightGpuProcessStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor);
            
            leftGpuProcessStreamReceiver.setPerfromanceCalculateBatchCount(gpuQueryContext.getPerfromanceCalculateBatchCount());
            rightGpuProcessStreamReceiver.setPerfromanceCalculateBatchCount(gpuQueryContext.getPerfromanceCalculateBatchCount());
            
            leftGpuProcessStreamReceiver.setSoftBatchScheduling(gpuQueryContext.isBatchSoftScheduling());
            rightGpuProcessStreamReceiver.setSoftBatchScheduling(gpuQueryContext.isBatchSoftScheduling());
            
            leftGpuProcessStreamReceiver.setMaximumEventBatchSize(gpuQueryContext.getEventBatchMaximumSize());
            rightGpuProcessStreamReceiver.setMaximumEventBatchSize(gpuQueryContext.getEventBatchMaximumSize());
            
            leftGpuProcessStreamReceiver.setMinimumEventBatchSize(gpuQueryContext.getEventBatchMinimumSize());
            rightGpuProcessStreamReceiver.setMinimumEventBatchSize(gpuQueryContext.getEventBatchMinimumSize());
            
            leftGpuProcessStreamReceiver.setSelectorWorkerCount(gpuQueryContext.getSelectorWorkerCount());
            rightGpuProcessStreamReceiver.setSelectorWorkerCount(gpuQueryContext.getSelectorWorkerCount());
            
            MetaStateEvent metaStateEvent = new MetaStateEvent(2);
            
            SingleStreamRuntime leftStreamRuntime = SingleInputStreamParser.parseInputStream(
                    (SingleInputStream) ((JoinInputStream) inputStream).getLeftInputStream(),
                    executionPlanContext, executors, definitionMap,
                    metaStateEvent, leftGpuProcessStreamReceiver, gpuQueryContext);
            
            SingleStreamRuntime rightStreamRuntime = SingleInputStreamParser.parseInputStream(
                    (SingleInputStream) ((JoinInputStream) inputStream).getRightInputStream(),
                    executionPlanContext, executors, definitionMap,
                    metaStateEvent, rightGpuProcessStreamReceiver, gpuQueryContext);
            
            return JoinInputStreamParser.parseInputStream(leftStreamRuntime, rightStreamRuntime,
                    (JoinInputStream) inputStream, executionPlanContext, metaStateEvent, executors, 
                    leftGpuProcessStreamReceiver, rightGpuProcessStreamReceiver, gpuQueryContext);
                        
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
