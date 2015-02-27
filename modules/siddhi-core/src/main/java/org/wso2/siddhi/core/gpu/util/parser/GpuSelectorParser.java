package org.wso2.siddhi.core.gpu.util.parser;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import javassist.CannotCompileException;
import javassist.ClassPool;
import javassist.CtClass;
import javassist.CtMethod;
import javassist.NotFoundException;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.MetaComplexEvent;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.state.MetaStateEventAttribute;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.ConditionExpressionExecutor;
import org.wso2.siddhi.core.gpu.config.GpuQueryContext;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.input.stream.GpuStreamRuntime;
import org.wso2.siddhi.core.gpu.query.selector.GpuFilterQuerySelector;
import org.wso2.siddhi.core.gpu.query.selector.GpuJoinQuerySelector;
import org.wso2.siddhi.core.gpu.query.selector.GpuQuerySelector;
import org.wso2.siddhi.core.query.input.stream.StreamRuntime;
import org.wso2.siddhi.core.query.input.stream.join.JoinStreamRuntime;
import org.wso2.siddhi.core.query.selector.GroupByKeyGenerator;
import org.wso2.siddhi.core.query.selector.QuerySelector;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;
import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.core.util.parser.ExpressionParser;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.query.api.definition.StreamDefinition;
import org.wso2.siddhi.query.api.execution.query.output.stream.OutputStream;
import org.wso2.siddhi.query.api.execution.query.selection.OutputAttribute;
import org.wso2.siddhi.query.api.execution.query.selection.Selector;
import org.wso2.siddhi.query.api.expression.Expression;

public class GpuSelectorParser {
    
    
    public static GpuJoinQuerySelector getGpuJoinQuerySelector(GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext) {
        
        GpuJoinQuerySelector gpuQuerySelector = null;
        try {
            CtClass ctClass = ClassPool.getDefault().get("org.wso2.siddhi.core.gpu.query.selector.GpuJoinQuerySelector");
            CtMethod method = ctClass.getDeclaredMethod("deserialize");
            
            StringBuffer content = new StringBuffer();
            content.append("{\n ");

            content.append("int workSize = segmentsPerWorker * segmentEventCount; \n");
            content.append("int indexInsideSegment = 0; \n");
            content.append("int segIdx = 0; \n");
            content.append("ComplexEvent.Type type; \n");
            content.append("for (int resultsIndex = workerSize * workSize; resultsIndex < eventCount; ++resultsIndex) { \n");
            content.append("    segIdx = resultsIndex / segmentEventCount; \n");
            content.append("    type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes \n");
            content.append("    if(type != Type.NONE && type != Type.RESET) { \n");
            content.append("        StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            content.append("        borrowedEvent.setType(type);      \n");
            content.append("        long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes \n");
            content.append("        borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes \n");
            
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                case BOOL:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getShort(); \n");
                    break;
                case INT:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getInt(); \n");
                    break;
                case LONG:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getLong(); \n");
                    break;
                case FLOAT:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getFloat(); \n");
                    break;
                case DOUBLE:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getDouble(); \n");
                    break;
                case STRING:
                    content.append("                short length = inputEventBuffer.getShort(); \n");
                    content.append("                inputEventBuffer.get(preAllocatedByteArray, 0, attrib.length); \n");
                    content.append("                attributeData[").append(index++).append("] = new String(preAllocatedByteArray, 0, length);  \n");
                    break;
                }
            }

            content.append("        System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index); \n");
            content.append("        for (AttributeProcessor attributeProcessor : attributeProcessorList) { \n");
            content.append("            attributeProcessor.process(borrowedEvent); \n");
            content.append("        } \n");
            content.append("        if (workerfirstEvent == null) { \n");
            content.append("            workerfirstEvent = borrowedEvent; \n");
            content.append("            workerLastEvent = workerfirstEvent; \n");
            content.append("        } else { \n");
            content.append("            workerLastEvent.setNext(borrowedEvent); \n");
            content.append("            workerLastEvent = borrowedEvent; \n");
            content.append("        } \n");
            content.append("        indexInsideSegment++; \n");
            content.append("        indexInsideSegment = indexInsideSegment % segmentEventCount; \n");
            content.append("    } else if (type == Type.RESET) { \n");
            content.append("        outputEventBuffer.position( \n");
            content.append("                outputEventBuffer.position() +  \n");
            content.append("                ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes())  \n");
            content.append("                - 2); \n");
            content.append("        resultsIndex = ((segIdx + 1) * segmentEventCount) - 1; \n");
            content.append("        indexInsideSegment = 0; \n");
            content.append("    } \n");
            content.append("} \n");

            content.append("} ");
            
            method.setBody(content.toString());
//            ctClass.writeFile();
            gpuQuerySelector = (GpuJoinQuerySelector)ctClass.toClass().getConstructor(
                    String.class, Selector.class, Boolean.class, Boolean.class, ExecutionPlanContext.class)
                    .newInstance(id, selector, currentOn, expiredOn, executionPlanContext);
            
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
        
        return gpuQuerySelector;
        
    }
    
    public static GpuFilterQuerySelector getGpuFilterQuerySelector(GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext) {
        
        GpuFilterQuerySelector gpuQuerySelector = null;
        try {
            CtClass ctClass = ClassPool.getDefault().get("org.wso2.siddhi.core.gpu.query.selector.GpuFilterQuerySelector");
            CtMethod method = ctClass.getDeclaredMethod("deserialize");
            
            StringBuffer content = new StringBuffer();
            content.append("{\n ");

            content.append("for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) { \n");
            content.append("    int matched = outputEventIndexBuffer.get(); \n");
            content.append("    if (matched >= 0) { \n");
            content.append("        StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            content.append("        ComplexEvent.Type type = eventTypes[inputEventBuffer.getShort()]; \n");
            content.append("        long sequence = inputEventBuffer.getLong(); \n");
            content.append("        long timestamp = inputEventBuffer.getLong(); \n");
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                case BOOL:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getShort(); \n");
                    break;
                case INT:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getInt(); \n");
                    break;
                case LONG:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getLong(); \n");
                    break;
                case FLOAT:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getFloat(); \n");
                    break;
                case DOUBLE:
                    content.append("                attributeData[").append(index++).append("] = inputEventBuffer.getDouble(); \n");
                    break;
                case STRING:
                    content.append("                short length = inputEventBuffer.getShort(); \n");
                    content.append("                inputEventBuffer.get(preAllocatedByteArray, 0, attrib.length); \n");
                    content.append("                attributeData[").append(index++).append("] = new String(preAllocatedByteArray, 0, length);  \n");
                    break;
                }
            }
            content.append("        streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); \n");
            content.append("        for (AttributeProcessor attributeProcessor : attributeProcessorList) { \n");
            content.append("            attributeProcessor.process(borrowedEvent); \n");
            content.append("        } \n");
            content.append("        if (firstEvent == null) { \n");
            content.append("            firstEvent = borrowedEvent; \n");
            content.append("            lastEvent = firstEvent; \n");
            content.append("        } else { \n");
            content.append("            lastEvent.setNext(borrowedEvent); \n");
            content.append("            lastEvent = borrowedEvent; \n");
            content.append("        } \n");
            content.append("    } else { \n");
            content.append("        inputEventBuffer.position(inputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes()); \n");
            content.append("    } \n");
            content.append("} \n");

            content.append("} ");
            
            method.setBody(content.toString());
//            ctClass.writeFile();
            gpuQuerySelector = (GpuFilterQuerySelector)ctClass.toClass().getConstructor(
                    String.class, Selector.class, Boolean.class, Boolean.class, ExecutionPlanContext.class)
                    .newInstance(id, selector, currentOn, expiredOn, executionPlanContext);
            
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
        
        return gpuQuerySelector;
    }
    
    public static GpuQuerySelector getGpuQuerySelector(GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext) {
        
        GpuQuerySelector gpuQuerySelector = null;
        try {
            CtClass ctClass = ClassPool.getDefault().get("org.wso2.siddhi.core.gpu.query.selector.GpuQuerySelector");
            CtMethod method = ctClass.getDeclaredMethod("deserialize");
            
            StringBuffer content = new StringBuffer();
            content.append("{\n ");
            
            content.append("for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) { \n");
            content.append("    ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()];  \n");
            content.append("    if(type != Type.NONE) { \n");
            content.append("        StreamEvent borrowedEvent = streamEventPool.borrowEvent();         \n");
            content.append("        long sequence = outputEventBuffer.getLong(); \n");
            content.append("        long timestamp = outputEventBuffer.getLong(); \n");
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                    case BOOL:
                        content.append("                attributeData[").append(index++).append("] = outputEventBuffer.getShort(); \n");
                        break;
                    case INT:
                        content.append("                attributeData[").append(index++).append("] = outputEventBuffer.getInt(); \n");
                        break;
                    case LONG:
                        content.append("                attributeData[").append(index++).append("] = outputEventBuffer.getLong(); \n");
                        break;
                    case FLOAT:
                        content.append("                attributeData[").append(index++).append("] = outputEventBuffer.getFloat(); \n");
                        break;
                    case DOUBLE:
                        content.append("                attributeData[").append(index++).append("] = outputEventBuffer.getDouble(); \n");
                        break;
                    case STRING:
                        content.append("                short length = outputEventBuffer.getShort(); \n");
                        content.append("                outputEventBuffer.get(preAllocatedByteArray, 0, attrib.length); \n");
                        content.append("                attributeData[").append(index++).append("] = new String(preAllocatedByteArray, 0, length); \n");
                        break;
                }
            }
            content.append("        streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); \n");
            content.append("        for (AttributeProcessor attributeProcessor : attributeProcessorList) { \n");
            content.append("            attributeProcessor.process(borrowedEvent); \n");
            content.append("        } \n");
            content.append("        if (workerfirstEvent == null) { \n");
            content.append("            workerfirstEvent = borrowedEvent; \n");
            content.append("            workerLastEvent = workerfirstEvent; \n");
            content.append("        } else { \n");
            content.append("            workerLastEvent.setNext(borrowedEvent); \n");
            content.append("            workerLastEvent = borrowedEvent; \n");
            content.append("        }        \n");
            content.append("    } else { \n");
            content.append("        outputEventBuffer.position(outputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2); \n");
            content.append("    }    \n");
            content.append("} \n");

            content.append("} ");
            
            method.setBody(content.toString());
//            ctClass.writeFile();
            gpuQuerySelector = (GpuQuerySelector)ctClass.toClass().getConstructor(
                    String.class, Selector.class, Boolean.class, Boolean.class, ExecutionPlanContext.class)
                    .newInstance(id, selector, currentOn, expiredOn, executionPlanContext);
            
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
        
        return gpuQuerySelector;
        
    }
    
    /**
     * Parse Selector portion of a query and return corresponding QuerySelector
     *
     * @param selector             selector to be parsed
     * @param outStream            output stream of the query
     * @param executionPlanContext query to be parsed
     * @param metaComplexEvent     Meta event used to collect execution info of stream associated with query
     * @param executors            List to hold VariableExpressionExecutors to update after query parsing
     * @return
     */
    public static GpuQuerySelector parse(Selector selector, OutputStream outStream, ExecutionPlanContext executionPlanContext,
                                      MetaComplexEvent metaComplexEvent, List<VariableExpressionExecutor> executors,
                                      StreamRuntime streamRuntime, GpuQueryContext gpuQueryContext) {
        boolean currentOn = false;
        boolean expiredOn = false;
        String id = null;
        
        if (outStream.getOutputEventType() == OutputStream.OutputEventType.CURRENT_EVENTS || outStream.getOutputEventType() == OutputStream.OutputEventType.ALL_EVENTS) {
            currentOn = true;
        }
        if (outStream.getOutputEventType() == OutputStream.OutputEventType.EXPIRED_EVENTS || outStream.getOutputEventType() == OutputStream.OutputEventType.ALL_EVENTS) {
            expiredOn = true;
        }

        id = outStream.getId();
        List<AttributeProcessor> attributeProcessorList = getAttributeProcessors(selector, id, executionPlanContext, 
                metaComplexEvent, executors, streamRuntime, gpuQueryContext, currentOn, expiredOn);
        
        GpuQuerySelector querySelector = null;
        
        if(streamRuntime instanceof JoinStreamRuntime) {
            
            querySelector = getGpuJoinQuerySelector(gpuQueryContext.getOutputStreamMetaEvent(), id, selector, currentOn, expiredOn, executionPlanContext);
            
            if(querySelector == null) {
                querySelector = new GpuJoinQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext);
            }
        
        } else if (streamRuntime instanceof GpuStreamRuntime){
            
            List<SiddhiGpu.GpuProcessor> gpuProcessors = 
                    ((GpuProcessStreamReceiver)((GpuStreamRuntime) streamRuntime).getProcessStreamReceiver()).getGpuProcessors();

            SiddhiGpu.GpuProcessor lastGpuProcessor = gpuProcessors.get(gpuProcessors.size() - 1);
            if(lastGpuProcessor != null) {
                if(lastGpuProcessor instanceof SiddhiGpu.GpuFilterProcessor) {
                    
                    querySelector = getGpuFilterQuerySelector(gpuQueryContext.getOutputStreamMetaEvent(), id, selector, currentOn, expiredOn, executionPlanContext);
                    
                    if(querySelector == null) {
                        querySelector = new GpuFilterQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext);
                    }
                } else {
                    querySelector = getGpuQuerySelector(gpuQueryContext.getOutputStreamMetaEvent(), id, selector, currentOn, expiredOn, executionPlanContext);
                    
                    if(querySelector == null) {
                        querySelector = new GpuQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext);
                    }
                }
            } else {
                querySelector = getGpuQuerySelector(gpuQueryContext.getOutputStreamMetaEvent(), id, selector, currentOn, expiredOn, executionPlanContext);
                
                if(querySelector == null) {
                    querySelector = new GpuQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext);
                }
            }
        }
        
        querySelector.setAttributeProcessorList(attributeProcessorList);

        ConditionExpressionExecutor havingCondition = generateHavingExecutor(selector.getHavingExpression(),
                metaComplexEvent, executionPlanContext, executors);
        querySelector.setHavingConditionExecutor(havingCondition);
        if (!selector.getGroupByList().isEmpty()) {
            querySelector.setGroupByKeyGenerator(new GroupByKeyGenerator(selector.getGroupByList(), metaComplexEvent, executors, executionPlanContext));
        }


        return querySelector;
    }

    /**
     * Method to construct AttributeProcessor list for the selector
     *
     * @param selector
     * @param id
     * @param executionPlanContext
     * @param metaComplexEvent
     * @param executors
     * @return
     */
    private static List<AttributeProcessor> getAttributeProcessors(Selector selector, String id,
                                                                   ExecutionPlanContext executionPlanContext,
                                                                   MetaComplexEvent metaComplexEvent,
                                                                   List<VariableExpressionExecutor> executors,
                                                                   StreamRuntime streamRuntime,
                                                                   GpuQueryContext gpuQueryContext,
                                                                   boolean currentOn,
                                                                   boolean expiredOn) {

        List<AttributeProcessor> attributeProcessorList = new ArrayList<AttributeProcessor>();
        StreamDefinition temp = new StreamDefinition(id);
        
        SiddhiGpu.AttributeMappings attributeMappings = new SiddhiGpu.AttributeMappings(selector.getSelectionList().size());
        
        int i = 0;
        for (OutputAttribute outputAttribute : selector.getSelectionList()) {

            ExpressionExecutor expressionExecutor = ExpressionParser.parseExpression(outputAttribute.getExpression(),
                    metaComplexEvent, SiddhiConstants.UNKNOWN_STATE, executors, executionPlanContext,
                    !(selector.getGroupByList().isEmpty()), 0);
            
            if (expressionExecutor instanceof VariableExpressionExecutor) {   //for variables we will directly put value at conversion stage
                
                VariableExpressionExecutor executor = ((VariableExpressionExecutor) expressionExecutor);
                int streamIndex = executor.getPosition()[SiddhiConstants.STREAM_EVENT_CHAIN_INDEX];
                int attributeIndex = -1;
                
                if (metaComplexEvent instanceof MetaStateEvent) {
                    ((MetaStateEvent) metaComplexEvent).addOutputData(new MetaStateEventAttribute(executor.getAttribute(), executor.getPosition()));
                    attributeIndex = ((MetaStateEvent) metaComplexEvent).getMetaStreamEvent(streamIndex).getInputDefinition().getAttributePosition(executor.getAttribute().getName());
                } else {
                    ((MetaStreamEvent) metaComplexEvent).addOutputData(executor.getAttribute());
                    attributeIndex = ((MetaStreamEvent) metaComplexEvent).getInputDefinition().getAttributePosition(executor.getAttribute().getName());
                }
                temp.attribute(outputAttribute.getRename(), ((VariableExpressionExecutor) expressionExecutor).getAttribute().getType());
                
                attributeMappings.AddMapping(i, streamIndex, attributeIndex, i);
                
            } else {
                //To maintain output variable positions
                if (metaComplexEvent instanceof MetaStateEvent) {
                    ((MetaStateEvent) metaComplexEvent).addOutputData(null);
                } else {
                    ((MetaStreamEvent) metaComplexEvent).addOutputData(null);
                }
                AttributeProcessor attributeProcessor = new AttributeProcessor(expressionExecutor);
                attributeProcessor.setOutputPosition(i);
                attributeProcessorList.add(attributeProcessor);
                temp.attribute(outputAttribute.getRename(), attributeProcessor.getOutputType());
                attributeMappings.AddMapping(i, -1, -1, i);
            }
            i++;
        }
        metaComplexEvent.setOutputDefinition(temp);
        
        GpuMetaStreamEvent gpuMetaStreamEvent = new GpuMetaStreamEvent(id, temp, gpuQueryContext);
        gpuMetaStreamEvent.setStreamIndex(0);
        gpuQueryContext.setOutputStreamMetaEvent(gpuMetaStreamEvent);
        
        SiddhiGpu.GpuMetaEvent siddhiGpuMetaEvent = new SiddhiGpu.GpuMetaEvent(gpuMetaStreamEvent.getStreamIndex(), 
                gpuMetaStreamEvent.getAttributes().size(), gpuMetaStreamEvent.getEventSizeInBytes());

        int index = 0;
        for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
            int dataType = -1;

            switch(attrib.type) {
            case BOOL:
                dataType = SiddhiGpu.DataType.Boolean;
                break;
            case DOUBLE:
                dataType = SiddhiGpu.DataType.Double;
                break;
            case FLOAT:
                dataType = SiddhiGpu.DataType.Float;
                break;
            case INT:
                dataType = SiddhiGpu.DataType.Int;
                break;
            case LONG:
                dataType = SiddhiGpu.DataType.Long;
                break;
            case STRING:
                dataType = SiddhiGpu.DataType.StringIn;
                break;
            default:
                break;
            }

            siddhiGpuMetaEvent.SetAttribute(index++, dataType, attrib.length, attrib.position);    
        }
        
        // get last GpuProcessor
        GpuStreamRuntime gpuStreamRuntime = null;
        if(streamRuntime instanceof GpuStreamRuntime) {
            gpuStreamRuntime = (GpuStreamRuntime) streamRuntime;
        } else if (streamRuntime instanceof JoinStreamRuntime) {
            JoinStreamRuntime joinStreamRuntime = (JoinStreamRuntime) streamRuntime;
            gpuStreamRuntime = (GpuStreamRuntime)joinStreamRuntime.getSingleStreamRuntimes().get(0); // left and right both have same last processor
        }
        
        GpuProcessStreamReceiver gpuProcessStreamReceiver = (GpuProcessStreamReceiver)gpuStreamRuntime.getProcessStreamReceiver();
        
        List<SiddhiGpu.GpuProcessor> gpuProcessors = gpuProcessStreamReceiver.getGpuProcessors();
        SiddhiGpu.GpuProcessor lastGpuProcessor = gpuProcessors.get(gpuProcessors.size() - 1);
        if(lastGpuProcessor != null) {
            // set output stream definition and mapping
            lastGpuProcessor.SetOutputStream(siddhiGpuMetaEvent, attributeMappings);
            lastGpuProcessor.SetCurrentOn(currentOn);
            lastGpuProcessor.SetExpiredOn(expiredOn);
        }
        
        return attributeProcessorList;
    }

    private static ConditionExpressionExecutor generateHavingExecutor(Expression expression,
                                                                      MetaComplexEvent metaComplexEvent,
                                                                      ExecutionPlanContext executionPlanContext,
                                                                      List<VariableExpressionExecutor> executors) {
        ConditionExpressionExecutor havingConditionExecutor = null;
        if (expression != null) {
            havingConditionExecutor = (ConditionExpressionExecutor) ExpressionParser.parseExpression(expression,
                    metaComplexEvent, SiddhiConstants.HAVING_STATE, executors, executionPlanContext, false, 0);

        }
        return havingConditionExecutor;
    }
}
