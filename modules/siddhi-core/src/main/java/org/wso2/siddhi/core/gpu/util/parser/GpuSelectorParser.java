package org.wso2.siddhi.core.gpu.util.parser;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

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
    
    
    public static GpuJoinQuerySelector getGpuJoinQuerySelector(String classId, GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext) {
        
        GpuJoinQuerySelector gpuQuerySelector = null;
        try {
            ClassPool pool = ClassPool.getDefault();
            
            Random r = new Random(System.nanoTime());
            String className = "GpuJoinQuerySelector" + classId + Integer.toString(r.nextInt());
            String fqdn = "org.wso2.siddhi.core.gpu.query.selector.gen." + className;
            CtClass gpuJoinQuerySelectorClass = pool.makeClass(fqdn);
            final CtClass superClass = pool.get( "org.wso2.siddhi.core.gpu.query.selector.GpuJoinQuerySelector" );
            gpuJoinQuerySelectorClass.setSuperclass(superClass);
            gpuJoinQuerySelectorClass.setModifiers( Modifier.PUBLIC );
            
            // public GpuJoinQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext, String queryName)
            
            StringBuffer constructor = new StringBuffer();
            constructor.append("public ").append(className).append("(String id, ")
                .append("org.wso2.siddhi.query.api.execution.query.selection.Selector selector, ")
                .append("boolean currentOn, ")
                .append("boolean expiredOn, ")
                .append("org.wso2.siddhi.core.config.ExecutionPlanContext executionPlanContext, ")
                .append("String queryName) {");
            constructor.append("   super(id, selector, currentOn, expiredOn, executionPlanContext, queryName); ");
            constructor.append("}");
            
            CtConstructor ctConstructor = CtNewConstructor.make(constructor.toString(), gpuJoinQuerySelectorClass);
            gpuJoinQuerySelectorClass.addConstructor(ctConstructor);
            
            // -- protected void deserialize(int eventCount) --
            StringBuilder deserializeBuffer = new StringBuilder();

            deserializeBuffer.append("protected void deserialize(int eventCount) { ");

            deserializeBuffer.append("int workSize = segmentsPerWorker * segmentEventCount; \n");
            deserializeBuffer.append("int indexInsideSegment = 0; \n");
            deserializeBuffer.append("int segIdx = 0; \n");
            deserializeBuffer.append("org.wso2.siddhi.core.event.ComplexEvent.Type type; \n");
            deserializeBuffer.append("for (int resultsIndex = workerSize * workSize; resultsIndex < eventCount; ++resultsIndex) { \n");
            deserializeBuffer.append("    segIdx = resultsIndex / segmentEventCount; \n");
            deserializeBuffer.append("    type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes \n");
            deserializeBuffer.append("    if(type != org.wso2.siddhi.core.event.ComplexEvent.Type.NONE && type != org.wso2.siddhi.core.event.ComplexEvent.Type.RESET) { \n");
            deserializeBuffer.append("        org.wso2.siddhi.core.event.stream.StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            deserializeBuffer.append("        borrowedEvent.setType(type);      \n");
            deserializeBuffer.append("        long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes \n");
            deserializeBuffer.append("        borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes \n");
            
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                case BOOL:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getShort(); \n");
                    break;
                case INT:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getInt(); \n");
                    break;
                case LONG:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getLong(); \n");
                    break;
                case FLOAT:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getFloat(); \n");
                    break;
                case DOUBLE:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getDouble(); \n");
                    break;
                case STRING:
                    deserializeBuffer.append("                short length = inputEventBuffer.getShort(); \n");
                    deserializeBuffer.append("                inputEventBuffer.get(preAllocatedByteArray, 0, ").append(attrib.length).append("); \n");
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = new String(preAllocatedByteArray, 0, length);  \n");
                    break;
                }
            }

            deserializeBuffer.append("        System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, ").append(index).append("); \n");
            deserializeBuffer.append("        for(java.util.Iterator<org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor> i = attributeProcessorList.iterator(); i.hasNext(); ) { \n");
            deserializeBuffer.append("            org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor a = i.next(); \n");
            deserializeBuffer.append("            a.process(borrowedEvent); \n");
            deserializeBuffer.append("        } \n");
            deserializeBuffer.append("        if (workerfirstEvent == null) { \n");
            deserializeBuffer.append("            workerfirstEvent = borrowedEvent; \n");
            deserializeBuffer.append("            workerLastEvent = workerfirstEvent; \n");
            deserializeBuffer.append("        } else { \n");
            deserializeBuffer.append("            workerLastEvent.setNext(borrowedEvent); \n");
            deserializeBuffer.append("            workerLastEvent = borrowedEvent; \n");
            deserializeBuffer.append("        } \n");
            deserializeBuffer.append("        indexInsideSegment++; \n");
            deserializeBuffer.append("        indexInsideSegment = indexInsideSegment % segmentEventCount; \n");
            deserializeBuffer.append("    } else if (type == org.wso2.siddhi.core.event.ComplexEvent.Type.RESET) { \n");
            deserializeBuffer.append("        outputEventBuffer.position( \n");
            deserializeBuffer.append("                outputEventBuffer.position() +  \n");
            deserializeBuffer.append("                ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes())  \n");
            deserializeBuffer.append("                - 2); \n");
            deserializeBuffer.append("        resultsIndex = ((segIdx + 1) * segmentEventCount) - 1; \n");
            deserializeBuffer.append("        indexInsideSegment = 0; \n");
            deserializeBuffer.append("    } \n");
            deserializeBuffer.append("} \n");

            deserializeBuffer.append("}");

            CtMethod deserializeMethod = CtNewMethod.make(deserializeBuffer.toString(), gpuJoinQuerySelectorClass);
            gpuJoinQuerySelectorClass.addMethod(deserializeMethod);

            gpuQuerySelector = (GpuJoinQuerySelector)gpuJoinQuerySelectorClass.toClass()
                    .getConstructor(String.class, Selector.class, Boolean.class, Boolean.class, ExecutionPlanContext.class, String.class)
                    .newInstance(id, selector, currentOn, expiredOn, executionPlanContext, classId);
              
            
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
    
    public static GpuFilterQuerySelector getGpuFilterQuerySelector(String classId, GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext) {
        
        GpuFilterQuerySelector gpuQuerySelector = null;
        try {
            ClassPool pool = ClassPool.getDefault();
            
            Random r = new Random(System.nanoTime());
            String className = "GpuFilterQuerySelector" + classId + Integer.toString(r.nextInt());
            String fqdn = "org.wso2.siddhi.core.gpu.query.selector.gen." + className;
            CtClass gpuFilterQuerySelectorClass = pool.makeClass(fqdn);
            final CtClass superClass = pool.get( "org.wso2.siddhi.core.gpu.query.selector.GpuFilterQuerySelector" );
            gpuFilterQuerySelectorClass.setSuperclass(superClass);
            gpuFilterQuerySelectorClass.setModifiers( Modifier.PUBLIC );
            
            // public GpuFilterQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext, String queryName)
            
            StringBuffer constructor = new StringBuffer();
            constructor.append("public ").append(className).append("(String id, ")
                .append("org.wso2.siddhi.query.api.execution.query.selection.Selector selector, ")
                .append("boolean currentOn, ")
                .append("boolean expiredOn, ")
                .append("org.wso2.siddhi.core.config.ExecutionPlanContext executionPlanContext, ")
                .append("String queryName) {");
            constructor.append("   super(id, selector, currentOn, expiredOn, executionPlanContext, queryName); ");
            constructor.append("}");
            
            CtConstructor ctConstructor = CtNewConstructor.make(constructor.toString(), gpuFilterQuerySelectorClass);
            gpuFilterQuerySelectorClass.addConstructor(ctConstructor);
            
            // -- protected void deserialize(int eventCount) --
            StringBuilder deserializeBuffer = new StringBuilder();

            deserializeBuffer.append("protected void deserialize(int eventCount) { ");

            deserializeBuffer.append("for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) { \n");
            deserializeBuffer.append("    int matched = outputEventIndexBuffer.get(); \n");
            deserializeBuffer.append("    if (matched >= 0) { \n");
            deserializeBuffer.append("        org.wso2.siddhi.core.event.stream.StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            deserializeBuffer.append("        org.wso2.siddhi.core.event.ComplexEvent.Type type = eventTypes[inputEventBuffer.getShort()]; \n");
            deserializeBuffer.append("        long sequence = inputEventBuffer.getLong(); \n");
            deserializeBuffer.append("        long timestamp = inputEventBuffer.getLong(); \n");
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                case BOOL:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getShort(); \n");
                    break;
                case INT:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getInt(); \n");
                    break;
                case LONG:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getLong(); \n");
                    break;
                case FLOAT:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getFloat(); \n");
                    break;
                case DOUBLE:
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = inputEventBuffer.getDouble(); \n");
                    break;
                case STRING:
                    deserializeBuffer.append("                short length = inputEventBuffer.getShort(); \n");
                    deserializeBuffer.append("                inputEventBuffer.get(preAllocatedByteArray, 0, ").append(attrib.length).append("); \n");
                    deserializeBuffer.append("                attributeData[").append(index++).append("] = new String(preAllocatedByteArray, 0, length);  \n");
                    break;
                }
            }
            deserializeBuffer.append("        streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); \n");
            deserializeBuffer.append("        for(java.util.Iterator<org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor> i = attributeProcessorList.iterator(); i.hasNext(); ) { \n");
            deserializeBuffer.append("            org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor a = i.next(); \n");
            deserializeBuffer.append("            a.process(borrowedEvent); \n");
            deserializeBuffer.append("        } \n");
            deserializeBuffer.append("        if (firstEvent == null) { \n");
            deserializeBuffer.append("            firstEvent = borrowedEvent; \n");
            deserializeBuffer.append("            lastEvent = firstEvent; \n");
            deserializeBuffer.append("        } else { \n");
            deserializeBuffer.append("            lastEvent.setNext(borrowedEvent); \n");
            deserializeBuffer.append("            lastEvent = borrowedEvent; \n");
            deserializeBuffer.append("        } \n");
            deserializeBuffer.append("    } else { \n");
            deserializeBuffer.append("        inputEventBuffer.position(inputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes()); \n");
            deserializeBuffer.append("    } \n");
            deserializeBuffer.append("} \n");

            deserializeBuffer.append("}");

            CtMethod deserializeMethod = CtNewMethod.make(deserializeBuffer.toString(), gpuFilterQuerySelectorClass);
            gpuFilterQuerySelectorClass.addMethod(deserializeMethod);

            gpuQuerySelector = (GpuFilterQuerySelector)gpuFilterQuerySelectorClass.toClass()
                    .getConstructor(String.class, Selector.class, Boolean.class, Boolean.class, ExecutionPlanContext.class, String.class)
                    .newInstance(id, selector, currentOn, expiredOn, executionPlanContext, classId);
              
            
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
    
    public static GpuQuerySelector getGpuQuerySelector(String classId, GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext) {
        
        GpuQuerySelector gpuQuerySelector = null;
        try {
            
            ClassPool pool = ClassPool.getDefault();
            
            Random r = new Random(System.nanoTime());
            String className = "GpuQuerySelector" + classId + Integer.toString(r.nextInt());
            String fqdn = "org.wso2.siddhi.core.gpu.query.selector.gen." + className;
            CtClass gpuQuerySelectorClass = pool.makeClass(fqdn);
            final CtClass superClass = pool.get( "org.wso2.siddhi.core.gpu.query.selector.GpuQuerySelector" );
            gpuQuerySelectorClass.setSuperclass(superClass);
            gpuQuerySelectorClass.setModifiers( Modifier.PUBLIC );
            
            // public GpuQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext, String queryName)
            
            StringBuffer constructor = new StringBuffer();
            constructor.append("public ").append(className).append("(String id, ")
                .append("org.wso2.siddhi.query.api.execution.query.selection.Selector selector, ")
                .append("boolean currentOn, ")
                .append("boolean expiredOn, ")
                .append("org.wso2.siddhi.core.config.ExecutionPlanContext executionPlanContext, ")
                .append("String queryName) {");
            constructor.append("   super(id, selector, currentOn, expiredOn, executionPlanContext, queryName); ");
            constructor.append("}");
            
            CtConstructor ctConstructor = CtNewConstructor.make(constructor.toString(), gpuQuerySelectorClass);
            gpuQuerySelectorClass.addConstructor(ctConstructor);
            
            // -- protected void deserialize(int eventCount) --
            StringBuilder deserializeBuffer = new StringBuilder();

            deserializeBuffer.append("protected void deserialize(int eventCount) { ");

            deserializeBuffer.append("for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) { \n");
            deserializeBuffer.append("    org.wso2.siddhi.core.event.ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()];  \n");
            deserializeBuffer.append("    if(type != org.wso2.siddhi.core.event.ComplexEvent.Type.NONE) { \n");
            deserializeBuffer.append("        org.wso2.siddhi.core.event.stream.StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            deserializeBuffer.append("        long sequence = outputEventBuffer.getLong(); \n");
            deserializeBuffer.append("        long timestamp = outputEventBuffer.getLong(); \n");
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                    case BOOL:
                        deserializeBuffer.append("                attributeData[").append(index++).append("] = outputEventBuffer.getShort(); \n");
                        break;
                    case INT:
                        deserializeBuffer.append("                attributeData[").append(index++).append("] = outputEventBuffer.getInt(); \n");
                        break;
                    case LONG:
                        deserializeBuffer.append("                attributeData[").append(index++).append("] = outputEventBuffer.getLong(); \n");
                        break;
                    case FLOAT:
                        deserializeBuffer.append("                attributeData[").append(index++).append("] = outputEventBuffer.getFloat(); \n");
                        break;
                    case DOUBLE:
                        deserializeBuffer.append("                attributeData[").append(index++).append("] = outputEventBuffer.getDouble(); \n");
                        break;
                    case STRING:
                        deserializeBuffer.append("                short length = outputEventBuffer.getShort(); \n");
                        deserializeBuffer.append("                outputEventBuffer.get(preAllocatedByteArray, 0, ").append(attrib.length).append("); \n");
                        deserializeBuffer.append("                attributeData[").append(index++).append("] = new String(preAllocatedByteArray, 0, length); \n");
                        break;
                }
            }
            deserializeBuffer.append("        streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); \n");

//            deserializeBuffer.append("        for (AttributeProcessor attributeProcessor : attributeProcessorList) { \n");
//            deserializeBuffer.append("            attributeProcessor.process(borrowedEvent); \n");
//            deserializeBuffer.append("        } \n");
            
            deserializeBuffer.append("        for(java.util.Iterator<org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor> i = attributeProcessorList.iterator(); i.hasNext(); ) { \n");
            deserializeBuffer.append("            org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor a = i.next(); \n");
            deserializeBuffer.append("            a.process(borrowedEvent); \n");
            deserializeBuffer.append("        } \n");
            
            deserializeBuffer.append("        if (workerfirstEvent == null) { \n");
            deserializeBuffer.append("            workerfirstEvent = borrowedEvent; \n");
            deserializeBuffer.append("            workerLastEvent = workerfirstEvent; \n");
            deserializeBuffer.append("        } else { \n");
            deserializeBuffer.append("            workerLastEvent.setNext(borrowedEvent); \n");
            deserializeBuffer.append("            workerLastEvent = borrowedEvent; \n");
            deserializeBuffer.append("        } \n");
            deserializeBuffer.append("    } else { \n");
            deserializeBuffer.append("        outputEventBuffer.position(outputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2); \n");
            deserializeBuffer.append("    }    \n");
            deserializeBuffer.append("} \n");

            deserializeBuffer.append("}");

            CtMethod deserializeMethod = CtNewMethod.make(deserializeBuffer.toString(), gpuQuerySelectorClass);
            gpuQuerySelectorClass.addMethod(deserializeMethod);

            gpuQuerySelector = (GpuQuerySelector)gpuQuerySelectorClass.toClass()
                    .getConstructor(String.class, Selector.class, Boolean.class, Boolean.class, ExecutionPlanContext.class, String.class)
                    .newInstance(id, selector, currentOn, expiredOn, executionPlanContext, classId);
                       
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
            
            querySelector = getGpuJoinQuerySelector(gpuQueryContext.getQueryName(), gpuQueryContext.getOutputStreamMetaEvent(), 
                    id, selector, currentOn, expiredOn, executionPlanContext);
            
            if(querySelector == null) {
                querySelector = new GpuJoinQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext, gpuQueryContext.getQueryName());
            }
        
        } else if (streamRuntime instanceof GpuStreamRuntime){
            
            List<SiddhiGpu.GpuProcessor> gpuProcessors = 
                    ((GpuProcessStreamReceiver)((GpuStreamRuntime) streamRuntime).getProcessStreamReceiver()).getGpuProcessors();

            SiddhiGpu.GpuProcessor lastGpuProcessor = gpuProcessors.get(gpuProcessors.size() - 1);
            if(lastGpuProcessor != null) {
                if(lastGpuProcessor instanceof SiddhiGpu.GpuFilterProcessor) {
                    
                    querySelector = getGpuFilterQuerySelector(gpuQueryContext.getQueryName(), gpuQueryContext.getOutputStreamMetaEvent(), 
                            id, selector, currentOn, expiredOn, executionPlanContext);
                    
                    if(querySelector == null) {
                        querySelector = new GpuFilterQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext, gpuQueryContext.getQueryName());
                    }
                } else {
                    querySelector = getGpuQuerySelector(gpuQueryContext.getQueryName(), gpuQueryContext.getOutputStreamMetaEvent(), 
                            id, selector, currentOn, expiredOn, executionPlanContext);
                    
                    if(querySelector == null) {
                        querySelector = new GpuQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext, gpuQueryContext.getQueryName());
                    }
                }
            } else {
                querySelector = getGpuQuerySelector(gpuQueryContext.getQueryName(), gpuQueryContext.getOutputStreamMetaEvent(), 
                        id, selector, currentOn, expiredOn, executionPlanContext);
                
                if(querySelector == null) {
                    querySelector = new GpuQuerySelector(id, selector, currentOn, expiredOn, executionPlanContext, gpuQueryContext.getQueryName());
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
