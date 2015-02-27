package org.wso2.siddhi.core.gpu.query.selector;

import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javassist.CannotCompileException;
import javassist.ClassPool;
import javassist.CtClass;
import javassist.CtMethod;
import javassist.NotFoundException;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventPool;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverter;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.util.parser.GpuSelectorParser;
import org.wso2.siddhi.core.query.selector.QuerySelector;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;
import org.wso2.siddhi.query.api.execution.query.selection.Selector;

public class GpuJoinQuerySelector extends GpuQuerySelector {
    private static final Logger log = Logger.getLogger(GpuJoinQuerySelector.class);
    private int segmentEventCount;
    private int threadWorkSize;
    private int segmentsPerWorker;
    
    public GpuJoinQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext) {
        super(id, selector, currentOn, expiredOn, executionPlanContext);
        this.segmentEventCount = -1;
        this.setThreadWorkSize(0);
        this.segmentsPerWorker = 0;
    }
  
    protected void deserialize(int eventCount) {
        int workSize = segmentsPerWorker * segmentEventCount;
        int indexInsideSegment = 0;
        int segIdx = 0;
        ComplexEvent.Type type;
        
        for (int resultsIndex = workerSize * workSize; resultsIndex < eventCount; ++resultsIndex) {

            segIdx = resultsIndex / segmentEventCount;

            type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes

//            log.debug("<" + id + " @ GpuJoinQuerySelector> process eventIndex=" + resultsIndex + " type=" + type 
//                    + " segIdx=" + segIdx + " segInternalIdx=" + indexInsideSegment);

            if(type != Type.NONE && type != Type.RESET) {
                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
                borrowedEvent.setType(type);
                
                long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes
                borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
                    switch(attrib.type) {
                    case BOOL:
                        attributeData[index++] = outputEventBuffer.getShort();
                        break;
                    case INT:
                        attributeData[index++] = outputEventBuffer.getInt();
                        break;
                    case LONG:
                        attributeData[index++] = outputEventBuffer.getLong();
                        break;
                    case FLOAT:
                        attributeData[index++] = outputEventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        attributeData[index++] = outputEventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = outputEventBuffer.getShort();
                        outputEventBuffer.get(preAllocatedByteArray, 0, attrib.length);
                        attributeData[index++] = new String(preAllocatedByteArray, 0, length); // TODO: avoid allocation
                        break;
                    }
                }
                
                //XXX: assume always ZeroStreamEventConvertor
                //                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); 
                System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index);

//                log.debug("<" + id + " @ GpuJoinQuerySelector> Converted event " + resultsIndex + " : [" + sequence + "] " + borrowedEvent.toString());

                // call actual select operations
                for (AttributeProcessor attributeProcessor : attributeProcessorList) {
                    attributeProcessor.process(borrowedEvent);
                }

                // add event to current list
                if (workerfirstEvent == null) {
                    workerfirstEvent = borrowedEvent;
                    workerLastEvent = workerfirstEvent;
                } else {
                    workerLastEvent.setNext(borrowedEvent);
                    workerLastEvent = borrowedEvent;
                }

                indexInsideSegment++;
                indexInsideSegment = indexInsideSegment % segmentEventCount;

            } else if (type == Type.RESET){
                // skip remaining bytes in segment
//                log.debug("<" + id + " @ GpuJoinQuerySelector> Skip to next segment : CurrPos=" + 
//                        outputEventBuffer.position() + " segInternalIdx=" + indexInsideSegment);

                outputEventBuffer.position(
                        outputEventBuffer.position() + 
                        ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes()) 
                        - 2);

//                log.debug("<" + id + " @ GpuJoinQuerySelector> buffer new pos : " + outputEventBuffer.position());
                resultsIndex = ((segIdx + 1) * segmentEventCount) - 1;
                indexInsideSegment = 0;
            }
        }
    }
    
    @Override
    public void process(int eventCount) {
        outputEventBuffer.position(0);
        inputEventBuffer.position(0);

//        log.debug("<" + id + " @ GpuJoinQuerySelector> process eventCount=" + eventCount + " eventSegmentSize=" + segmentEventCount
//                + " workerSize=" + workerSize + " segmentsPerWorker=" + segmentsPerWorker);
        
        int workSize = segmentsPerWorker * segmentEventCount; // workSize should be in segment boundary
        
        for(int i=0; i<workerSize; ++i) {
            ByteBuffer dup = outputEventBuffer.duplicate();
            dup.order(outputEventBuffer.order());
            workers[i].setOutputEventBuffer(dup);
            workers[i].setBufferStartPosition(i * workSize * gpuMetaStreamEvent.getEventSizeInBytes());
            workers[i].setEventCount(workSize);
            ((GpuJoinQuerySelectorWorker)workers[i]).setSegmentEventCount(segmentEventCount);
            ((GpuJoinQuerySelectorWorker)workers[i]).setWorkStartEvent(i * workSize);
            ((GpuJoinQuerySelectorWorker)workers[i]).setWorkEndEvent((i * workSize) + workSize);
            
            futures[i] = executorService.submit(workers[i]);
        }
        
//        log.debug("<" + id + " @ GpuJoinQuerySelector> process remaining from=" + (workerSize * workSize) + " To=" + eventCount);
        // do remaining task
        outputEventBuffer.position(workerSize * workSize * gpuMetaStreamEvent.getEventSizeInBytes());
        
//        int indexInsideSegment = 0;
//        int segIdx = 0;
//        ComplexEvent.Type type;
//        for (int resultsIndex = workerSize * workSize; resultsIndex < eventCount; ++resultsIndex) {
//
//            segIdx = resultsIndex / segmentEventCount;
//
//            type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes
//
////            log.debug("<" + id + " @ GpuJoinQuerySelector> process eventIndex=" + resultsIndex + " type=" + type 
////                    + " segIdx=" + segIdx + " segInternalIdx=" + indexInsideSegment);
//
//            if(type != Type.NONE && type != Type.RESET) {
//                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
//                borrowedEvent.setType(type);
//                
//                long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes
//                borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes
//
//                int index = 0;
//                for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
//                    switch(attrib.type) {
//                    case BOOL:
//                        attributeData[index++] = outputEventBuffer.getShort();
//                        break;
//                    case INT:
//                        attributeData[index++] = outputEventBuffer.getInt();
//                        break;
//                    case LONG:
//                        attributeData[index++] = outputEventBuffer.getLong();
//                        break;
//                    case FLOAT:
//                        attributeData[index++] = outputEventBuffer.getFloat();
//                        break;
//                    case DOUBLE:
//                        attributeData[index++] = outputEventBuffer.getDouble();
//                        break;
//                    case STRING:
//                        short length = outputEventBuffer.getShort();
//                        outputEventBuffer.get(preAllocatedByteArray, 0, attrib.length);
//                        attributeData[index++] = new String(preAllocatedByteArray, 0, length); // TODO: avoid allocation
//                        break;
//                    }
//                }
//                
//                //XXX: assume always ZeroStreamEventConvertor
//                //                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); 
//                System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index);
//
////                log.debug("<" + id + " @ GpuJoinQuerySelector> Converted event " + resultsIndex + " : [" + sequence + "] " + borrowedEvent.toString());
//
//                // call actual select operations
//                for (AttributeProcessor attributeProcessor : attributeProcessorList) {
//                    attributeProcessor.process(borrowedEvent);
//                }
//
//                // add event to current list
//                if (workerfirstEvent == null) {
//                    workerfirstEvent = borrowedEvent;
//                    workerLastEvent = workerfirstEvent;
//                } else {
//                    workerLastEvent.setNext(borrowedEvent);
//                    workerLastEvent = borrowedEvent;
//                }
//
//                indexInsideSegment++;
//                indexInsideSegment = indexInsideSegment % segmentEventCount;
//
//            } else if (type == Type.RESET){
//                // skip remaining bytes in segment
////                log.debug("<" + id + " @ GpuJoinQuerySelector> Skip to next segment : CurrPos=" + 
////                        outputEventBuffer.position() + " segInternalIdx=" + indexInsideSegment);
//
//                outputEventBuffer.position(
//                        outputEventBuffer.position() + 
//                        ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes()) 
//                        - 2);
//
////                log.debug("<" + id + " @ GpuJoinQuerySelector> buffer new pos : " + outputEventBuffer.position());
//                resultsIndex = ((segIdx + 1) * segmentEventCount) - 1;
//                indexInsideSegment = 0;
//            }
//        }
        
        deserialize(eventCount);
        
        for(int i=0; i<workerSize; ++i) {
            try {
                while(futures[i].get() != null) { }

                StreamEvent workerResultsFirst = workers[i].getFirstEvent();
                StreamEvent workerResultsLast = workers[i].getLastEvent();

                if(workerResultsFirst != null) {
                    if (firstEvent != null) {
                        lastEvent.setNext(workerResultsFirst);
                        lastEvent = workerResultsLast;
                    } else {
                        firstEvent = workerResultsFirst;
                        lastEvent = workerResultsLast;
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
        
        if(workerfirstEvent != null) {
            if (firstEvent != null) {
                lastEvent.setNext(workerfirstEvent);
                lastEvent = workerLastEvent;
            } else {
                firstEvent = workerfirstEvent;
                lastEvent = workerLastEvent;
            }
        }

        //        log.debug("<" + id + " @ GpuJoinQuerySelector> Call outputRateLimiter " + outputRateLimiter);

        // call output rate limiter
        if (firstEvent != null) {
            outputComplexEventChunk.add(firstEvent);
            outputRateLimiter.process(outputComplexEventChunk);
        }
        firstEvent = null;
        lastEvent = null;
        outputComplexEventChunk.clear();
    }

    public int getSegmentEventCount() {
        return segmentEventCount;
    }

    public void setSegmentEventCount(int segmentEventCount) {
        this.segmentEventCount = segmentEventCount;
    }
    
    @Override
    public QuerySelector clone(String key) {
        GpuJoinQuerySelector clonedQuerySelector = GpuSelectorParser.getGpuJoinQuerySelector(this.gpuMetaStreamEvent, 
                id + key, selector, currentOn, expiredOn, executionPlanContext);
        
        if(clonedQuerySelector == null) {
            clonedQuerySelector = new GpuJoinQuerySelector(id + key, selector, currentOn, expiredOn, executionPlanContext);
        }
        List<AttributeProcessor> clonedAttributeProcessorList = new ArrayList<AttributeProcessor>();
        for (AttributeProcessor attributeProcessor : attributeProcessorList) {
            clonedAttributeProcessorList.add(attributeProcessor.cloneProcessor());
        }
        clonedQuerySelector.attributeProcessorList = clonedAttributeProcessorList;
        clonedQuerySelector.eventPopulator = eventPopulator;
        clonedQuerySelector.segmentEventCount = segmentEventCount;
        clonedQuerySelector.outputRateLimiter = outputRateLimiter;
        return clonedQuerySelector;
    }

    public int getThreadWorkSize() {
        return threadWorkSize;
    }

    public void setThreadWorkSize(int threadWorkSize) {
        this.threadWorkSize = threadWorkSize;
        if(threadWorkSize != 0) {
            segmentEventCount = threadWorkSize;
        }
    }
    
    public void setWorkerSize(int workerSize) {
        this.workerSize = workerSize;
        this.executorService = Executors.newFixedThreadPool(workerSize);
        this.workers = new GpuJoinQuerySelectorWorker[workerSize];
        this.futures = new Future[workerSize];
        
        for(int i=0; i<workerSize; ++i) {
            
            this.workers[i] = getGpuJoinQuerySelectorWorker(gpuMetaStreamEvent,
                    id + "_" + Integer.toString(i), streamEventPool.clone(), streamEventConverter);
            
            if(this.workers[i] == null) {
                this.workers[i] = new GpuJoinQuerySelectorWorker(id + "_" + Integer.toString(i), streamEventPool.clone(), streamEventConverter); 
            }
            
            this.workers[i].setAttributeProcessorList(attributeProcessorList); // TODO: attributeProcessorList should be cloned
            this.workers[i].setGpuMetaStreamEvent(gpuMetaStreamEvent);
        }
            
    }
    
    public void setSegmentsPerWorker(int segmentsPerWorker) {
        this.segmentsPerWorker = segmentsPerWorker;
    }
    
    private GpuJoinQuerySelectorWorker getGpuJoinQuerySelectorWorker(
            GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, StreamEventPool streamEventPool, StreamEventConverter streamEventConverter) {
        
        GpuJoinQuerySelectorWorker gpuQuerySelectorWorker = null;
        try {
            CtClass ctClass = ClassPool.getDefault().get("org.wso2.siddhi.core.gpu.query.selector.GpuJoinQuerySelectorWorker");
            CtMethod method = ctClass.getDeclaredMethod("run");
            
            StringBuffer content = new StringBuffer();
            content.append("{\n ");
            
            content.append("int indexInsideSegment = 0; \n");
            content.append("int segIdx = 0; \n");
            content.append("ComplexEvent.Type type; \n");
            content.append("for (int resultsIndex = workStartEvent; resultsIndex < workEndEvent; ++resultsIndex) { \n");
            content.append("    segIdx = resultsIndex / segmentEventCount; \n");
            content.append("    type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes \n");
            content.append("    if(type != Type.NONE && type != Type.RESET) { \n");
            content.append("        StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            content.append("        borrowedEvent.setType(type); \n");
            content.append("        long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes \n");
            content.append("        borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes \n");
            
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

            content.append("        System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index); \n");
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
            content.append("        indexInsideSegment++; \n");
            content.append("        indexInsideSegment = indexInsideSegment % segmentEventCount; \n");
            content.append("    } else if (type == Type.RESET){ \n");
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
            gpuQuerySelectorWorker = (GpuJoinQuerySelectorWorker)ctClass.toClass().getConstructor(
                    String.class, StreamEventPool.class, StreamEventConverter.class)
                    .newInstance(id, streamEventPool, streamEventConverter);
            
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
        
        return gpuQuerySelectorWorker;
        
    }
}
