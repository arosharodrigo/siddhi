package org.wso2.siddhi.core.gpu.query.selector;

import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import javassist.CannotCompileException;
import javassist.ClassPool;
import javassist.CtClass;
import javassist.CtConstructor;
import javassist.CtMethod;
import javassist.CtNewConstructor;
import javassist.CtNewMethod;
import javassist.Modifier;
import javassist.NotFoundException;

import org.apache.commons.lang3.concurrent.BasicThreadFactory;
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
    protected int segmentEventCount;
    protected int threadWorkSize;
    protected int segmentsPerWorker;
    
    public GpuJoinQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, 
            ExecutionPlanContext executionPlanContext, String queryName) {
        super(id, selector, currentOn, expiredOn, executionPlanContext, queryName);
        this.segmentEventCount = -1;
        this.setThreadWorkSize(0);
        this.segmentsPerWorker = 0;
    }
  
    public void deserialize(int eventCount) {
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
                
                Object[] outputData = borrowedEvent.getOutputData();

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
                    switch(attrib.type) {
                    case BOOL:
                        outputData[index++] = outputEventBuffer.getShort();
                        break;
                    case INT:
                        outputData[index++] = outputEventBuffer.getInt();
                        break;
                    case LONG:
                        outputData[index++] = outputEventBuffer.getLong();
                        break;
                    case FLOAT:
                        outputData[index++] = outputEventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        outputData[index++] = outputEventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = outputEventBuffer.getShort();
                        outputEventBuffer.get(preAllocatedByteArray, 0, attrib.length);
                        outputData[index++] = new String(preAllocatedByteArray, 0, length).intern(); // TODO: avoid allocation, check interns
                        break;
                    }
                }
                
                //XXX: assume always ZeroStreamEventConvertor
                //                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); 
//                System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index);

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

                processedEventCount++;
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
        processedEventCount = 0;

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
               
        deserialize(eventCount);
        
        for(int i=0; i<workerSize; ++i) {
            try {
                while(futures[i].get() != null) { }

                StreamEvent workerResultsFirst = workers[i].getFirstEvent();
                StreamEvent workerResultsLast = workers[i].getLastEvent();
                processedEventCount += workers[i].getProcessedEventCount();

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
        
        log.info("<" + queryName + "> processedEventCount=" + processedEventCount);

        // call output rate limiter
        if (firstEvent != null) {
            outputComplexEventChunk.add(firstEvent, lastEvent);
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
        GpuJoinQuerySelector clonedQuerySelector = GpuSelectorParser.getGpuJoinQuerySelector(queryName, this.gpuMetaStreamEvent, 
                id + key, selector, currentOn, expiredOn, executionPlanContext);
        
        if(clonedQuerySelector == null) {
            clonedQuerySelector = new GpuJoinQuerySelector(id + key, selector, currentOn, expiredOn, executionPlanContext, queryName);
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
        
        if(this.workerSize > 0) {
            ThreadFactory threadFactory = new BasicThreadFactory.Builder().namingPattern("gpu_join_sel_wrk_" + queryName + "-%d").build();
            this.executorService =  new ThreadPoolExecutor(workerSize, 
                    workerSize,
                    60L, TimeUnit.SECONDS,
                    new LinkedBlockingDeque<Runnable>(),
                    threadFactory); 
            this.workers = new GpuJoinQuerySelectorWorker[workerSize];
            this.futures = new Future[workerSize];

            for(int i=0; i<workerSize; ++i) {

                this.workers[i] = getGpuJoinQuerySelectorWorker(queryName, gpuMetaStreamEvent,
                        id + "_" + Integer.toString(i), streamEventPool.clone(), streamEventConverter);

                if(this.workers[i] == null) {
                    this.workers[i] = new GpuJoinQuerySelectorWorker(id + "_" + Integer.toString(i), streamEventPool.clone(), streamEventConverter); 
                }

                this.workers[i].setAttributeProcessorList(attributeProcessorList); // TODO: attributeProcessorList should be cloned
                this.workers[i].setGpuMetaStreamEvent(gpuMetaStreamEvent);
            }
        }
            
    }
    
    public void setSegmentsPerWorker(int segmentsPerWorker) {
        this.segmentsPerWorker = segmentsPerWorker;
    }
    
    private GpuJoinQuerySelectorWorker getGpuJoinQuerySelectorWorker(
            String queryName,
            GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, StreamEventPool streamEventPool, StreamEventConverter streamEventConverter) {
        
        GpuJoinQuerySelectorWorker gpuQuerySelectorWorker = null;
        try {
            ClassPool pool = ClassPool.getDefault();
            
            String className = "GpuJoinQuerySelectorWorker" + queryName + GpuSelectorParser.atomicSelectorClassId.getAndIncrement();
            String fqdn = "org.wso2.siddhi.core.gpu.query.selector.gen." + className;
            CtClass gpuJoinQuerySelectorWorkerClass = pool.makeClass(fqdn);
            final CtClass superClass = pool.get( "org.wso2.siddhi.core.gpu.query.selector.GpuJoinQuerySelectorWorker" );
            gpuJoinQuerySelectorWorkerClass.setSuperclass(superClass);
            gpuJoinQuerySelectorWorkerClass.setModifiers( Modifier.PUBLIC );
            
            log.info("[getGpuJoinQuerySelectorWorker] Class=" + className + " FQDN=" + fqdn);
            // public GpuJoinQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext, String queryName)
            
            StringBuffer constructor = new StringBuffer();
            constructor.append("public ").append(className).append("(String id, ")
                .append("org.wso2.siddhi.core.event.stream.StreamEventPool streamEventPool, ")
                .append("org.wso2.siddhi.core.event.stream.converter.StreamEventConverter streamEventConverter) {");
            constructor.append("   super(id, streamEventPool, streamEventConverter); ");
            constructor.append("}");
            
            log.debug("[getGpuJoinQuerySelectorWorker] Constructor=" + constructor.toString());
            
            CtConstructor ctConstructor = CtNewConstructor.make(constructor.toString(), gpuJoinQuerySelectorWorkerClass);
            gpuJoinQuerySelectorWorkerClass.addConstructor(ctConstructor);
            
         // -- public void run() --
            StringBuilder deserializeBuffer = new StringBuilder();

            deserializeBuffer.append("public void run() { ");
            deserializeBuffer.append("processedEventCount = 0; \n");
            deserializeBuffer.append("int indexInsideSegment = 0; \n");
            deserializeBuffer.append("int segIdx = 0; \n");
            deserializeBuffer.append("org.wso2.siddhi.core.event.ComplexEvent.Type type; \n");
            deserializeBuffer.append("for (int resultsIndex = workStartEvent; resultsIndex < workEndEvent; ++resultsIndex) { \n");
            deserializeBuffer.append("    segIdx = resultsIndex / segmentEventCount; \n");
            deserializeBuffer.append("    type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes \n");
            deserializeBuffer.append("    if(type != org.wso2.siddhi.core.event.ComplexEvent.Type.NONE && type != org.wso2.siddhi.core.event.ComplexEvent.Type.RESET) { \n");
            deserializeBuffer.append("        org.wso2.siddhi.core.event.stream.StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
            deserializeBuffer.append("        borrowedEvent.setType(type); \n");
            deserializeBuffer.append("        long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes \n");
            deserializeBuffer.append("        borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes \n");
            deserializeBuffer.append("        attributeData = borrowedEvent.getOutputData(); \n");
          
            int index = 0;
            for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                switch(attrib.type) {
                    case BOOL:
                        deserializeBuffer.append("setAttributeData(").append(index++).append(", outputEventBuffer.getShort()); \n");
                        break;
                    case INT:
                        deserializeBuffer.append("setAttributeData(").append(index++).append(", outputEventBuffer.getInt()); \n");
                        break;
                    case LONG:
                        deserializeBuffer.append("setAttributeData(").append(index++).append(", outputEventBuffer.getLong()); \n");
                        break;
                    case FLOAT:
                        deserializeBuffer.append("setAttributeData(").append(index++).append(", outputEventBuffer.getFloat()); \n");
                        break;
                    case DOUBLE:
                        deserializeBuffer.append("setAttributeData(").append(index++).append(", outputEventBuffer.getDouble()); \n");
                        break;
                    case STRING:
                        deserializeBuffer.append("short length = outputEventBuffer.getShort(); \n");
                        deserializeBuffer.append("outputEventBuffer.get(preAllocatedByteArray, 0, ").append(attrib.length).append("); \n");
                        deserializeBuffer.append("setAttributeData(").append(index++).append(", new String(preAllocatedByteArray, 0, length).intern()); \n");
                        break;
                     default:
                         break;
                }
            }

//            deserializeBuffer.append("        System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, ").append(index).append("); \n");

            deserializeBuffer.append("        java.util.Iterator i = attributeProcessorList.iterator(); \n");
            deserializeBuffer.append("        while(i.hasNext()) { \n");
            deserializeBuffer.append("            org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor a = (org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor)i.next(); \n");
            deserializeBuffer.append("            a.process(borrowedEvent); \n");
            deserializeBuffer.append("        } \n");
            
            deserializeBuffer.append("        if (firstEvent == null) { \n");
            deserializeBuffer.append("            firstEvent = borrowedEvent; \n");
            deserializeBuffer.append("            lastEvent = firstEvent; \n");
            deserializeBuffer.append("        } else { \n");
            deserializeBuffer.append("            lastEvent.setNext(borrowedEvent); \n");
            deserializeBuffer.append("            lastEvent = borrowedEvent; \n");
            deserializeBuffer.append("        } \n");
            deserializeBuffer.append("        processedEventCount++; \n");
            deserializeBuffer.append("        indexInsideSegment++; \n");
            deserializeBuffer.append("        indexInsideSegment = indexInsideSegment % segmentEventCount; \n");
            
            deserializeBuffer.append("    } else if (type == org.wso2.siddhi.core.event.ComplexEvent.Type.RESET){ \n");
            deserializeBuffer.append("        outputEventBuffer.position( \n");
            deserializeBuffer.append("                outputEventBuffer.position() +  \n");
            deserializeBuffer.append("                ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes())  \n");
            deserializeBuffer.append("                - 2); \n");
            deserializeBuffer.append("        resultsIndex = ((segIdx + 1) * segmentEventCount) - 1; \n");
            deserializeBuffer.append("        indexInsideSegment = 0; \n");
            deserializeBuffer.append("    } \n");
            deserializeBuffer.append("} \n");

            deserializeBuffer.append("}");
            
            log.debug("[getGpuJoinQuerySelectorWorker] deserialize=" + deserializeBuffer.toString());

            CtMethod deserializeMethod = CtNewMethod.make(deserializeBuffer.toString(), gpuJoinQuerySelectorWorkerClass);
            gpuJoinQuerySelectorWorkerClass.addMethod(deserializeMethod);

            gpuQuerySelectorWorker = (GpuJoinQuerySelectorWorker)gpuJoinQuerySelectorWorkerClass.toClass()
                    .getConstructor(String.class, StreamEventPool.class, StreamEventConverter.class)
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
