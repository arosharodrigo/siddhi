package org.wso2.siddhi.core.gpu.query.selector;

import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
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
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventPool;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverter;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverterFactory;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.util.parser.GpuSelectorParser;
import org.wso2.siddhi.core.query.selector.QuerySelector;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;
import org.wso2.siddhi.query.api.execution.query.selection.Selector;

public class GpuQuerySelector extends QuerySelector {

    private static final Logger log = Logger.getLogger(GpuQuerySelector.class);
    protected ByteBuffer outputEventBuffer;
    protected ByteBuffer inputEventBuffer; 
    protected StreamEventPool streamEventPool;
    protected MetaStreamEvent metaStreamEvent;
    protected GpuMetaStreamEvent gpuMetaStreamEvent;
    protected StreamEventConverter streamEventConverter;
    protected ComplexEventChunk outputComplexEventChunk;
    protected List<GpuEventAttribute> gpuMetaEventAttributeList;
    
    protected Object attributeData[];
    protected ComplexEvent.Type eventTypes[]; 
    protected byte preAllocatedByteArray[];
    protected StreamEvent firstEvent;
    protected StreamEvent lastEvent;
    
    protected StreamEvent workerfirstEvent;
    protected StreamEvent workerLastEvent;
    
    protected int workerSize;
    
    protected ExecutorService executorService;
    protected GpuQuerySelectorWorker workers[];
    protected Future futures[];
    
    public GpuQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext) {
        super(id, selector, currentOn, expiredOn, executionPlanContext);
        this.eventTypes = ComplexEvent.Type.values();
        this.outputComplexEventChunk = new ComplexEventChunk<StreamEvent>();
        
        this.firstEvent = null;
        this.lastEvent = null;
        
        this.outputEventBuffer = null;
        this.inputEventBuffer = null;
        this.streamEventPool = null;
        this.metaStreamEvent = null;
        this.gpuMetaStreamEvent = null;
        this.streamEventConverter = null;
        this.attributeData = null;
        this.preAllocatedByteArray = null;
        this.gpuMetaEventAttributeList = null;
        
        this.executorService = null;
        this.workers = null;
        this.futures = null;
        
        this.workerfirstEvent = null;
        this.workerLastEvent = null;
    }
    
    @Override
    public void process(ComplexEventChunk complexEventChunk) {
        complexEventChunk.reset();

        if (log.isTraceEnabled()) {
            log.trace("event is processed by selector " + id + this);
        }

        while (complexEventChunk.hasNext()) {       //todo optimize
            ComplexEvent event = complexEventChunk.next();
            eventPopulator.populateStateEvent(event);

            if (event.getType() == StreamEvent.Type.CURRENT || event.getType() == StreamEvent.Type.EXPIRED) {

                //TODO: have to change for windows
                for (AttributeProcessor attributeProcessor : attributeProcessorList) {
                    attributeProcessor.process(event);
                }

            } else {
                complexEventChunk.remove();
            }

        }

        if (complexEventChunk.getFirst() != null) {
            outputRateLimiter.process(complexEventChunk);
        }
    }
    
    protected void deserialize(int eventCount) {
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {

            ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()];
            
            if(type != Type.NONE) {
                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
                
                long sequence = outputEventBuffer.getLong();
                long timestamp = outputEventBuffer.getLong();

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

                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
                //log.debug("Converted event " + borrowedEvent.toString());
                
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
                
            } else {
                outputEventBuffer.position(outputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2);
            }
            
        }
    }
       
    public void process(int eventCount) {
        outputEventBuffer.position(0);
        inputEventBuffer.position(0);
        
        int workSize = eventCount / workerSize;
        int remainWork = eventCount % workSize;
        
        for(int i=0; i<workerSize; ++i) {
            ByteBuffer dup = outputEventBuffer.duplicate();
            dup.order(outputEventBuffer.order());
            workers[i].setOutputEventBuffer(dup);
            workers[i].setBufferStartPosition(i * workSize * gpuMetaStreamEvent.getEventSizeInBytes());
            workers[i].setEventCount(workSize);
            
            futures[i] = executorService.submit(workers[i]);
        }
        
        // do remaining task
        outputEventBuffer.position(workSize * workerSize * gpuMetaStreamEvent.getEventSizeInBytes());
        deserialize(remainWork);
        
//        for (int resultsIndex = 0; resultsIndex < remainWork; ++resultsIndex) {
//
//            ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()];
//            
//            if(type != Type.NONE) {
//                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
//                
//                long sequence = outputEventBuffer.getLong();
//                long timestamp = outputEventBuffer.getLong();
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
//                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
//                //log.debug("Converted event " + borrowedEvent.toString());
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
//            } else {
//                outputEventBuffer.position(outputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2);
//            }
//            
//        }


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
        
        // all workers complete, 
        
        // call output rate limiter
        if (firstEvent != null) {
            outputComplexEventChunk.add(firstEvent);
            outputRateLimiter.process(outputComplexEventChunk);
        }
        firstEvent = null;
        lastEvent = null;
        outputComplexEventChunk.clear();
    }
     
    public QuerySelector clone(String key) {
        GpuQuerySelector clonedQuerySelector = GpuSelectorParser.getGpuQuerySelector(this.gpuMetaStreamEvent, id + key, selector, currentOn, expiredOn, executionPlanContext);
        
        if(clonedQuerySelector == null) {
            clonedQuerySelector = new GpuQuerySelector(id + key, selector, currentOn, expiredOn, executionPlanContext);
        }
        List<AttributeProcessor> clonedAttributeProcessorList = new ArrayList<AttributeProcessor>();
        for (AttributeProcessor attributeProcessor : attributeProcessorList) {
            clonedAttributeProcessorList.add(attributeProcessor.cloneProcessor());
        }
        clonedQuerySelector.attributeProcessorList = clonedAttributeProcessorList;
        clonedQuerySelector.eventPopulator = eventPopulator;
        clonedQuerySelector.outputRateLimiter = outputRateLimiter;
        
        return clonedQuerySelector;
    }

    public ByteBuffer getOutputEventBuffer() {
        return outputEventBuffer;
    }

    public void setOutputEventBuffer(ByteBuffer outputEventBuffer) {
        this.outputEventBuffer = outputEventBuffer;
    }

    public ByteBuffer getInputEventBuffer() {
        return inputEventBuffer;
    }

    public void setInputEventBuffer(ByteBuffer inputByteBuffer) {
        this.inputEventBuffer = inputByteBuffer;
    }
    
    public StreamEventPool getStreamEventPool() {
        return streamEventPool;
    }

    public void setStreamEventPool(StreamEventPool streamEventPool) {
        this.streamEventPool = streamEventPool;
    }
    
    public MetaStreamEvent getMetaStreamEvent() {
        return metaStreamEvent;
    }

    public void setMetaStreamEvent(MetaStreamEvent metaStreamEvent) {
        this.metaStreamEvent = metaStreamEvent;
        
        streamEventConverter = StreamEventConverterFactory.constructEventConverter(metaStreamEvent);
    }

    public GpuMetaStreamEvent getGpuMetaStreamEvent() {
        return gpuMetaStreamEvent;
    }

    public void setGpuMetaStreamEvent(GpuMetaStreamEvent gpuMetaStreamEvent) {
        this.gpuMetaStreamEvent = gpuMetaStreamEvent;
        this.gpuMetaEventAttributeList = gpuMetaStreamEvent.getAttributes();
        
        int maxStringLength = 0;
        
        attributeData = new Object[gpuMetaEventAttributeList.size()];
        int index = 0;
        for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
            switch(attrib.type) {
            case BOOL:
                attributeData[index++] = new Boolean(false);
                break;
            case INT:
                attributeData[index++] = new Integer(0);
                break;
            case LONG:
                attributeData[index++] = new Long(0);
                break;
            case FLOAT:
                attributeData[index++] = new Float(0);
                break;
            case DOUBLE:
                attributeData[index++] = new Double(0);
                break;
            case STRING:
                attributeData[index++] = new String();
                maxStringLength = (attrib.length > maxStringLength ? attrib.length : maxStringLength);
                break;
            }
        }
        
        preAllocatedByteArray = new byte[maxStringLength + 1];
    }
    
    public void setWorkerSize(int workerSize) {
        if(workerSize > 0) {
            this.workerSize = workerSize;
            this.executorService = Executors.newFixedThreadPool(workerSize);
            this.workers = new GpuQuerySelectorWorker[workerSize];
            this.futures = new Future[workerSize];

            for(int i=0; i<workerSize; ++i) {

                this.workers[i] = getGpuQuerySelectorWorker(gpuMetaStreamEvent, 
                        id + "_" + Integer.toString(i), streamEventPool.clone(), streamEventConverter);
                
                if(this.workers[i] == null) {
                    this.workers[i] = new GpuQuerySelectorWorker(id + "_" + Integer.toString(i), streamEventPool.clone(), streamEventConverter); 
                }
                
                this.workers[i].setAttributeProcessorList(attributeProcessorList);// TODO: attributeProcessorList should be cloned
                this.workers[i].setGpuMetaStreamEvent(gpuMetaStreamEvent);
            }
        }
    }
    
    private GpuQuerySelectorWorker getGpuQuerySelectorWorker(
            GpuMetaStreamEvent gpuMetaStreamEvent,
            String id, StreamEventPool streamEventPool, 
            StreamEventConverter streamEventConverter) {
        
        GpuQuerySelectorWorker gpuQuerySelectorWorker = null;
        try {
            CtClass ctClass = ClassPool.getDefault().get("org.wso2.siddhi.core.gpu.query.selector.GpuQuerySelectorWorker");
            CtMethod method = ctClass.getDeclaredMethod("run");
            
            StringBuffer content = new StringBuffer();
            content.append("{\n ");
            
            content.append("for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) { \n");
            content.append("    ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()]; \n");
            content.append("    if(type != Type.NONE) { \n");
            content.append("        StreamEvent borrowedEvent = streamEventPool.borrowEvent(); \n");
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
            content.append("        if (firstEvent == null) { \n");
            content.append("            firstEvent = borrowedEvent; \n");
            content.append("            lastEvent = firstEvent; \n");
            content.append("        } else { \n");
            content.append("            lastEvent.setNext(borrowedEvent); \n");
            content.append("            lastEvent = borrowedEvent; \n");
            content.append("        } \n");
            content.append("    } else { \n");
            content.append("        outputEventBuffer.position(outputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2); \n");
            content.append("    } \n");
            content.append("} \n");

            content.append("} ");
            
            method.setBody(content.toString());
//            ctClass.writeFile();
            gpuQuerySelectorWorker = (GpuQuerySelectorWorker)ctClass.toClass().getConstructor(
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
