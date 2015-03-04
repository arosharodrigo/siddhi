package org.wso2.siddhi.core.gpu.config;

import java.util.List;

import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.query.api.annotation.Annotation;
import org.wso2.siddhi.query.api.annotation.Element;
import org.wso2.siddhi.query.api.exception.DuplicateAnnotationException;
import org.wso2.siddhi.query.api.util.AnnotationHelper;

public class GpuQueryContext {

    private Integer threadsPerBlock;
    private String stringAttributeSizes;
    private Integer cudaDeviceId;
    private String queryName;
    private int inputEventBufferSize;
    private Integer eventBatchMaximumSize;
    private Integer eventBatchMinimumSize;
    private Integer perfromanceCalculateBatchCount;
    private boolean batchSoftScheduling;
    private Integer threadWorkSize;
    private GpuMetaStreamEvent inputStreamMetaEvent;
    private GpuMetaStreamEvent outputStreamMetaEvent;
    
    public GpuQueryContext(List<Annotation> annotationList) {
        threadsPerBlock = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_BLOCK_SIZE, annotationList);

        stringAttributeSizes = getAnnotationStringValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_STRING_SIZES, annotationList);
        
        cudaDeviceId = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_CUDA_DEVICE, annotationList);

        queryName = getAnnotationStringValue(SiddhiConstants.ANNOTATION_INFO, 
                SiddhiConstants.ANNOTATION_ELEMENT_INFO_NAME, annotationList);
        
        perfromanceCalculateBatchCount = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_PERFORMANCE, 
                SiddhiConstants.ANNOTATION_ELEMENT_PERFORMANCE_CALC_BATCH_COUNT, annotationList);
        
        eventBatchMaximumSize = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_BATCH_MAX_SIZE, annotationList);
        
        eventBatchMinimumSize = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_BATCH_MIN_SIZE, annotationList);
        
        threadWorkSize = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_THREAD_WORK_SIZE, annotationList);
        
        String gpuBatchScheludeString = getAnnotationStringValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_BATCH_SCHEDULE, annotationList);

        if(threadsPerBlock == null) {
            threadsPerBlock = new Integer(128);
        }

        if(cudaDeviceId == null) {
            cudaDeviceId = new Integer(0); //default CUDA device 
        }
        
        if(perfromanceCalculateBatchCount == null) {
            perfromanceCalculateBatchCount = 100000;
        }
        
        batchSoftScheduling = true;
        if(gpuBatchScheludeString != null && gpuBatchScheludeString.compareTo("hard") == 0) {
            batchSoftScheduling = false;
        }
     
        inputEventBufferSize = 1024;
        
        if(eventBatchMinimumSize == null) {
            eventBatchMinimumSize = 1;
        }
        
        if(threadWorkSize == null) {
            threadWorkSize = 0;
        }
            
        this.outputStreamMetaEvent = null;
        this.inputStreamMetaEvent = null;
    }

    public boolean isBatchSoftScheduling() {
        return batchSoftScheduling;
    }

    public void setBatchSoftScheduling(boolean batchSoftScheduling) {
        this.batchSoftScheduling = batchSoftScheduling;
    }

    public int getThreadsPerBlock() {
        return threadsPerBlock;
    }

    public void setThreadsPerBlock(int eventsPerBlock) {
        this.threadsPerBlock = eventsPerBlock;
    }

    public String getStringAttributeSizes() {
        return stringAttributeSizes;
    }

    public void setStringAttributeSizes(String stringAttributeSizes) {
        this.stringAttributeSizes = stringAttributeSizes;
    }

    public int getCudaDeviceId() {
        return cudaDeviceId;
    }

    public void setCudaDeviceId(int cudaDeviceId) {
        this.cudaDeviceId = cudaDeviceId;
    }

    public String getQueryName() {
        return queryName;
    }

    public void setQueryName(String queryName) {
        this.queryName = queryName;
    }
    
    private String getAnnotationStringValue(String annotationName,
            String elementName, List<Annotation> annotationList) {
        String value = null;
        try  {
            Element element;
            element = AnnotationHelper.getAnnotationElement(annotationName,
                    elementName, annotationList);
            if (element != null) {
                value = element.getValue();
            }
        } catch (DuplicateAnnotationException e) {
        }
        return value;
    }

    private boolean getAnnotationBooleanValue(String annotationName, String elementName,
            List<Annotation> annotationList) {
        Boolean value = false;
        try {
            Element element;
            element = AnnotationHelper.getAnnotationElement(annotationName,
                    elementName, annotationList);
            if (element != null) {
                value = SiddhiConstants.TRUE.equalsIgnoreCase(element
                        .getValue());
            }
        } catch (DuplicateAnnotationException e) {
        }
        return value;
    }

    private Integer getAnnotationIntegerValue(String annotationName, String elementName,
            List<Annotation> annotationList) {
        Integer value = null;
        try {
            Element element;
            element = AnnotationHelper.getAnnotationElement(annotationName,
                    elementName, annotationList);
            if (element != null) {
                value = Integer.parseInt(element.getValue());
            }
        } catch (DuplicateAnnotationException e) {
        }
        return value;
    }

    public int getInputEventBufferSize() {
        return inputEventBufferSize;
    }

    public void setInputEventBufferSize(int getInputEventBufferSize) {
        this.inputEventBufferSize = getInputEventBufferSize;
    }
    
    public int getPerfromanceCalculateBatchCount() {
        return perfromanceCalculateBatchCount;
    }

    public void setPerfromanceCalculateBatchCount(int perfromanceCalculateBatchCount) {
        this.perfromanceCalculateBatchCount = perfromanceCalculateBatchCount;
    }

    public int getEventBatchMaximumSize() {
        if(eventBatchMaximumSize == null)
            return inputEventBufferSize;
        return eventBatchMaximumSize;
    }

    public void setEventBatchMaximumSize(int eventBatchSize) {
        this.eventBatchMaximumSize = eventBatchSize;
    }

    public int getEventBatchMinimumSize() {
        return eventBatchMinimumSize;
    }

    public void setEventBatchMinimumSize(Integer eventBatchMinimumSize) {
        this.eventBatchMinimumSize = eventBatchMinimumSize;
    }

    public int getThreadWorkSize() {
        return threadWorkSize;
    }

    public void setThreadWorkSize(int threadWorkSize) {
        this.threadWorkSize = threadWorkSize;
    }

    public GpuMetaStreamEvent getOutputStreamMetaEvent() {
        return outputStreamMetaEvent;
    }

    public void setOutputStreamMetaEvent(GpuMetaStreamEvent outputStreamMetaEvent) {
        this.outputStreamMetaEvent = outputStreamMetaEvent;
    }

    public GpuMetaStreamEvent getInputStreamMetaEvent() {
        return inputStreamMetaEvent;
    }

    public void setInputStreamMetaEvent(GpuMetaStreamEvent inputStreamMetaEvent) {
        this.inputStreamMetaEvent = inputStreamMetaEvent;
    }
}
