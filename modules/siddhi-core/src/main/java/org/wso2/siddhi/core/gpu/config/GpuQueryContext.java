package org.wso2.siddhi.core.gpu.config;

import java.util.List;

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
    
    public GpuQueryContext(List<Annotation> annotationList) {
        threadsPerBlock = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_BLOCK_SIZE, annotationList);

        stringAttributeSizes = getAnnotationStringValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_STRING_SIZES, annotationList);
        
        cudaDeviceId = getAnnotationIntegerValue(SiddhiConstants.ANNOTATION_GPU, 
                SiddhiConstants.ANNOTATION_ELEMENT_GPU_CUDA_DEVICE, annotationList);

        queryName = getAnnotationStringValue(SiddhiConstants.ANNOTATION_INFO, 
                SiddhiConstants.ANNOTATION_ELEMENT_INFO_NAME, annotationList);

        if(threadsPerBlock == null) {
            threadsPerBlock = new Integer(128);
        }

        if(cudaDeviceId == null) {
            cudaDeviceId = new Integer(0); //default CUDA device 
        }
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
}
