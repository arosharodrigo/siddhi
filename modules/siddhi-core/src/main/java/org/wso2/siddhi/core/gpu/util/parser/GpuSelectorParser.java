package org.wso2.siddhi.core.gpu.util.parser;

import java.util.ArrayList;
import java.util.List;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.MetaComplexEvent;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.state.MetaStateEventAttribute;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.ConditionExpressionExecutor;
import org.wso2.siddhi.core.gpu.config.GpuQueryContext;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.input.stream.GpuStreamRuntime;
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
    public static QuerySelector parse(Selector selector, OutputStream outStream, ExecutionPlanContext executionPlanContext,
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
        QuerySelector querySelector = new QuerySelector(id, selector, currentOn, expiredOn, executionPlanContext);
        querySelector.setAttributeProcessorList(getAttributeProcessors(selector, id, executionPlanContext, 
                metaComplexEvent, executors, streamRuntime, gpuQueryContext, currentOn, expiredOn));

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
