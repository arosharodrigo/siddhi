package org.wso2.siddhi.core.gpu.query.input.stream.join;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.input.stream.GpuStreamRuntime;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.query.input.stream.join.JoinStreamRuntime;
import org.wso2.siddhi.core.query.input.stream.single.SingleStreamRuntime;
import org.wso2.siddhi.core.query.processor.Processor;

public class GpuJoinStreamRuntime extends JoinStreamRuntime {
    
    public GpuJoinStreamRuntime(ExecutionPlanContext executionPlanContext, MetaStateEvent metaStateEvent) {
        super(executionPlanContext, metaStateEvent);
    }
    
//    public void addRuntime(GpuStreamRuntime singleStreamRuntime) {
//        singleStreamRuntimeList.add(singleStreamRuntime);
//        
//        if(gpuQueryProcessor == null) {
////            gpuQueryProcessor = singleStreamRuntime.getProcessStreamReceiver();
//        }
//    }
//
//    @Override
//    public void setCommonProcessor(Processor commonProcessor) {
//        for (SingleStreamRuntime singleStreamRuntime : singleStreamRuntimeList) {
//            singleStreamRuntime.setCommonProcessor(commonProcessor);
//        }
//        
////        if (processorChain == null) {
////            processStreamReceiver.setNext(commonProcessor);
////        } else {
////            processStreamReceiver.setNext(processorChain);
////            processorChain.setToLast(commonProcessor);
////        }
////        
////        ((GpuProcessStreamReceiver)processStreamReceiver).setSelectProcessor(commonProcessor);
//    }
}
