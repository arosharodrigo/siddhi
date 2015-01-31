package org.wso2.siddhi.core.gpu.query.input.stream.join;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.gpu.query.input.stream.GpuStreamRuntime;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.query.input.stream.join.JoinStreamRuntime;

public class GpuJoinStreamRuntime extends JoinStreamRuntime {
    
    private GpuQueryProcessor gpuQueryProcessor;

    public GpuJoinStreamRuntime(ExecutionPlanContext executionPlanContext, MetaStateEvent metaStateEvent) {
        super(executionPlanContext, metaStateEvent);
        gpuQueryProcessor = null;
    }
    
    public void addRuntime(GpuStreamRuntime singleStreamRuntime) {
        singleStreamRuntimeList.add(singleStreamRuntime);
        
        if(gpuQueryProcessor == null) {
//            gpuQueryProcessor = singleStreamRuntime.getProcessStreamReceiver();
        }
    }

}
