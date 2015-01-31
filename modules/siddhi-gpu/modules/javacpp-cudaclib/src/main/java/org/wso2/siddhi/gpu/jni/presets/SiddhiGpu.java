package org.wso2.siddhi.gpu.jni.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;


@Properties(target="org.wso2.siddhi.gpu.jni.SiddhiGpu", value={
    @Platform(include={"<DataTypes.h>","<Value.h>","<GpuQueryRuntime.h>","<GpuStreamProcessor.h>","<GpuProcessor.h>",
            "<GpuMetaEvent.h>","<GpuKernel.h>","<GpuFilterProcessor.h>","<GpuLengthSlidingWindowProcessor.h>",
            "<GpuJoinProcessor.h>"}, 
            link={"gpuqueryruntime"} ),
    @Platform(value="windows", link="gpuqueryruntime") })
public class SiddhiGpu implements InfoMapper {
    public void map(InfoMap infoMap) {
    	//infoMap.put(new Info("::std::vector<int>").cast().valueTypes("StdVector"));
    }
}
