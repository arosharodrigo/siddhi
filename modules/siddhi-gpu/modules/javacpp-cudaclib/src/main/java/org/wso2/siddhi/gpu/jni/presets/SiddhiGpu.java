package org.wso2.siddhi.gpu.jni.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;


@Properties(target="org.wso2.siddhi.gpu.jni.SiddhiGpu", value={
    @Platform(include={"<CudaEvent.h>", "<Filter.h>", "<GpuEventConsumer.h>","<CudaKernelBase.h>", "<CudaFilterKernel.h>",
    		"<CudaSingleFilterKernel.h>"}, link={"gpueventconsumer"} ),
    @Platform(value="windows", link="gpueventconsumer") })
public class SiddhiGpu implements InfoMapper {
    public void map(InfoMap infoMap) {
    	//infoMap.put(new Info("::std::vector<int>").cast().valueTypes("StdVector"));
    }
}
